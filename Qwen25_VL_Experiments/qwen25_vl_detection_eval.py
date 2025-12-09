import json
import os
import tempfile
from PIL import Image
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import random
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

class Qwen25VLEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load COCO annotations
        self.coco_gt = COCO(self.args.anno_dir)
        
        # COCO categories mapping
        self.cat_name_to_id = {cat['name']: cat['id'] for cat in self.coco_gt.dataset['categories']}
        
        # Detection question for COCO categories
        self.question = "Locate every item from the category list in the image and output the coordinates in JSON format. The category set includes person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush."
        
        # Get sample image IDs
        all_img_ids = self.coco_gt.getImgIds()
        if self.args.sample_num > 0:
            self.img_ids = all_img_ids[:self.args.sample_num]
        else:
            self.img_ids = [random.choice(all_img_ids)]  # Default to 1 random sample
            
        print(f"Selected {len(self.img_ids)} images for evaluation")

    def load_model(self):
        """Load Qwen2.5-VL-3B-Instruct model with VLLM"""
        print("Loading Qwen2.5-VL-3B-Instruct model with VLLM...")
        
        model_args = {
            "model_path": self.args.model_path,
            "gpu_memory_utilization": 0.85,
            "max_model_len": 4096
        }
        
        llm = LLM(
            model=model_args["model_path"],
            gpu_memory_utilization=model_args["gpu_memory_utilization"],
            max_model_len=model_args["max_model_len"],
            tensor_parallel_size=1,
        )
        
        processor = AutoProcessor.from_pretrained(self.args.model_path)
        
        print(f"Model loaded successfully")
        return llm, processor

    def parse_detection_output(self, output_text):
        """Parse model output to extract bounding boxes with robust handling of truncated JSON"""
        try:
            # Clean up output
            cleaned_output = output_text.strip()
            if '<answer>' in cleaned_output:
                cleaned_output = cleaned_output.split('<answer>')[1].split('</answer>')[0]
            if '```json' in cleaned_output:
                cleaned_output = cleaned_output.split('```json')[1].split('```')[0]
            elif '```' in cleaned_output:
                cleaned_output = cleaned_output.split('```')[1].split('```')[0]
            
            cleaned_output = cleaned_output.strip()
            
            # Try to parse as complete JSON first
            try:
                detections = json.loads(cleaned_output)
            except json.JSONDecodeError:
                
                # Handle truncated JSON by attempting to complete it
                if cleaned_output.endswith('"label": "'):
                    # Most common truncation point - close the incomplete entry and array
                    cleaned_output = cleaned_output[:-11] + ']'  # Remove incomplete entry and close array
                elif cleaned_output.endswith('"label":'):
                    cleaned_output = cleaned_output[:-8] + ']'
                elif cleaned_output.endswith('"'):
                    # Try to close any incomplete string/object/array
                    cleaned_output += '}]'
                elif cleaned_output.endswith(','):
                    cleaned_output = cleaned_output[:-1] + ']'  # Remove trailing comma and close
                elif not cleaned_output.endswith(']'):
                    cleaned_output += ']'  # Just add closing bracket
                
                # Try parsing again
                try:
                    detections = json.loads(cleaned_output)
                except json.JSONDecodeError:
                    detections = self._extract_detections_regex(output_text)
            
            parsed_detections = []
            for det in detections:
                if isinstance(det, dict) and 'label' in det and ('bbox_2d' in det or 'bbox' in det):
                    # Handle different bbox formats
                    if 'bbox_2d' in det:
                        bbox = det['bbox_2d']
                    else:
                        bbox = det['bbox']
                    
                    # Ensure bbox has 4 coordinates
                    if isinstance(bbox, list) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        parsed_detections.append({
                            'label': det['label'],
                            'bbox': [x1, y1, x2, y2],
                            'score': det.get('score', 1.0)
                        })
            
            return parsed_detections
            
        except Exception as e:
            # Final fallback to regex
            return self._extract_detections_regex(output_text)
    
    def _extract_detections_regex(self, text):
        """Extract detections using regex as fallback"""
        import re
        
        detections = []
        
        # More flexible patterns to handle different JSON structures and truncation
        patterns = [
            # Standard format: bbox_2d before label
            r'"bbox_2d":\s*\[([0-9,\.\s]+)\][^}]*"label":\s*"([^"]+)"',
            # Reverse format: label before bbox_2d
            r'"label":\s*"([^"]+)"[^}]*"bbox_2d":\s*\[([0-9,\.\s]+)\]',
            # Alternative bbox format
            r'"bbox":\s*\[([0-9,\.\s]+)\][^}]*"label":\s*"([^"]+)"',
            r'"label":\s*"([^"]+)"[^}]*"bbox":\s*\[([0-9,\.\s]+)\]'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Handle different order of captures based on pattern
                    if pattern.startswith('"bbox'):
                        bbox_str, label = match
                    else:
                        label, bbox_str = match
                    
                    # Parse bbox coordinates
                    bbox_coords = [float(x.strip()) for x in bbox_str.split(',') if x.strip()]
                    
                    if len(bbox_coords) == 4:
                        detections.append({
                            'label': label,
                            'bbox': bbox_coords,
                            'score': 1.0
                        })
                except Exception as e:
                    continue
            
            if detections:
                break
        
        return detections

    def _process_images_in_batches(self, llm, processor, images, img_ids, img_dims, batch_size):
        """Process images in batches using VLLM"""
        all_results = []
        
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=4096,
            skip_special_tokens=True,
        )
        
        # Prepare all messages
        messages = []
        for img_path in images:
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": self.question}
                ]
            }])
        
        total_batches = (len(messages) + batch_size - 1) // batch_size
        print(f"Processing {len(messages)} images in {total_batches} batches")
        
        for i in tqdm(range(0, len(messages), batch_size), desc="Processing batches"):
            batch = messages[i:i+batch_size]
            batch_img_ids = img_ids[i:i+batch_size]
            batch_img_dims = img_dims[i:i+batch_size]
            
            try:
                # Process text templates
                text = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
                
                # Process visual input
                image_inputs, _ = process_vision_info(batch)
                inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} for prompt, image in zip(text, image_inputs)]
                
                # Generate predictions
                outputs = llm.generate(
                    inputs,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                outputs_decoded = [o.outputs[0].text for o in outputs]
                
                # Parse outputs and convert to COCO format
                for output_text, img_id, (img_width, img_height) in zip(outputs_decoded, batch_img_ids, batch_img_dims):
                    try:
                        pred_boxes = self.parse_detection_output(output_text)
                        print(f"Image {img_id}: {len(pred_boxes)} detections")
                        
                        # Convert to COCO format
                        coco_results = self.convert_to_coco_format(pred_boxes, img_id, img_width, img_height)
                        all_results.extend(coco_results)
                        
                    except Exception as e:
                        print(f"Error processing image {img_id}: {e}")
                        continue
                
                print(f"Batch {i//batch_size + 1}/{total_batches}: {len(batch)} images processed")
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        return all_results

    def get_ground_truth_boxes(self, image_id):
        """Get ground truth bounding boxes for an image"""
        ann_ids = self.coco_gt.getAnnIds(imgIds=image_id)
        anns = self.coco_gt.loadAnns(ann_ids)
        
        gt_boxes = []
        for ann in anns:
            if ann['iscrowd'] == 0:  # Only non-crowd annotations
                bbox = ann['bbox']  # [x, y, w, h]
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                
                cat_info = self.coco_gt.loadCats([ann['category_id']])[0]
                gt_boxes.append({
                    'label': cat_info['name'],
                    'bbox': [x1, y1, x2, y2],
                    'category_id': ann['category_id']
                })
        
        return gt_boxes

    def convert_to_coco_format(self, pred_boxes, image_id, img_width, img_height):
        """Convert predictions to COCO evaluation format"""
        coco_results = []
        
        for pred in pred_boxes:
            if pred['label'] in self.cat_name_to_id:
                x1, y1, x2, y2 = pred['bbox']
                
                # Calculate confidence as bbox area / image area
                bbox_area = (x2 - x1) * (y2 - y1)
                img_area = img_width * img_height
                confidence = float(bbox_area / img_area) if img_area > 0 else 0.0
                
                coco_results.append({
                    "image_id": image_id,
                    "category_id": self.cat_name_to_id[pred['label']],
                    "bbox": [x1, y1, x2-x1, y2-y1],  # COCO format: [x, y, w, h]
                    "score": confidence
                })
        
        return coco_results

    def run_coco_evaluation(self, all_results):
        """Run COCO evaluation"""
        if not all_results:
            print("No valid results to evaluate!")
            return None
        
        # Create output directory
        output_dir = "logs/qwen25_vl_3b_instruct/detection"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(output_dir, f"qwen25_results_{len(self.img_ids)}samples.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Saved {len(all_results)} detection results to: {results_file}")
        
        try:
            # Load results and run evaluation
            coco_dt = self.coco_gt.loadRes(results_file)
            
            # Initialize evaluator
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval.params.imgIds = self.img_ids
            
            print("Running COCO evaluation...")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract results
            result = {
                "model": "Qwen2.5-VL-3B-Instruct",
                "mAP": coco_eval.stats[0],
                "mAP_50": coco_eval.stats[1],
                "mAP_75": coco_eval.stats[2],
                "mAP_small": coco_eval.stats[3],
                "mAP_medium": coco_eval.stats[4],
                "mAP_large": coco_eval.stats[5],
                "AR@1": coco_eval.stats[6],
                "AR@10": coco_eval.stats[7],
                "AR@100": coco_eval.stats[8],
                "num_images": len(self.img_ids),
                "num_detections": len(all_results),
            }
            
            # Save evaluation results
            eval_results_file = os.path.join(output_dir, f"evaluation_results_{len(self.img_ids)}samples.json")
            with open(eval_results_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            return result
            
        except Exception as e:
            print(f"Error in COCO evaluation: {e}")
            return None

    def evaluate(self):
        """Main evaluation function with VLLM batch processing"""
        # Load model
        llm, processor = self.load_model()
        
        # Prepare batch processing
        batch_size = self.args.batch_size  # Use batch size from arguments
        all_results = []
        
        print(f"\n{'='*60}")
        print(f"STARTING EVALUATION ON {len(self.img_ids)} IMAGES (Batch size: {batch_size})")
        print(f"{'='*60}")
        
        # Get all image paths and info first
        valid_images = []
        valid_img_ids = []
        valid_img_dims = []
        
        for img_id in self.img_ids:
            img_info = self.coco_gt.loadImgs([img_id])[0]
            image_path = os.path.join(self.args.image_dir, img_info['file_name'])
            
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                    valid_images.append(image_path)
                    valid_img_ids.append(img_id)
                    valid_img_dims.append((img_width, img_height))
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
            else:
                print(f"Image not found: {image_path}")
        
        print(f"Valid images to process: {len(valid_images)}")
        
        # Process in batches using VLLM
        all_results = self._process_images_in_batches(
            llm, processor, valid_images, valid_img_ids, valid_img_dims, batch_size
        )
            
        # Run COCO evaluation
        if all_results:
            print(f"\n{'='*60}")
            print(f"RUNNING COCO EVALUATION")
            print(f"{'='*60}")
            
            eval_result = self.run_coco_evaluation(all_results)
            
            if eval_result:
                print(f"\n{'='*60}")
                print(f"FINAL RESULTS - {eval_result['model']}")
                print(f"{'='*60}")
                print(f"mAP (IoU 0.5:0.95): {eval_result['mAP']:.4f}")
                print(f"mAP@50 (IoU 0.5): {eval_result['mAP_50']:.4f}")
                print(f"mAP@75 (IoU 0.75): {eval_result['mAP_75']:.4f}")
                print(f"mAP (small): {eval_result['mAP_small']:.4f}")
                print(f"mAP (medium): {eval_result['mAP_medium']:.4f}")
                print(f"mAP (large): {eval_result['mAP_large']:.4f}")
                print(f"AR@1: {eval_result['AR@1']:.4f}")
                print(f"AR@10: {eval_result['AR@10']:.4f}")
                print(f"AR@100: {eval_result['AR@100']:.4f}")
                print(f"\nImages processed: {eval_result['num_images']}")
                print(f"Total detections: {eval_result['num_detections']}")
        else:
            print("No valid results for evaluation!")

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B-Instruct COCO Detection Evaluation")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model path")
    parser.add_argument("--anno_dir", type=str, default="/home/mmpl/workspace/data/annotations/instances_val2017.json", help="COCO annotation file path")
    parser.add_argument("--image_dir", type=str, default="/home/mmpl/workspace/data/val2017", help="COCO images directory")
    parser.add_argument("--sample_num", type=int, default=5000, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = Qwen25VLEvaluator(args)
    evaluator.evaluate()

if __name__ == "__main__":
    main()