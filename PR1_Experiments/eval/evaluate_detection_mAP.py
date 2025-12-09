from transformers import AutoProcessor
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import time
import argparse
import ray
import torch
import tempfile
import numpy as np

class COCOEvaluatorMAP:
    def __init__(self, args):
        self.args = args

        self.model_args = {
            "model_path": args.model_path,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 8192
        }
        self.coco_gt = COCO(self.args.anno_dir)
        
        # Use the same question format from the working examples
        self.question = "Locate every item from the category list in the image and output the coordinates in JSON format. The category set includes person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush."
        
        # Sample selection - limit to reasonable number for testing
        all_img_ids = self.coco_gt.getImgIds()
        if self.args.sample_num > 0:
            self.img_ids = all_img_ids[:self.args.sample_num]
        else:
            self.img_ids = all_img_ids[:100]  # Default to 100 samples for testing
            
        self.images = [os.path.join(self.args.image_dir, x['file_name']) for x in self.coco_gt.loadImgs(self.img_ids)]
        
        # Filter out missing images
        valid_images = []
        valid_img_ids = []
        for img_path, img_id in zip(self.images, self.img_ids):
            if os.path.exists(img_path):
                valid_images.append(img_path)
                valid_img_ids.append(img_id)
            else:
                print(f"Warning: Image not found: {img_path}")
        
        self.images = valid_images
        self.img_ids = valid_img_ids
        
        self.gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.cat_name_to_id = {cat['name']: cat['id'] for cat in self.coco_gt.dataset['categories']}
        
        print(f"Total valid images to process: {len(self.images)}")

    def evaluate_dataset(self):
        """Evaluate dataset using single GPU with batch processing"""
        # Simplified single GPU approach for better reliability
        try:
            results = self._infer_on_single_gpu_batch(
                self.images, self.img_ids, self.model_args, self.question, self.cat_name_to_id, self.args.batch_size
            )
            return self._run_coco_eval(results)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None

    def _infer_on_single_gpu_batch(self, images, img_ids, model_args, question, cat_name_to_id, batch_size):
        """Run inference on single GPU with batch processing"""
        from vllm import LLM, SamplingParams
        from qwen_vl_utils import process_vision_info
        from PIL import Image
        
        print("Loading model...")
        llm = LLM(
            model=model_args["model_path"],
            gpu_memory_utilization=model_args["gpu_memory_utilization"],
            max_model_len=model_args["max_model_len"],
            tensor_parallel_size=1,
        )
        
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=4096,
            skip_special_tokens=True,
        )

        processor = AutoProcessor.from_pretrained(model_args["model_path"])
        
        # Prepare all messages
        messages = []
        for img_path in images:
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": question}
                ]
            }])

        results = []
        total_batches = (len(messages) + batch_size - 1) // batch_size
        
        print(f"Processing {len(messages)} images in {total_batches} batches of size {batch_size}")
        
        for i in tqdm(range(0, len(messages), batch_size), desc="Processing batches"):
            batch = messages[i:i+batch_size]
            batch_img_ids = img_ids[i:i+batch_size]
            batch_img_paths = images[i:i+batch_size]
            
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
                
                # Get image dimensions for proper confidence calculation
                batch_img_dims = []
                for img_path in batch_img_paths:
                    try:
                        with Image.open(img_path) as img:
                            batch_img_dims.append((img.width, img.height))
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        batch_img_dims.append((640, 480))  # Default size
                
                # Parse outputs and add to results
                batch_results = self._parse_output_improved(outputs_decoded, batch_img_ids, cat_name_to_id, batch_img_dims)
                results.extend(batch_results)
                
                print(f"Batch {i//batch_size + 1}/{total_batches}: {len(batch_results)} detections")
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                continue

        print(f"Total detections collected: {len(results)}")
        return results
    
    def _parse_output_improved(self, outputs, img_ids, cat_name_to_id, img_dims):
        """Enhanced parsing with better confidence calculation"""
        results = []
        
        for o, img_id, (img_width, img_height) in zip(outputs, img_ids, img_dims):
            try:
                # Clean up output like in the working examples
                cleaned_output = o
                if '<answer>' in cleaned_output:
                    cleaned_output = cleaned_output.split('<answer>')[1].split('</answer>')[0]
                cleaned_output = cleaned_output.replace("```json", "").replace("```", "").strip()
                
                # Parse JSON
                detections = json.loads(cleaned_output)
                
                for det in detections:
                    # Check if detection has required fields
                    if "label" not in det or "bbox_2d" not in det:
                        continue
                        
                    label = cat_name_to_id.get(det["label"], None)
                    if label is None:
                        continue
                    
                    x1, y1, x2, y2 = det["bbox_2d"]
                    
                    # Calculate confidence as bbox area / image area
                    bbox_area = (x2 - x1) * (y2 - y1)
                    img_area = img_width * img_height
                    confidence = float(bbox_area / img_area) if img_area > 0 else 0.0
                    
                    results.append({
                        "image_id": img_id,
                        "category_id": label,
                        "bbox": [x1, y1, x2-x1, y2-y1],  # COCO format: [x, y, w, h]
                        "score": confidence
                    })
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for image {img_id}: {e}")
                continue
            except Exception as e:
                print(f"Other parsing error for image {img_id}: {e}")
                continue
                
        return results

    def _run_coco_eval(self, results):
        """Run COCO evaluation with detailed output"""
        if not results:
            print("No valid results to evaluate!")
            return None
            
        # Create output directory
        output_path = os.path.join('logs', os.path.basename(self.args.model_path), 'detection_map')
        os.makedirs(output_path, exist_ok=True)
        
        # Save results to temporary file
        temp_file = os.path.join(output_path, f"temp_results_{time.time()}.json")
        with open(temp_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved {len(results)} detection results to: {temp_file}")
        
        try:
            # Load results and run evaluation
            coco_dt = self.coco_gt.loadRes(temp_file)
            
            # Initialize evaluator
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            
            # Set evaluation parameters
            coco_eval.params.imgIds = self.img_ids  # Evaluate only processed images
            
            print("Running COCO evaluation...")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Extract results
            result = {
                "mAP": coco_eval.stats[0],
                "mAP_50": coco_eval.stats[1], 
                "mAP_75": coco_eval.stats[2],
                "mAP_small": coco_eval.stats[3],
                "mAP_medium": coco_eval.stats[4],
                "mAP_large": coco_eval.stats[5],
                "AR@1": coco_eval.stats[6],
                "AR@10": coco_eval.stats[7],
                "AR@100": coco_eval.stats[8],
                "AR_small": coco_eval.stats[9],
                "AR_medium": coco_eval.stats[10],
                "AR_large": coco_eval.stats[11],
            }
            
            # Add additional statistics
            result.update({
                "num_images": len(self.img_ids),
                "num_detections": len(results),
                "avg_detections_per_image": len(results) / len(self.img_ids) if self.img_ids else 0
            })
            
            # Save final results
            final_results_file = os.path.join(output_path, f'coco_eval_results_{len(self.img_ids)}samples.json')
            with open(final_results_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n" + "="*60)
            print(f"FINAL mAP RESULTS ({len(self.img_ids)} images):")
            print("="*60)
            print(f"mAP (IoU 0.5:0.95): {result['mAP']:.4f}")
            print(f"mAP@50 (IoU 0.5): {result['mAP_50']:.4f}")
            print(f"mAP@75 (IoU 0.75): {result['mAP_75']:.4f}")
            print(f"mAP (small): {result['mAP_small']:.4f}")
            print(f"mAP (medium): {result['mAP_medium']:.4f}")
            print(f"mAP (large): {result['mAP_large']:.4f}")
            print(f"AR@1: {result['AR@1']:.4f}")
            print(f"AR@10: {result['AR@10']:.4f}")
            print(f"AR@100: {result['AR@100']:.4f}")
            print(f"\nImages processed: {result['num_images']}")
            print(f"Total detections: {result['num_detections']}")
            print(f"Avg detections/image: {result['avg_detections_per_image']:.1f}")
            print(f"\nResults saved to: {final_results_file}")
            print(f"Detection file: {temp_file}")
            
            return result
            
        except Exception as e:
            print(f"Error in COCO evaluation: {e}")
            # Provide basic statistics as fallback
            self._basic_statistics(results)
            return None

    def _basic_statistics(self, results):
        """Provide basic statistics when COCO eval fails"""
        print("\n" + "="*50)
        print("BASIC STATISTICS (COCO eval failed):")
        print("="*50)
        
        if not results:
            print("No results to analyze")
            return
            
        # Count by category
        cat_counts = {}
        for r in results:
            cat_id = r['category_id']
            cat_name = None
            for cat in self.coco_gt.dataset['categories']:
                if cat['id'] == cat_id:
                    cat_name = cat['name']
                    break
            if cat_name:
                cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
        
        print(f"Total detections: {len(results)}")
        print(f"Images processed: {len(self.img_ids)}")
        print(f"Avg detections per image: {len(results)/len(self.img_ids):.1f}")
        print("\nDetections by category:")
        for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")

def main(args):
    evaluator = COCOEvaluatorMAP(args)
    result = evaluator.evaluate_dataset()
    
    if result:
        print(f"\nEvaluation completed successfully!")
        print(f"Final mAP: {result['mAP']:.4f}")
    else:
        print("Evaluation failed or produced no results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced COCO Detection mAP Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--anno_dir", type=str, required=True, help="COCO annotation file path")
    parser.add_argument("--image_dir", type=str, required=True, help="COCO images directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--sample_num", type=int, default=50, help="Number of samples to evaluate (default: 50)")
    
    args = parser.parse_args()
    main(args)