#!/usr/bin/env python3
"""
YOLOv8-World COCO Detection Evaluation Script
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from ultralytics import YOLOWorld
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class YOLOv8WorldDetector:
    def __init__(self, model_name='yolov8s-world', device='auto'):
        """
        Initialize YOLOv8-World model
        
        Args:
            model_name (str): Model name (yolov8n-world, yolov8s-world, yolov8m-world, yolov8l-world, yolov8x-world)
            device (str): Device for inference ('auto', 'cpu', 'cuda')
        """
        self.model = YOLOWorld(model_name)
        self.device = device
        print(f"Loaded YOLOv8-World model: {model_name}")
        
        # COCO class names for open-vocabulary detection
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # COCO category ID mapping (COCO uses non-sequential IDs)
        self.coco_category_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]
        
        # Set vocabulary for open-vocabulary detection
        self.model.set_classes(self.coco_classes)
        print(f"Set vocabulary with {len(self.coco_classes)} COCO classes")

    def detect(self, image_path, conf_threshold=0.25):
        """
        Detect objects in image
        
        Args:
            image_path (str): Path to image
            conf_threshold (float): Confidence threshold
            
        Returns:
            list: Detections in COCO format
        """
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # Additional confidence filtering (stricter than threshold)
                    if conf < conf_threshold * 1.2:  # 20% higher than minimum
                        continue
                    
                    # Validate bounding box
                    width = x2 - x1
                    height = y2 - y1
                    if width <= 0 or height <= 0:
                        continue
                    
                    # Convert to COCO bbox format [x, y, width, height]
                    bbox = [float(x1), float(y1), float(width), float(height)]
                    
                    detection = {
                        'bbox': bbox,
                        'score': float(conf),
                        'category_id': self.coco_category_ids[cls_id],  # Use COCO category ID
                    }
                    detections.append(detection)
        
        return detections

    def evaluate_coco(self, coco_json_path, image_dir, output_file, sample_num=None):
        """
        Evaluate YOLOv8-World on COCO dataset
        
        Args:
            coco_json_path (str): Path to COCO annotation file
            image_dir (str): Directory containing images
            output_file (str): Output file for results
            sample_num (int): Number of samples to evaluate (None for all)
        """
        # Load COCO dataset
        coco_gt = COCO(coco_json_path)
        all_image_ids = list(coco_gt.imgs.keys())
        
        if sample_num:
            all_image_ids = all_image_ids[:sample_num]
        
        # Pre-validate images and collect valid ones only (like Qwen25_VL approach)
        valid_image_ids = []
        valid_image_paths = []
        
        for image_id in all_image_ids:
            image_info = coco_gt.imgs[image_id]
            image_path = os.path.join(image_dir, image_info['file_name'])
            
            if os.path.exists(image_path):
                valid_image_ids.append(image_id)
                valid_image_paths.append(image_path)
            else:
                print(f"Warning: Image not found: {image_path}")
        
        print(f"Evaluating on {len(valid_image_ids)} valid samples (out of {len(all_image_ids)} requested)")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        results = []
        start_time = time.time()
        
        for i, (image_id, image_path) in enumerate(tqdm(zip(valid_image_ids, valid_image_paths), desc="Processing images", total=len(valid_image_ids))):
            # Detect objects with confidence threshold
            detections = self.detect(image_path, conf_threshold=0.25)
            
            # Add image_id to each detection
            for detection in detections:
                detection['image_id'] = image_id
                results.append(detection)
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                print(f"Processed {i + 1}/{len(valid_image_ids)} images, avg time: {avg_time:.3f}s/img")
        
        # Note: Results will be saved with summary after evaluation
        
        # Evaluate with COCO metrics (only on valid images)
        metrics = None
        if results:
            metrics = self._compute_coco_metrics(coco_gt, results, valid_image_ids)
        
        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.2f}s")
        print(f"Average time per image: {total_time / len(valid_image_ids):.3f}s")
        
        # Create final results with summary
        final_results = {
            'detections': results,
            'summary': {
                'model': 'yolov8s-world',
                'total_images': len(valid_image_ids),
                'total_detections': len(results),
                'total_time_seconds': total_time,
                'avg_time_per_image': total_time / len(valid_image_ids),
                'metrics': metrics
            }
        }
        
        # Save final results with summary
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        return final_results

    def _compute_coco_metrics(self, coco_gt, results, valid_image_ids):
        """Compute COCO evaluation metrics"""
        try:
            # Load results
            coco_dt = coco_gt.loadRes(results)
            
            # Run evaluation (only on valid images, like Qwen25_VL approach)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.params.imgIds = valid_image_ids  # Only evaluate processed images
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Print formatted results
            print("\n" + "="*60)
            print("COCO EVALUATION RESULTS - YOLOv8-World")
            print("="*60)
            print(f"mAP (IoU 0.5:0.95): {coco_eval.stats[0]:.4f}")
            print(f"mAP@50 (IoU 0.5): {coco_eval.stats[1]:.4f}")
            print(f"mAP@75 (IoU 0.75): {coco_eval.stats[2]:.4f}")
            print(f"mAP (small): {coco_eval.stats[3]:.4f}")
            print(f"mAP (medium): {coco_eval.stats[4]:.4f}")
            print(f"mAP (large): {coco_eval.stats[5]:.4f}")
            print(f"AR@1: {coco_eval.stats[6]:.4f}")
            print(f"AR@10: {coco_eval.stats[7]:.4f}")
            print(f"AR@100: {coco_eval.stats[8]:.4f}")
            print(f"AR (small): {coco_eval.stats[9]:.4f}")
            print(f"AR (medium): {coco_eval.stats[10]:.4f}")
            print(f"AR (large): {coco_eval.stats[11]:.4f}")
            
            # Return metrics for saving
            return {
                'mAP_0.5:0.95': float(coco_eval.stats[0]),
                'mAP_0.5': float(coco_eval.stats[1]),
                'mAP_0.75': float(coco_eval.stats[2]),
                'mAP_small': float(coco_eval.stats[3]),
                'mAP_medium': float(coco_eval.stats[4]),
                'mAP_large': float(coco_eval.stats[5]),
                'AR_1': float(coco_eval.stats[6]),
                'AR_10': float(coco_eval.stats[7]),
                'AR_100': float(coco_eval.stats[8]),
                'AR_small': float(coco_eval.stats[9]),
                'AR_medium': float(coco_eval.stats[10]),
                'AR_large': float(coco_eval.stats[11])
            }
            
        except Exception as e:
            print(f"Error computing COCO metrics: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='YOLOv8-World COCO Detection Evaluation')
    parser.add_argument('--model', type=str, default='yolov8s-world',
                        choices=['yolov8n-world', 'yolov8s-world', 'yolov8m-world', 'yolov8l-world', 'yolov8x-world'],
                        help='YOLOv8-World model variant')
    parser.add_argument('--coco_json', type=str, 
                        default='/home/mmpl/workspace/data/annotations/instances_val2017.json',
                        help='Path to COCO annotation file')
    parser.add_argument('--image_dir', type=str,
                        default='/home/mmpl/workspace/data/val2017',
                        help='Directory containing COCO images')
    parser.add_argument('--output', type=str,
                        default='results/yolov8_world_results.json',
                        help='Output file for detection results')
    parser.add_argument('--sample_num', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='Confidence threshold for detection')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for inference')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLOv8WorldDetector(model_name=args.model, device=args.device)
    
    # Run evaluation
    detector.evaluate_coco(
        coco_json_path=args.coco_json,
        image_dir=args.image_dir,
        output_file=args.output,
        sample_num=args.sample_num
    )


if __name__ == "__main__":
    main()