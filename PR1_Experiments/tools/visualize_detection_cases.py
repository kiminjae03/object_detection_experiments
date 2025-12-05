import json
import cv2
import numpy as np
import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from collections import defaultdict

def load_detection_results(json_file):
    """Load detection results from JSON file"""
    with open(json_file, 'r') as f:
        results = json.load(f)
    return results

def group_detections_by_image(detections):
    """Group detections by image ID"""
    by_image = defaultdict(list)
    for det in detections:
        by_image[det['image_id']].append(det)
    return by_image

def calculate_detection_quality(detections, gt_boxes):
    """Calculate simple detection quality score based on detection count vs ground truth"""
    pred_count = len(detections)
    gt_count = len(gt_boxes)
    
    if gt_count == 0:
        return 0.0 if pred_count > 0 else 1.0
    
    # Simple score: closer to 1.0 when prediction count matches ground truth
    detection_ratio = pred_count / gt_count
    if detection_ratio > 2.0:  # Too many detections
        quality = 0.5 / detection_ratio
    elif detection_ratio < 0.3:  # Too few detections
        quality = detection_ratio * 0.5
    else:  # Reasonable detection count
        quality = 1.0 - abs(1.0 - detection_ratio)
    
    return quality

def get_ground_truth_boxes(coco_gt, image_id):
    """Get ground truth bounding boxes for an image"""
    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
    anns = coco_gt.loadAnns(ann_ids)
    
    gt_boxes = []
    for ann in anns:
        if ann['iscrowd'] == 0:  # Only non-crowd annotations
            bbox = ann['bbox']  # [x, y, w, h]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            cat_info = coco_gt.loadCats([ann['category_id']])[0]
            gt_boxes.append({
                'label': cat_info['name'],
                'bbox': [x1, y1, x2, y2],
                'category_id': ann['category_id']
            })
    
    return gt_boxes

def convert_detection_format(detections):
    """Convert COCO detection format to visualization format"""
    pred_boxes = []
    for det in detections:
        x, y, w, h = det['bbox']  # COCO format: [x, y, w, h]
        x2, y2 = x + w, y + h
        pred_boxes.append({
            'label': get_category_name(det['category_id']),
            'bbox': [x, y, x2, y2],  # Convert to [x1, y1, x2, y2]
            'score': det['score']
        })
    return pred_boxes

def get_category_name(category_id):
    """Get category name from ID (COCO categories)"""
    coco_categories = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
        27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
        39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
        43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
        48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
        53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
        58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
        63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
        70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
        76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
        85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
        89: 'hair drier', 90: 'toothbrush'
    }
    return coco_categories.get(category_id, f'category_{category_id}')

def visualize_detections(image_path, pred_boxes, gt_boxes, output_path, title):
    """Visualize predictions and ground truth on image"""
    # Load image
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None
        
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: Predictions
    ax1.imshow(img)
    ax1.set_title(f'Predictions ({len(pred_boxes)} detected)', fontsize=14)
    
    colors_pred = plt.cm.Set1(np.linspace(0, 1, max(len(pred_boxes), 1)))
    for i, det in enumerate(pred_boxes):
        x1, y1, x2, y2 = det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=colors_pred[i % len(colors_pred)], 
                               facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, f"{det['label']} ({det['score']:.2f})", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors_pred[i % len(colors_pred)], alpha=0.7),
                fontsize=8, color='white')
    
    ax1.axis('off')
    
    # Right: Ground Truth
    ax2.imshow(img)
    ax2.set_title(f'Ground Truth ({len(gt_boxes)} objects)', fontsize=14)
    
    colors_gt = plt.cm.Set2(np.linspace(0, 1, max(len(gt_boxes), 1)))
    for i, gt in enumerate(gt_boxes):
        x1, y1, x2, y2 = gt['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=colors_gt[i % len(colors_gt)], 
                               facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y1-5, f"{gt['label']}", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors_gt[i % len(colors_gt)], alpha=0.7),
                fontsize=8, color='white')
    
    ax2.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'predictions': len(pred_boxes),
        'ground_truth': len(gt_boxes),
        'image_path': image_path
    }

def main():
    # Paths
    results_file = '/home/mmpl/workspace/PR1_Experiments/logs/PR1-Qwen2.5-VL-3B-Detection/detection_map/temp_results_1764933671.2941504.json'
    coco_annotation = '/home/mmpl/workspace/Qwen25_VL_Experiments/data/coco_val2017.json'
    image_dir = '/home/mmpl/workspace/Qwen25_VL_Experiments/data/coco/val2017'
    output_dir = '/home/mmpl/workspace/PR1_Experiments/visualization_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading detection results...")
    detections = load_detection_results(results_file)
    print(f"Loaded {len(detections)} detections")
    
    print("Loading COCO ground truth...")
    coco_gt = COCO(coco_annotation)
    
    # Group detections by image
    detections_by_image = group_detections_by_image(detections)
    print(f"Found detections for {len(detections_by_image)} images")
    
    # Calculate quality scores for each image
    image_quality_scores = []
    
    for image_id, image_detections in detections_by_image.items():
        # Get ground truth
        gt_boxes = get_ground_truth_boxes(coco_gt, image_id)
        
        # Calculate quality score
        quality_score = calculate_detection_quality(image_detections, gt_boxes)
        
        # Get image info
        img_info = coco_gt.loadImgs([image_id])[0]
        image_path = os.path.join(image_dir, img_info['file_name'])
        
        image_quality_scores.append({
            'image_id': image_id,
            'image_path': image_path,
            'quality_score': quality_score,
            'detections': image_detections,
            'gt_boxes': gt_boxes,
            'det_count': len(image_detections),
            'gt_count': len(gt_boxes)
        })
    
    # Sort by quality score
    image_quality_scores.sort(key=lambda x: x['quality_score'])
    
    print(f"Analyzed {len(image_quality_scores)} images")
    print(f"Quality score range: {image_quality_scores[0]['quality_score']:.3f} - {image_quality_scores[-1]['quality_score']:.3f}")
    print(f"Images with GT > 0: {len([case for case in image_quality_scores if case['gt_count'] > 0])}")
    print(f"Images with 5+ detections: {len([case for case in image_quality_scores if case['det_count'] >= 5])}")
    
    # Select cases to visualize
    # 3 worst cases (exclude GT count = 0)
    worst_cases_filtered = [case for case in image_quality_scores if case['gt_count'] > 0]
    worst_cases = worst_cases_filtered[:3]
    
    # 3 best cases (only cases with 5+ detections)
    best_cases_filtered = [case for case in image_quality_scores if case['det_count'] >= 5]
    best_cases = best_cases_filtered[-3:] if len(best_cases_filtered) >= 3 else best_cases_filtered
    
    # 3 medium cases
    mid_idx = len(image_quality_scores) // 2
    medium_cases = image_quality_scores[mid_idx-1:mid_idx+2]
    
    cases_to_visualize = [
        ('worst', worst_cases),
        ('medium', medium_cases), 
        ('best', best_cases)
    ]
    
    visualization_results = []
    
    for case_type, cases in cases_to_visualize:
        print(f"\nProcessing {case_type} cases...")
        
        for i, case in enumerate(cases):
            image_id = case['image_id']
            image_path = case['image_path']
            quality_score = case['quality_score']
            detections = case['detections']
            gt_boxes = case['gt_boxes']
            
            print(f"  {case_type.upper()} #{i+1}: Image ID {image_id}, Quality: {quality_score:.3f}, "
                  f"Pred: {len(detections)}, GT: {len(gt_boxes)}")
            
            # Convert detections to visualization format
            pred_boxes = convert_detection_format(detections)
            
            # Create output filename
            output_filename = f"{case_type}_{i+1}_id{image_id}_quality{quality_score:.3f}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Create title
            title = f"{case_type.upper()} Case #{i+1} - ID: {image_id} (Quality: {quality_score:.3f})\n" \
                   f"Predictions: {len(pred_boxes)} | Ground Truth: {len(gt_boxes)}"
            
            # Visualize
            try:
                result = visualize_detections(image_path, pred_boxes, gt_boxes, output_path, title)
                if result:
                    visualization_results.append({
                        'case_type': case_type,
                        'case_number': i+1,
                        'image_id': image_id,
                        'quality_score': quality_score,
                        'output_file': output_filename,
                        'metrics': result
                    })
                    print(f"    Saved: {output_filename}")
                else:
                    print(f"    Failed to process image {image_id}")
            except Exception as e:
                print(f"    Error processing {image_id}: {e}")
    
    # Save summary
    summary = {
        'total_images_analyzed': len(image_quality_scores),
        'filtering_stats': {
            'images_with_gt_objects': len([case for case in image_quality_scores if case['gt_count'] > 0]),
            'images_with_5plus_detections': len([case for case in image_quality_scores if case['det_count'] >= 5]),
            'worst_cases_available': len(worst_cases),
            'best_cases_available': len(best_cases)
        },
        'quality_score_stats': {
            'min': min(s['quality_score'] for s in image_quality_scores),
            'max': max(s['quality_score'] for s in image_quality_scores),
            'avg': np.mean([s['quality_score'] for s in image_quality_scores])
        },
        'visualizations_created': len(visualization_results),
        'visualization_details': visualization_results
    }
    
    summary_path = os.path.join(output_dir, 'visualization_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "="*60)
    print("VISUALIZATION COMPLETED!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Images visualized: {len(visualization_results)}")
    print(f"Summary saved: {summary_path}")
    
    print(f"\nFiltering Statistics:")
    print(f"  Worst cases (GT > 0): {summary['filtering_stats']['worst_cases_available']}/3")
    print(f"  Best cases (5+ detections): {summary['filtering_stats']['best_cases_available']}/3")
    print(f"  Images with GT objects: {summary['filtering_stats']['images_with_gt_objects']}")
    print(f"  Images with 5+ detections: {summary['filtering_stats']['images_with_5plus_detections']}")
    
    print(f"\nQuality Score Statistics:")
    print(f"  Minimum: {summary['quality_score_stats']['min']:.3f}")
    print(f"  Maximum: {summary['quality_score_stats']['max']:.3f}")
    print(f"  Average: {summary['quality_score_stats']['avg']:.3f}")
    
    print(f"\nGenerated files:")
    for result in visualization_results:
        print(f"  {result['output_file']} - {result['case_type'].upper()} case (Quality: {result['quality_score']:.3f})")

if __name__ == "__main__":
    main()