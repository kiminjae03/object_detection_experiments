# Multi-Model Object Detection Evaluation Framework

This repository contains a comprehensive evaluation framework for comparing different object detection models on COCO2017 dataset.

## Models Evaluated

### 1. YOLOv8-World
- **Performance**: 27.38% mAP (1000 samples)
- **Speed**: 19.61 FPS
- **Features**: Open-vocabulary detection, real-time inference
- **Location**: `YOLOv8_World_Experiments/`

### 2. Qwen2.5-VL-3B-Instruct
- **Performance**: 27.31% mAP (100 samples)
- **Speed**: ~5.4 FPS
- **Features**: Vision-language model, detailed scene understanding
- **Location**: `Qwen25_VL_Experiments/`

### 3. PR1-Qwen2.5-VL-3B (Enhanced)
- **Performance**: Similar to base Qwen2.5-VL with improvements
- **Features**: Enhanced detection pipeline with improved evaluation
- **Location**: `PR1_Experiments/`

## Key Results

| Model | mAP (0.5:0.95) | mAP@50 | Speed (FPS) | Advantage |
|-------|----------------|---------|-------------|-----------|
| YOLOv8-World | 27.38% | 42.33% | 19.61 | Real-time speed |
| Qwen2.5-VL | 27.31% | 35.83% | ~5.4 | Scene understanding |
| PR1-Qwen2.5-VL | ~27.3% | ~36% | ~5.4 | Enhanced pipeline |

## Repository Structure

```
workspace/
├── YOLOv8_World_Experiments/     # YOLOv8-World evaluation
│   ├── scripts/                  # Detection scripts
│   ├── results/                  # Evaluation results
│   ├── logs/                     # Performance logs
│   └── tools/                    # Comparison tools
├── Qwen25_VL_Experiments/        # Qwen2.5-VL evaluation
│   ├── data/                     # COCO dataset links
│   ├── logs/                     # Evaluation logs
│   └── results/                  # Detection results
└── PR1_Experiments/              # Enhanced Qwen2.5-VL
    ├── tools/                    # Visualization tools
    ├── logs/                     # Results and metrics
    └── visualization_results/    # Generated visualizations
```

## Quick Start

### 1. YOLOv8-World Evaluation
```bash
cd YOLOv8_World_Experiments
conda activate yolov8_world
python scripts/yolov8_world_detection.py --sample_num 1000
```

### 2. Qwen2.5-VL Evaluation
```bash
cd Qwen25_VL_Experiments  
python qwen25_vl_detection_eval.py --sample_num 100
```

### 3. Visualization and Comparison
```bash
cd PR1_Experiments
python tools/visualize_detection_cases.py
```

## Key Findings

1. **Performance Parity**: YOLOv8-World achieves comparable accuracy to state-of-the-art vision-language models
2. **Speed Advantage**: YOLOv8-World is 3.6x faster, making it suitable for real-time applications
3. **Use Case Optimization**: Each model has distinct advantages for different scenarios

## Technical Improvements

### YOLOv8-World Enhancements
- Fixed evaluation pipeline issues (0.84% → 27.38% mAP)
- Implemented proper confidence thresholds
- Added comprehensive result formatting
- Improved COCO evaluation scope

### Evaluation Framework
- Cross-model comparison tools
- Comprehensive visualization system
- Statistical analysis and filtering
- Performance benchmarking

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- ultralytics (YOLOv8-World)
- transformers (Qwen2.5-VL)
- pycocotools
- OpenCV, matplotlib

## Citation

If you use this evaluation framework, please cite:

```bibtex
@misc{multimodel_detection_eval_2024,
  title={Multi-Model Object Detection Evaluation Framework},
  author={kiminjae03},
  year={2024},
  url={https://github.com/kiminjae03/multi-model-detection-eval}
}
```

## License

This project is licensed under the MIT License - see individual model licenses for specific components.

## Acknowledgments

- YOLOv8-World: Ultralytics team
- Qwen2.5-VL: Alibaba DAMO Academy
- COCO Dataset: Microsoft COCO team