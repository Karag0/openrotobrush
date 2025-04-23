# openrotobrush
rotoscope test

**OpenRotoBrush Project**

A video object segmentation tool with GUI for creating alpha masks using YOLO segmentation. Built with Python, Tkinter, and OpenCV.


## Features

- **Video Loading**: Supports common video formats (MP4, AVI, etc.)
- **Frame Navigation**: Play/pause, slider scrubbing, manual frame selection
- **Object Selection**: Draw bounding boxes to initialize object tracking
- **AI Segmentation**: Uses YOLOv11-seg model for mask generation
- **Mask Visualization**: Toggle overlay of colored segmentation masks
- **Batch Processing**: Auto-generate masks for all frames
- **Alpha Channel Export**: Save PNG sequence with transparency
- **History Management**: Reset masks and clear selections

## Installation

1. **Prerequisites**:
   - Python 3.8+
   - Tkinter (usually included with Python)

2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy Pillow ultralytics
