🔐 Face Detection & Privacy Enforcement on Media Streams

This project is a practical implementation of the IEEE research paper:  
**“A Privacy-Enforcing Framework for Data Streams on the Edge”**  
It focuses on preserving privacy by detecting and transforming faces in images and video streams using lightweight, edge-compatible methods.

---

📌 Features

- `facedetectandblur.py` module for face anonymization
- Supports both **image and video inputs**
- Uses ONNX-based **UltraFace** for fast face detection
- Applies **OpenCV-based blurring/pixelation** to anonymize detected faces
- Tested on videos with different **frame rates (30 FPS & 16 FPS)** for latency and performance evaluation

---

🧪 Experiment Overview

Two videos with varying frame rates were processed to:

- Measure **frame-wise latency**
- Analyze performance impact due to FPS variations
- Assess **quality of anonymization**

This prototype replicates the **trigger → transformation** mechanism proposed in the paper:
- **Trigger**: Detect faces (possible privacy violation)
- **Transformation**: Blur faces (privacy enforcement)

---

 📂 Repository Structure

```bash
facedetectandblur/
├── facedetectandblur.py
├── models/
│   └── ultraface.onnx              # ONNX model file
├── examples/
│   ├── images/
│   │   └── sample1.jpg
│   └── videos/
│       ├── video_30fps.mp4
│       └── video_16fps.mp4
├── outputs/
│   ├── blurred_video_30fps.mp4
│   └── blurred_video_16fps.mp4
├── screenshots/
│   ├── original_frame.jpg
│   └── blurred_frame.jpg
├── requirements.txt
├── README.md
├── .gitignore
└── download_models.py
