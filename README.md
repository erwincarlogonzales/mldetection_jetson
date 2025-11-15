# ML Detection on Jetson Nano

Real-time object detection demo for dissertation defense. Tests on PC/Mac, deploys to NVIDIA Jetson Nano.

## Quick Start

**For testing on your PC/Mac:**
```bash
git clone <your-repo>
cd mldetection_jetson
uv sync
uv run app.py
```

**For Jetson Nano deployment (if asked during defense):**
See [Jetson Deployment](#jetson-deployment) section below.

---

## Part 1: Development & Testing (PC/Mac)

### Prerequisites
- Python 3.8+
- Webcam
- uv package manager

### Installation

1. **Install uv** (if you don't have it):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and setup**:
```bash
git clone <your-repo>
cd mldetection_jetson
```

3. **Install dependencies**:
```bash
uv sync
```

This installs:
- ultralytics (YOLO inference)
- supervision (visualization)
- opencv-python (camera + display)

### Running on PC/Mac

```bash
uv run app.py
```

**What you'll see:**
- Webcam feed with bounding boxes around detected objects
- Object count in top-left corner
- Press `q` to quit

### Troubleshooting (PC/Mac)

**Camera not opening:**
```bash
# Mac: Grant terminal camera permissions in System Preferences > Security & Privacy
# Linux: Add yourself to video group
sudo usermod -a -G video $USER
```

**Model not found:**
```bash
# Check the model is there
ls -la models/model.onnx
# Should show: 3265 KB file
```

---

## Part 2: Jetson Deployment

**Use this section if committee asks: "Does this actually run on embedded hardware?"**

### Prerequisites (Jetson Nano)
- Jetson Nano 4GB with JetPack 4.6 installed
- USB webcam
- Power supply (5V 4A)
- SSH access to Jetson

### Quick Deploy to Jetson

**1. Clone repo on Jetson:**
```bash
# SSH into Jetson
ssh jetson@<jetson-ip>

# Clone your repo (login with GitHub credentials if private)
cd ~
git clone https://github.com/yourusername/mldetection_jetson.git
cd mldetection_jetson
```

**2. Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

**3. Setup and run:**
```bash
uv sync
uv run app.py
```

**Note:** If repo is private, use your GitHub username and personal access token when prompted.

### Performance Optimization (Jetson)

**Set to max performance mode:**
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

**Monitor resources:**
```bash
sudo tegrastats
```

**Expected performance on Jetson Nano 4GB:**
- Model size: 3.3 MB (YOLO ONNX)
- FPS: ~10-15 FPS at 640x480
- RAM usage: ~2.5-3 GB
- GPU usage: 70-90%

### Jetson Troubleshooting

**Out of memory:**
```bash
# Enable 4GB swap
sudo systemctl disable nvzramconfig
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Camera issues:**
```bash
# Check camera device
ls -l /dev/video*
# Test camera
v4l2-ctl --list-devices
```

**Slow performance:**
- Close other applications
- Lower webcam resolution (modify app.py line 24)
- Ensure Jetson is in max performance mode

---

## Project Structure

```
mldetection_jetson/
├── models/
│   └── model.onnx          # YOLO ONNX model (3.3 MB)
├── app.py                   # Main application
├── pyproject.toml           # Dependencies
├── uv.lock                  # Locked versions
└── README.md               # This file
```

## Configuration

### Change Camera Source

Edit `app.py` line 22:
```python
camera_id = 0  # Change to 1 for second camera
```

### Adjust Detection Confidence

Edit `app.py` line 54:
```python
object_count = sum(1 for conf in detections.confidence if conf > 0.5)
# Change 0.5 to your threshold (0.3 = more detections, 0.7 = fewer false positives)
```

---

## Dissertation Defense Notes

**If asked about edge deployment:**
- Model runs on Jetson Nano 4GB (embedded hardware)
- ~10-15 FPS real-time inference
- 3.3 MB model size (efficient for edge)
- No cloud/internet required

**If asked about portability:**
- Same code runs on PC/Mac for development
- Deploys to Jetson with zero code changes
- Uses relative paths (cross-platform compatible)

**If asked for live demo:**
1. Ensure Jetson is powered on and connected
2. SSH in: `ssh jetson@<jetson-ip>`
3. Run: `cd mldetection_jetson && uv run app.py`
4. Show webcam feed with real-time detections

---

## Dependencies

- **ultralytics** (>=8.3.228): YOLO model inference
- **supervision** (>=0.26.1): Detection visualization
- **opencv-python**: Camera capture and display
- **numpy**: Installed automatically

## Technical Details

- **Model format**: ONNX (cross-platform)
- **Input size**: 640x640 (configurable)
- **Detection threshold**: 0.5 confidence (configurable)
- **Framework**: PyTorch via Ultralytics
