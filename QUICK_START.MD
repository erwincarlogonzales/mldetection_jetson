# Quick Start Guide

For users who already have:
- ✅ Jetson Nano with JetPack 4.6.1
- ✅ Docker installed with NVIDIA runtime
- ✅ Git installed

---

## 1. Clone Repository

```bash
cd ~
git clone https://github.com/erwincarlogonzales/mldetection_jetson.git
cd mldetection_jetson
```

---

## 2. Create Docker Container

```bash
# Enable X11 display
xhost +local:root

# Pull image (first time only, ~3-5 GB)
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack4

# Create container with GPU access
sudo docker run -d \
  --name mldetection \
  --runtime nvidia \
  --gpus all \
  --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/mldetection_jetson:/ultralytics/mldetection_jetson \
  --device /dev/video0:/dev/video0 \
  --privileged \
  ultralytics/ultralytics:latest-jetson-jetpack4 \
  sleep infinity
```

---

## 3. Build TensorRT Engine

```bash
# Get shell in container
sudo docker exec -it mldetection /bin/bash

# Inside container - navigate to project
cd /ultralytics/mldetection_jetson

# Install dependencies
pip3 install supervision opencv-python numpy==1.19.4

# Build TensorRT engine (takes 2-5 minutes)
python3 -c "from ultralytics import YOLO; model = YOLO('models/best_f16.pt'); model.export(format='engine', device=0, half=True)"
```

---

## 4. Run Detection

```bash
# Still inside container
python3 app.py

# Press 'q' to quit
```

---

## Running Again (After Setup)

```bash
# On Jetson host
xhost +local:root
sudo docker start mldetection
sudo docker exec -it mldetection /bin/bash

# Inside container
cd /ultralytics/mldetection_jetson
python3 app.py
```

---

## Expected Performance

- **Inference Time:** 47-60ms per frame
- **FPS:** ~20 FPS
- **GPU Utilization:** 70-90%

---

## Troubleshooting

### NumPy Error
```bash
# Inside container
pip3 install numpy==1.19.4
```

### No GPU Access
```bash
# Inside container - verify CUDA
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Must show: CUDA: True
```

### Camera Not Found
```bash
# On host - check camera device
ls -l /dev/video*
# Update app.py camera_id if needed
```

---

## Full Documentation

See [README.md](README.md) for complete setup from scratch including JetPack flashing.