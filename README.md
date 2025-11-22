# ML Object Detection on Jetson Nano

Real-time object detection system optimized for NVIDIA Jetson Nano 4GB with TensorRT acceleration. Achieves **~47ms inference time** (21 FPS) using GPU-accelerated TensorRT engine.

---

## Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [Part 1: Flash JetPack 4.6.1](#part-1-flash-jetpack-461)
3. [Part 2: Initial Jetson Setup](#part-2-initial-jetson-setup)
4. [Part 3: Install Docker & NVIDIA Runtime](#part-3-install-docker--nvidia-runtime)
5. [Part 4: Clone Project & Setup](#part-4-clone-project--setup)
6. [Part 5: Build Docker Container](#part-5-build-docker-container)
7. [Part 6: Build TensorRT Engine](#part-6-build-tensorrt-engine)
8. [Part 7: Run Object Detection](#part-7-run-object-detection)
9. [Troubleshooting](#troubleshooting)
10. [Performance](#performance)

---

## Hardware Requirements

- **Jetson Nano 4GB Developer Kit**
- **Power Supply:** 5V 4A barrel jack (recommended) or USB-C with 5V 3A minimum
- **Storage:** MicroSD card 32GB+ (64GB recommended, Class 10 or UHS-1)
- **Webcam:** USB webcam (UVC compatible)
- **For Setup:** Monitor (HDMI), USB keyboard, USB mouse, Ethernet cable
- **Host Computer:** For flashing SD card (Windows/Mac/Linux)

---

## Part 1: Flash JetPack 4.6.1

### Step 1.1: Download JetPack Image

1. Go to [NVIDIA JetPack Download](https://developer.nvidia.com/embedded/jetpack-sdk-461)
2. Download: **Jetson Nano Developer Kit SD Card Image** (JetPack 4.6.1)
3. File size: ~6GB (e.g., `jetson-nano-jp461-sd-card-image.zip`)

### Step 1.2: Flash SD Card

**Using Balena Etcher (Recommended):**

1. Download [Balena Etcher](https://www.balena.io/etcher/)
2. Insert microSD card into your computer
3. Open Etcher:
   - Click "Flash from file" → Select downloaded `.zip` file
   - Click "Select target" → Choose your SD card
   - Click "Flash!"
4. Wait 10-15 minutes for flashing + verification

**⚠️ Warning:** This will erase everything on the SD card.

### Step 1.3: First Boot

1. Insert flashed SD card into Jetson Nano
2. Connect monitor (HDMI), keyboard, mouse, Ethernet
3. Connect power supply
4. Green LED should light up, system boots in ~30 seconds

### Step 1.4: Complete Setup Wizard

Follow on-screen prompts:
- Accept license
- Select language/timezone
- Create user account (e.g., username: `object-detection`, password: `123456`)
- Wait for initial setup to complete (~5-10 minutes)

---

## Part 2: Initial Jetson Setup

### Step 2.1: Verify JetPack Version

```bash
# Check JetPack version
cat /etc/nv_tegra_release
# Should show: R32 (release), REVISION: 7.1
```

### Step 2.2: Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 2.3: Set Jetson to Max Performance Mode

```bash
# Set to 10W mode (all cores active)
sudo nvpmodel -m 0

# Maximize clock speeds
sudo jetson_clocks

# Verify power mode
sudo nvpmodel -q
# Should show: NV Power Mode: MAXN
```

### Step 2.4: Verify CUDA Installation

```bash
# Check CUDA version (add to PATH first)
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
# Should show: Cuda compilation tools, release 10.2

# Make PATH permanent (optional)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Check CUDA version
nvcc --version
# Should show: Cuda compilation tools, release 10.2

# Test tegrastats (the Jetson equivalent of nvidia-smi)
tegrastats
# Press Ctrl+C to exit
```

---

## Part 3: Install Docker & NVIDIA Runtime

### Step 3.1: Install Docker

```bash
# Install curl
sudo apt-get install -y curl

# Now continue with Docker install
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add yourself to docker group
sudo usermod -aG docker $USER

# Check version
docker --version
# Should show: Docker version 20.x.x or higher
```

**⚠️ Important:** Log out and log back in for group changes to take effect.

```bash
# Log out
logout
# Log back in, then verify
docker run hello-world
```

### Step 3.2: Install NVIDIA Container Runtime

```bash
# Install NVIDIA container runtime
sudo apt-get install -y nvidia-container-runtime

# Restart Docker
sudo systemctl restart docker

# Verify NVIDIA runtime
sudo docker info | grep -i runtime
# Should show: runtimes: nvidia runc
```

### Step 3.3: Test GPU Access in Docker

```bash
# Pull NVIDIA L4T base image (takes 5-10 minutes)
sudo docker pull nvcr.io/nvidia/l4t-base:r32.7.1
```

---

## Part 4: Clone Project & Setup

### Step 4.1: Install Git (if not installed)

```bash
sudo apt-get install -y git
```

### Step 4.2: Clone Repository

```bash
cd ~
git clone https://github.com/erwincarlogonzales/mldetection_jetson.git
cd mldetection_jetson
```

### Step 4.3: Verify Project Structure

```bash
ls -la
# Should see:
# - app.py
# - pyproject.toml
# - models/ (folder)
# - README.md

ls -la models/
# Should see:
# - best_f16.pt (your trained model)
```

**Note:** If `best_f16.pt` is large and stored with Git LFS, you may need:
```bash
sudo apt-get install git-lfs
git lfs pull
```

---

## Part 5: Build Docker Container

### Step 5.1: Pull Ultralytics Jetson Image

```bash
# Pull pre-built image with YOLO, PyTorch, TensorRT (3-5 GB, takes 10-20 minutes)
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack4
```

### Step 5.2: Create Container with GPU Access

```bash
# Enable X11 for GUI display
xhost +local:root

# Create container
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

**What this does:**
- `--runtime nvidia --gpus all`: Enables GPU access
- `-e DISPLAY` + `-v /tmp/.X11-unix`: Enables GUI windows
- `-v ~/mldetection_jetson`: Mounts your code into container
- `--device /dev/video0`: Gives access to webcam
- `sleep infinity`: Keeps container running

### Step 5.3: Verify Container

```bash
# Check container is running
sudo docker ps
# Should show mldetection container with status "Up"

# Get shell inside container
sudo docker exec -it mldetection /bin/bash

# You're now inside the container (prompt changes to root@...)
```

### Step 5.4: Verify GPU Access in Container

```bash
# Inside container
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# Should show:
# CUDA available: True
# Device count: 1
```

**✅ If you see `True`, GPU is working! Continue to next step.**

**❌ If you see `False`, see [Troubleshooting - No GPU Access](#no-gpu-access-in-container)**

---

## Part 6: Build TensorRT Engine

### Step 6.1: Navigate to Project Directory

```bash
# Inside container
cd /ultralytics/mldetection_jetson

# Verify files
ls -la
ls -la models/
# Should see best_f16.pt
```

### Step 6.2: Install Dependencies

```bash
# Install required packages
pip3 install supervision opencv-python

# Fix numpy version for TensorRT compatibility
pip3 install numpy==1.23.5
```

### Step 6.3: Build TensorRT Engine

```bash
# Build optimized engine from PyTorch model (takes 2-5 minutes)
python3 -c "from ultralytics import YOLO; model = YOLO('models/best_f16.pt'); model.export(format='engine', device=0, half=True)"
```

**What this does:**
- Loads `best_f16.pt` (PyTorch model)
- Converts to TensorRT engine format
- Optimizes for Jetson Nano GPU
- Uses FP16 precision (half=True) for speed
- Outputs: `models/best_f16.engine`

**Expected output:**
```
Ultralytics YOLOv8.x.x 🚀 Python-3.8.10 torch-x.x.x CUDA:0 (NVIDIA Tegra X1, 3964MiB)

Exporting model...
[TensorRT logs...]
Export complete (2.3s)
Results saved to /ultralytics/mldetection_jetson/models
Predict: yolo predict task=detect model=models/best_f16.engine ...
```

### Step 6.4: Verify Engine Creation

```bash
ls -lh models/
# Should see:
# best_f16.pt (~11-12 MB)
# best_f16.engine (~10-11 MB)
```

---

## Part 7: Run Object Detection

### Step 7.1: Run Application

```bash
# Inside container, in /ultralytics/mldetection_jetson
python3 app.py
```

**Expected output:**
```
Loading model from: models/best_f16.engine
Model loaded successfully!
Opening camera 0...
Camera opened successfully!
Starting detection... Press 'q' to quit
[TensorRT initialization logs...]
Inference: 47.3ms
Inference: 48.1ms
Inference: 46.8ms
...
```

**You should see:**
- Webcam window opens
- Real-time object detection with bounding boxes
- Object count and inference time displayed in top-left corner
- Console showing inference times (~47-60ms)

### Step 7.2: Stop Application

- Press `q` in the detection window to quit
- Or press `Ctrl+C` in terminal

### Step 7.3: Exit Container

```bash
# Exit container shell
exit

# You're back on Jetson host
```

---

## Running After Initial Setup

Once everything is set up, you only need these steps to run again:

### Quick Start Commands

```bash
# On Jetson host
xhost +local:root
sudo docker start mldetection
sudo docker exec -it mldetection /bin/bash

# Inside container
cd /ultralytics/mldetection_jetson
python3 app.py

# When done, press 'q' to quit, then:
exit
```

### If Jetson Restarts

```bash
xhost +local:root
sudo docker start mldetection
sudo docker exec -it mldetection /bin/bash
cd /ultralytics/mldetection_jetson
python3 app.py
```

---

## Troubleshooting

### No GPU Access in Container

**Symptom:** `CUDA available: False` inside container

**Solution:**
```bash
# Exit container
exit

# Verify NVIDIA runtime on host
sudo docker info | grep -i runtime
# Should show "nvidia" in runtimes list

# If missing, reinstall NVIDIA runtime
sudo apt-get install -y nvidia-container-runtime
sudo systemctl restart docker

# Recreate container with GPU flags
sudo docker stop mldetection
sudo docker rm mldetection
# Then run the docker run command from Step 5.2 again
```

### NumPy Compatibility Error

**Symptom:** 
```
AttributeError: module 'numpy' has no attribute 'bool'
```

**Solution:**
```bash
# Inside container
pip3 uninstall numpy -y
pip3 install numpy==1.19.4

# Restart app
python3 app.py
```

### Camera Not Found

**Symptom:** `Error: Could not open camera 0`

**Solution:**
```bash
# On Jetson host, check camera devices
ls -l /dev/video*
# Should show /dev/video0

# Test camera
v4l2-ctl --list-devices

# If camera is /dev/video1, update app.py:
# Change camera_id = 0 to camera_id = 1
```

### Model Not Found Error

**Symptom:** `Error: Model not found at models/best_f16.engine`

**Solution:**
```bash
# Inside container
cd /ultralytics/mldetection_jetson
ls -la models/

# If no .engine file, rebuild it:
python3 -c "from ultralytics import YOLO; model = YOLO('models/best_f16.pt'); model.export(format='engine', device=0, half=True)"
```

### Out of Memory Error

**Symptom:** CUDA out of memory errors

**Solution:**
```bash
# Close other applications
# Restart Jetson to clear memory

# Reduce batch size or resolution if modifying code
```

### Slow Performance / High Inference Time

**Symptom:** Inference time >200ms (should be ~47-60ms)

**Check:**
```bash
# Inside container, verify GPU is being used
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Must show True

# On host, verify max performance mode
sudo nvpmodel -q
# Should show MAXN

# Ensure jetson_clocks is enabled
sudo jetson_clocks
```

### Display Window Doesn't Open

**Symptom:** No GUI window, just console output

**Solution:**
```bash
# On Jetson host (before starting container)
xhost +local:root
echo $DISPLAY
# Should show :0 or :1

# If blank, set it:
export DISPLAY=:0

# Recreate container with correct DISPLAY variable
```

---

## Performance

### Benchmark Results (Jetson Nano 4GB)

| Metric | Value |
|--------|-------|
| **Inference Time** | 47-60ms per frame |
| **Theoretical FPS** | ~21 FPS |
| **Model Size** | 10 MB (TensorRT engine) |
| **GPU Utilization** | 70-90% |
| **RAM Usage** | ~2.8 GB |
| **Power Mode** | MAXN (10W) |

### Comparison: CPU vs GPU

| Backend | Inference Time | FPS |
|---------|---------------|-----|
| CPU Only | ~600ms | 1.6 FPS |
| GPU (TensorRT) | ~50ms | 20 FPS |
| **Speedup** | **12x faster** | |

### Why TensorRT is Fast

1. **GPU Acceleration:** Uses Jetson's Maxwell GPU (128 CUDA cores)
2. **FP16 Precision:** Half-precision reduces memory and compute
3. **Layer Fusion:** Combines operations for efficiency
4. **Hardware-Specific:** Engine built specifically for Jetson Nano architecture

---

## Project Structure

```
mldetection_jetson/
├── app.py                  # Main application
├── pyproject.toml          # Dependencies
├── models/
│   ├── best_f16.pt        # PyTorch model (11 MB)
│   └── best_f16.engine    # TensorRT engine (10 MB, generated)
├── README.md              # This file
└── uv.lock                # Dependency lock file
```

---

## Technical Details

- **Framework:** Ultralytics YOLOv8
- **Backend:** TensorRT 8.2.1
- **CUDA:** 10.2
- **Python:** 3.8
- **JetPack:** 4.6.1
- **Container:** Docker with NVIDIA runtime
- **Precision:** FP16 (half precision)

---

## Notes

- **TensorRT engines are GPU-specific:** The `.engine` file built on Jetson Nano will NOT work on other GPUs (T4, V100, desktop GPUs, etc.). Always rebuild the engine on target hardware.
- **First run is slower:** TensorRT initialization takes ~5-10 seconds on first inference, then speeds up.
- **Container persistence:** Your Docker container and files persist across reboots. Just `docker start mldetection` to resume.
- **Model training:** This guide assumes you have a pre-trained `best_f16.pt` model. For training, see [Ultralytics documentation](https://docs.ultralytics.com/).

---

## Additional Resources

- [NVIDIA Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- [JetPack Documentation](https://docs.nvidia.com/jetson/jetpack/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

---

## Citation

If you use this project in your research, please cite:

```
erwin carlo gonzales
```
