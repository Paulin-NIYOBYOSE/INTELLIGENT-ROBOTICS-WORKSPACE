# Face Locking System

Real-time face tracking that locks onto a specific person, detects their movements (left/right), blinks, and smiles, then logs everything to timestamped files.

## Clone This Folder Only

If this is part of a larger repo, clone just this folder:

```bash
# Using sparse checkout (recommended)
git clone --depth 1 --filter=blob:none --sparse https://github.com/Paulin-NIYOBYOSE/INTELLIGENT-ROBOTICS-WORKSPACE.git
cd INTELLIGENT-ROBOTICS-WORKSPACE
git sparse-checkout set Face_Locking
cd Face_Locking
```

Or use degit (faster):

```bash
npx degit Paulin-NIYOBYOSE/INTELLIGENT-ROBOTICS-WORKSPACE/Face_Locking Face_Locking
cd Face_Locking
```

## Quick Setup

### 1. Create Virtual Environment

```bash
python3 -m venv face_env
source face_env/bin/activate  # On Windows: face_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install opencv-python numpy onnxruntime mediapipe==0.10.9 pillow tqdm matplotlib
```

### 3. Download ArcFace Model

```bash
curl -L -o models/arcface.onnx \
  https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx
```

## Enroll Your Face

### Take 3-5 Photos

- Good lighting, clear frontal face
- Save as: `photo1.jpg`, `photo2.jpg`, `photo3.jpg`

### Create `enroll_me.py`:

```python
from src.embed import ArcFaceEmbedder
from src.enroll import enroll_identity

embedder = ArcFaceEmbedder("models/arcface.onnx")

image_paths = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]

count, folder = enroll_identity(
    name="YourName",  # Change this
    image_paths=image_paths,
    embedder=embedder,
    output_dir="data/identities",
)

print(f"✓ Enrolled {count} samples")
```

### Run Enrollment:

```bash
python enroll_me.py
```

## Run Face Locking

```bash
python -m src.run_pipeline
```

1. Select your enrolled identity (type number + Enter)
2. Wait for **RED box** around your face (lock established)
3. Test actions:
   - **Move head left/right** - Slowly move your head
   - **Blink** - Close and open eyes deliberately
   - **Smile** - Smile broadly
4. Watch **green banner** at top for detected actions
5. Press **'q'** to quit

## Check Results

```bash
ls action_history/
cat action_history/yourname_history_*.txt
```

You'll see timestamped logs of all detected actions.

## What You Get

- **Face Locking**: Locks onto selected person (red box)
- **Stable Tracking**: Maintains lock even during brief occlusions (2s timeout)
- **Action Detection**:
  - `moved_left` - Head moved left
  - `moved_right` - Head moved right
  - `blink` - Eye blink detected
  - `smile` - Smile/laugh detected
- **Action History**: Timestamped logs in `action_history/yourname_history_YYYYMMDDHHMMSS.txt`

## File Structure

```
Face_Locking/
├── src/
│   ├── run_pipeline.py       # Main program
│   ├── face_locker.py        # Lock management
│   ├── action_detector.py    # Action detection
│   ├── action_logger.py      # History logging
│   └── ... (other modules)
├── models/                   # ArcFace model (download)
├── data/identities/          # Enrolled faces
├── action_history/           # Action logs (auto-created)
└── README.md
```

## Troubleshooting

**Lock not establishing?**

- Ensure good lighting
- Face camera directly
- Lower `lock_threshold` to `0.35` in `src/run_pipeline.py` line 119

**Actions not detected?**

- Make movements more pronounced
- Ensure face is well-lit
- Adjust thresholds in `src/action_detector.py` lines 24-26

**Camera not opening?**

- Check camera permissions
- Close other apps using camera

## Requirements

- Python 3.8+
- Webcam
- CPU only (no GPU needed)

## Tech Stack

- **Face Recognition**: ArcFace ResNet100 (ONNX)
- **Face Detection**: MediaPipe
- **Action Detection**: MediaPipe FaceMesh (468 landmarks)
- **Framework**: OpenCV, NumPy

---

**Built for Intelligent Robotics Course**
