# ArcFace CPU Face Recognition

CPU-only face recognition pipeline using ArcFace ONNX, MediaPipe detection, and 5-point alignment.

## Setup

1. Activate the virtual environment (already provided as `face_env`).
   ```bash
   source face_env/bin/activate
   ```
2. Install dependencies if needed:
   ```bash
   pip install opencv-python numpy onnxruntime mediapipe pillow tqdm matplotlib
   ```
3. Download the ArcFace model to `models/arcface.onnx`:
   ```bash
   curl -L -o models/arcface.onnx \
     https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx
   ```

## Directory Structure

```
face_recognition_arcface/
├── src/                # Core pipeline modules
├── data/identities/    # Enrolled identities + embeddings
├── models/             # ArcFace ONNX model
├── tests/              # Tests
└── README.md
```

## Enroll New Identities

Prepare a folder of images for each person and run enrollment in Python:

```python
from src.embed import ArcFaceEmbedder
from src.enroll import enroll_identity

embedder = ArcFaceEmbedder("models/arcface.onnx")
image_paths = [
    "samples/alice_1.jpg",
    "samples/alice_2.jpg",
]

count, folder = enroll_identity(
    name="alice",
    image_paths=image_paths,
    embedder=embedder,
    output_dir="data/identities",
)

print(f"Enrolled {count} samples to {folder}")
```

Each enrolled identity saves:

- `data/identities/<name>/crops/*.jpg` (112x112 aligned faces)
- `data/identities/<name>/embeddings.npy` (L2-normalized embeddings)

## Run Live Recognition

```bash
python -m src.run_pipeline
```

Press **q** to quit.

## Recommended Threshold

Start with a cosine similarity threshold of **0.45**. Increase it for stricter matching.

## Notes

- All embeddings are L2-normalized.
- The pipeline runs fully on CPU and is suitable for laptops.
- MediaPipe is used for detection and 5-point landmark alignment.
- `models/` and `data/identities/` are gitignored so you can push the repo safely.
