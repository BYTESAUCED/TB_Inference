# TB Inference

FastAPI service for tuberculosis screening from chest X-ray images and DICOM studies. The system loads a trained transformer model from `model/best_model.pth`, preprocesses image inputs, and returns a normal/abnormal prediction with confidence and optional heatmap data.

## Requirements

- Python 3.12
- pip or uv
- Optional: CUDA GPU or Apple Silicon MPS for faster inference

## Setup

1. Open the project folder.
2. Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pydicom pillow fastapi uvicorn python-multipart timm opencv-python matplotlib
```

4. Confirm the checkpoint exists:

```bash
ls model/best_model.pth
```

5. Validate the model file loads:

```bash
python verify_model_load.py
```

## Run the service

```bash
python main.py
```

The service listens on:

- http://localhost:8001/health
- http://localhost:8001/docs

## API usage

Send a request to `/analyze`:

```json
{
  "patient_ids": ["PID-12345"]
}
```

The server expects patient folders in `organized_scans` under the project root. Each patient directory should be named with the patient ID prefix and contain `.dcm` or `.dicom` files.

## Docker

```bash
docker build -t tb-inference .
docker run -p 8001:8001 tb-inference
```

If the host supports GPU execution, attach GPU access in Docker for better throughput.

## Technical design

### 1. Runtime startup and model selection

The application entrypoint is `main.py`, which starts Uvicorn with:

```python
uvicorn.run("app.server:app", host=host, port=port, reload=False)
```

On server startup, `app.server` creates a singleton `engine` and calls `get_system_strategy()`. This function selects the backend based on hardware:

- Apple Silicon with MPS support -> `mps`
- CUDA available -> `cuda`
- Otherwise -> `cpu`

This is implemented in `app/inference.py` and is the main platform-aware distribution point for execution. The selected strategy drives both device placement and whether `torch.compile` is enabled.

### 2. Model architecture

The inference model is an EVA-X Vision Transformer (`EVA_X`) defined in `app/eva_x.py`. It is a binary classifier tuned for 224x224 RGB inputs and outputs two logits for:

- normal
- abnormal

The final classification head is replaced with a linear layer of shape `[in_features -> 2]` before weights are loaded from the checkpoint.

### 3. Model loading and optimization

`InferenceEngine._load_model()` does the following:

1. Validates that the checkpoint exists.
2. Constructs the EVA-X architecture.
3. Loads `model/best_model.pth` into the model state.
4. Moves the model to the selected device.
5. Calls `torch.compile` when the backend supports it.

A warmup pass is executed after compilation to trigger graph generation and reduce first-request latency. The compiled model is used for fast inference, while an uncompiled copy is retained for Grad-CAM heatmap generation because gradient hooks are easier to manage in eager mode.

### 4. Preprocessing pipeline

Before inference, each file passes through `preprocess_image()` in `app/inference.py`.

For DICOM input:

- read using `pydicom`
- extract raw pixel matrix
- normalize pixel intensity to a valid dynamic range
- apply CLAHE contrast enhancement with OpenCV
- resize to 224x224
- convert grayscale to RGB
- normalize with ImageNet mean/std values
- convert from HWC to CHW and add batch dimension

For non-DICOM images:

- open with PIL
- resize to 224x224
- convert to RGB
- normalize with the same mean/std values
- reshape to `[1, 3, 224, 224]`

This standardizes the input format for the model while preserving the radiographic contrast features needed for classification.

### 5. Inference flow

The request lifecycle is controlled by `POST /analyze` in `app/server.py`.

For each patient ID in the payload:

1. Search `organized_scans/<patient_id>*` for a matching folder.
2. Collect all `.dcm` and `.dicom` files in that folder.
3. For each scan:
   - preprocess the file
   - run `engine.predict_with_heatmap(...)`
   - compute softmax probabilities from model logits
   - classify as abnormal if `abnormal_prob >= 0.61`
   - attach confidence and optional heatmap payload
4. Aggregate the patient result and summary.

The softmax step is implemented in `softmax(x)` as:

```python
exp(x - max(x)) / sum(exp(x - max(x)))
```

The model returns logits in order `[normal, abnormal]`, so the abnormal class score is interpreted as the disease probability.

### 6. Heatmap generation

The optional heatmap is generated with Grad-CAM.

The logic in `InferenceEngine.predict_with_heatmap()`:

- registers forward and backward hooks on the last transformer block
- runs a forward pass with gradients enabled
- computes the gradient-weighted activation map
- removes the CLS token and reshapes the remaining tokens into a grid
- applies ReLU and normalization
- resizes the map to 224x224

This produces a saliency map that highlights regions most responsible for the model’s prediction, which can be returned as `heatmap_grid` in the JSON response.

### 7. Distribution model

This project is not a multi-node distributed ML pipeline in its current form. The architecture is a single-process inference service with the following distribution characteristics:

- model is loaded once at application startup as a singleton
- a single FastAPI process handles all requests
- each request fans out across patient IDs and DICOM files
- hardware distribution is handled by device selection: CPU, MPS, or CUDA
- the service can be replicated horizontally by running multiple containers or instances behind a load balancer

In practical terms, the system is distributed at the deployment layer, not at the computation layer. Each instance owns one model copy in memory, and multiple replicas can serve traffic independently. That makes the application easy to scale but not fully sharded or parallelized within one process.

### 8. Operational considerations

- The service expects the model file to exist in `model/best_model.pth`.
- It assumes patient scan folders are mounted or available in `organized_scans`.
- Inference latency depends on the selected device and whether compilation is active.
- Health checks are served by `/health` and are used to verify whether a model instance is loaded successfully.

## Summary

This project implements a compact inference service for tuberculosis detection from X-ray studies. The model is a transformer-based classifier, input preprocessing standardizes medical images, and the API exposes a batch analysis endpoint that evaluates each patient’s scans and returns abnormality probabilities with optional heatmap overlays. Distribution is currently achieved by process replication and hardware-aware execution rather than by a clustered inference framework.
