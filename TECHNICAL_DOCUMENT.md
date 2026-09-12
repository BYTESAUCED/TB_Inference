# TB Inference Technical Document

## 1. Overview

This project implements a tuberculosis screening inference service built with FastAPI and PyTorch. It loads a trained EVA-X Vision Transformer model from `model/best_model.pth`, preprocesses medical images or DICOM slices, and predicts whether a chest X-ray is normal or abnormal. The API also returns a confidence score and an optional Grad-CAM-style heatmap to help identify suspicious image regions.

## 2. Runtime architecture

The application is started by `main.py`:

```python
uvicorn.run("app.server:app", host=host, port=port, reload=False)
```

This creates a FastAPI application defined in `app/server.py`. At startup, a singleton `engine` is initialized and the model is loaded once into memory. This pattern reduces repeated model initialization overhead and keeps the service responsive to repeated requests.

## 3. Model selection and hardware distribution

The model runtime is selected in `app/inference.py` via `get_system_strategy()`:

- Apple Silicon with MPS available -> MPS backend
- CUDA available -> CUDA backend
- otherwise -> CPU backend

This is the system’s hardware-aware distribution layer. The selected backend is used for model placement and optimization. The service can therefore run on:

- local CPU machines
- Apple Silicon devices using MPS
- GPU-enabled servers with CUDA

### Why this matters

This decision is important because inference throughput and latency depend directly on the device. CUDA and MPS allow the model to run on accelerators, while CPU remains the fallback mode for portability.

## 4. Model definition

The model class is defined in `app/eva_x.py`. It uses an EVA-X Vision Transformer backbone and replaces the output head with a binary classification head:

- input: 224x224 RGB image
- output: 2 logits
- classes: normal, abnormal

The checkpoint is loaded into the architecture using `torch.load(...)` with the device mapped appropriately.

## 5. Compilation strategy

After loading, the engine optionally calls `torch.compile` for acceleration. The code keeps an uncompiled model copy for heatmap generation because gradient hooks and backpropagation are easier to manage in eager mode.

The compile decision is based on the selected strategy:

- CUDA -> `torch.compile(..., backend="torch_tensorrt", mode="max-autotune")`
- MPS -> `torch.compile(self.model, backend="mps")`
- CPU -> no compilation

A warmup inference pass is run once after compilation to stabilize performance and trigger the compile path.

## 6. Preprocessing pipeline

The preprocessing path is implemented in `preprocess_image()` and `preprocess_dicom()` in `app/inference.py`.

### For DICOM files

1. Read the file with `pydicom`
2. Extract the pixel array
3. Normalize intensity values to a usable range
4. Apply CLAHE contrast enhancement using OpenCV
5. Resize to 224x224
6. Convert grayscale to RGB
7. Normalize using ImageNet mean/std values
8. Reorder the array to CHW format
9. Add a batch dimension

### For regular image files

1. Open with PIL
2. Resize to 224x224
3. Convert to RGB
4. Normalize with mean/std values
5. Convert to CHW and add batch dimension

This prepares the image in the format expected by the ViT encoder so the model can classify it consistently.

## 7. Inference flow

The inference request is handled by `POST /analyze` in `app/server.py`.

For each patient ID in the payload:

1. Search the patient folder under `organized_scans/<patient_id>*`
2. Collect all `.dcm` and `.dicom` files inside
3. Preprocess each scan
4. Call the model to obtain logits
5. Compute class probabilities with softmax
6. Decide whether the image is abnormal using a threshold of `0.61`
7. Return file-level and patient-level summaries

The abnormal score is taken from the second class in the logits array:

```python
probs = softmax(logits)
abnormal_prob = float(probs[0, 1])
```

If `abnormal_prob >= 0.61`, the image is labeled as abnormal; otherwise it is labeled as normal.

## 8. Heatmap and explainability

The model includes a Grad-CAM-style heatmap path in `InferenceEngine.predict_with_heatmap()`.

This process:

- registers hooks on the final transformer block
- captures activations and gradients
- computes weighted activation scores
- removes the CLS token
- reshapes the activation map to a grid
- applies ReLU and normalization
- resizes the result to 224x224

The heatmap is then returned as a grid in the response payload for each analyzed file. This helps explain which regions contributed most strongly to the abnormal prediction.

## 9. API contract

The service exposes a batch inference API:

### Endpoint

- `POST /analyze`

### Request

```json
{
  "patient_ids": ["PID-12345"]
}
```

### Response structure

Each patient result contains:

- `patient_id`
- `files[]`
- `summary`

Each file result contains:

- `filename`
- `prediction`
- `confidence`
- `abnormal_probability`
- `heatmap_grid` (optional)
- `error` (optional)

## 10. Distribution model

This system is currently a single-service deployment architecture, not a distributed deep learning system in the strict cluster sense.

### Current distribution pattern

- one FastAPI process owns one model instance
- requests are handled serially within the process
- patient IDs are processed in a loop, and each scan is processed individually
- GPU or MPS acceleration is chosen based on host hardware

### Deployment-level scaling

The service can be distributed by running multiple instances behind a load balancer or in container orchestration. In that model:

- each instance loads its own model copy
- traffic is split across replicas
- scaling is horizontal, not model-sharded

So the system is distributed operationally, but not by partitioning model inference across multiple nodes inside a single request.

## 11. Operational notes

- The system expects a trained model at `model/best_model.pth`
- Patient data is read from a folder structure under `organized_scans`
- Runtime health can be checked at `/health`
- API documentation is available at `/docs`

## 12. Summary

The inference pipeline is a straightforward medical imaging classification service: preprocess image or DICOM data, run a transformer-based model, compute abnormal probability, and return results with an explainability overlay. The service is optimized by hardware-aware backend selection and optional compilation, and it can be scaled horizontally by running multiple service instances behind a load balancer.
