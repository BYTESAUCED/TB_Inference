"""
TB Inference Engine - Tuberculosis detection using EVA-X Vision Transformer
Supports CUDA with torch.compile optimization, MPS (Mac), and CPU fallback.
"""
import argparse
import json
import os
import sys
import numpy as np
import platform
from PIL import Image

# Import PyTorch
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

# Import DICOM/CV2 for preprocessing
try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import cv2
except ImportError:
    cv2 = None

# Import model definition
try:
    from .eva_x import EVA_X
except ImportError:
    try:
        from eva_x import EVA_X
    except ImportError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from eva_x import EVA_X
        except ImportError:
            print("Warning: could not import EVA_X")
            EVA_X = None


def get_system_strategy(force_cpu=False):
    """Determine optimal inference backend based on available hardware."""
    strategy = {
        "backend": "cpu",
        "device": "cpu",
        "compile": False,
        "description": "PyTorch CPU"
    }
    
    if force_cpu or not torch:
        return strategy

    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        strategy = {
            "backend": "mps",
            "device": "mps",
            "compile": True,
            "description": "PyTorch MPS (Apple Silicon)"
        }
    elif torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        strategy = {
            "backend": "cuda",
            "device": "cuda",
            "compile": True,
            "description": f"PyTorch CUDA ({gpu_name})"
        }
    
    return strategy


class InferenceEngine:
    """Handles model loading, compilation, and inference."""
    
    def __init__(self, model_path: str, strategy: dict):
        self.strategy = strategy
        self.model = None
        self.model_uncompiled = None
        self.device = torch.device(strategy["device"]) if torch else "cpu"
        self.model_path = model_path
        self.is_compiled = False
        
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load and optionally compile the model."""
        if not torch or not EVA_X:
            print("ERROR: PyTorch or EVA_X model not available")
            return False
        
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model file not found: {self.model_path}")
            return False
        
        try:
            print(f"Loading model on {self.strategy['description']}...")
            
            # Create model architecture
            self.model = EVA_X(
                img_size=224,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                qkv_fused=False,
                mlp_ratio=4 * 2 / 3,
                swiglu_mlp=True,
                scale_mlp=True,
                use_rot_pos_emb=True,
                ref_feat_shape=(14, 14),
            )
            
            # Replace head for binary classification
            self.model.head = nn.Linear(self.model.head.in_features, 2)
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            
            # Keep uncompiled copy for heatmap generation (hooks need eager mode)
            self.model_uncompiled = self.model
            
            # Compile for faster inference
            if self.strategy["compile"] and self.strategy["backend"] == "cuda":
                self._compile_cuda()
            elif self.strategy["compile"] and self.strategy["backend"] == "mps":
                self._compile_mps()
            
            print(f"Model ready: compiled={self.is_compiled}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            return False
    
    def _compile_cuda(self):
        """Compile model with torch.compile for CUDA."""
        print("Compiling model with inductor backend...")
        try:
            self.model = torch.compile(
                self.model,
                backend="torch_tensorrt",
                mode="max-autotune"
            )
            self._warmup()
            self.is_compiled = True
            self.strategy["description"] += " + torch.compile"
            print("Compilation successful")
        except Exception as e:
            print(f"Compilation failed, using eager mode: {e}")
            self.model = self.model_uncompiled
    
    def _compile_mps(self):
        """Compile model for Apple Silicon."""
        print("Compiling model for MPS...")
        try:
            self.model = torch.compile(self.model, backend="mps")
            self._warmup()
            self.is_compiled = True
        except Exception as e:
            print(f"MPS compilation failed: {e}")
            self.model = self.model_uncompiled
    
    def _warmup(self):
        """Run warmup inference to trigger compilation."""
        print("Running warmup inference...")
        dummy = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.inference_mode():
            _ = self.model(dummy)
        print("Warmup complete")
    
    def predict(self, img_numpy: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed image.
        Args:
            img_numpy: [1, 3, 224, 224] float32 normalized array
        Returns:
            logits: [1, 2] array with [normal, abnormal] scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        with torch.inference_mode():
            tensor = torch.from_numpy(img_numpy).to(self.device)
            logits = self.model(tensor)
            return logits.cpu().numpy()
    
    def predict_with_heatmap(self, img_numpy: np.ndarray) -> tuple:
        """
        Run inference and generate Grad-CAM heatmap.
        Returns: (logits, heatmap) where heatmap is 224x224 numpy array or None
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Use uncompiled model for gradient hooks
        model = self.model_uncompiled
        
        gradients = []
        activations = []
        
        def save_gradient(module, grad_in, grad_out):
            gradients.append(grad_out[0])
        
        def save_activation(module, inp, out):
            activations.append(out)
        
        # Hook the last transformer block
        target = model.blocks[-1].norm1
        h1 = target.register_forward_hook(save_activation)
        h2 = target.register_full_backward_hook(save_gradient)
        
        try:
            tensor = torch.from_numpy(img_numpy).to(self.device)
            tensor.requires_grad_(True)
            model.zero_grad()
            
            logits = model(tensor)
            pred_class = logits.argmax(dim=1).item()
            logits[0, pred_class].backward()
            
            # Compute Grad-CAM
            grads = gradients[0].detach().cpu().numpy()[0]  # [197, 768]
            acts = activations[0].detach().cpu().numpy()[0]  # [197, 768]
            
            weights = grads.mean(axis=0)  # [768]
            cam = np.dot(acts, weights)   # [197]
            
            # Remove CLS token and reshape to grid
            n_tokens = cam.shape[0]
            grid_size = int(np.sqrt(n_tokens - 1))
            
            if n_tokens == grid_size**2 + 1:
                cam = cam[1:].reshape(grid_size, grid_size)
            else:
                return logits.detach().cpu().numpy(), None
            
            # ReLU and normalize
            cam = np.maximum(cam, 0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Resize to image size
            heatmap = np.array(Image.fromarray(cam).resize((224, 224), Image.Resampling.BILINEAR))
            
            return logits.detach().cpu().numpy(), heatmap
            
        except Exception as e:
            print(f"Heatmap generation failed: {e}")
            # Try to return logits at least
            try:
                with torch.inference_mode():
                    tensor = torch.from_numpy(img_numpy).to(self.device)
                    logits = self.model(tensor)
                    return logits.cpu().numpy(), None
            except:
                return None, None
        finally:
            h1.remove()
            h2.remove()


# ============== Preprocessing Functions ==============

def preprocess_dicom(image_path: str, img_size: int = 224) -> tuple:
    """
    Preprocess a DICOM file for inference.
    Returns: (numpy_array, error_string) - one will be None
    """
    if pydicom is None:
        return None, "pydicom not installed"
    if cv2 is None:
        return None, "opencv-python not installed"
    
    try:
        ds = pydicom.dcmread(image_path)
        
        if not hasattr(ds, 'pixel_array'):
            return None, "DICOM has no pixel data"
        
        pixels = ds.pixel_array
        if pixels is None:
            return None, "pixel_array is None"
        
        # Convert to float for processing
        pixels = pixels.astype(np.float32)
        
        # Normalize to 0-255
        pmin, pmax = pixels.min(), pixels.max()
        if pmax > pmin:
            pixels = (pixels - pmin) / (pmax - pmin) * 255.0
        else:
            pixels = np.full_like(pixels, 128.0)
        
        pixels = pixels.astype(np.uint8)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        pixels = clahe.apply(pixels)
        
        # Resize
        pixels = cv2.resize(pixels, (img_size, img_size))
        
        # Convert grayscale to RGB
        img = np.stack([pixels] * 3, axis=-1).astype(np.float32)
        
        # Normalize and standardize (ImageNet stats)
        img = img / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # CHW format with batch dimension
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        
        return img, None
        
    except Exception as e:
        return None, f"DICOM error: {str(e)}"


def preprocess_image(image_path: str, img_size: int = 224) -> tuple:
    """
    Preprocess any image file for inference.
    Returns: (numpy_array, error_string) - one will be None
    """
    if image_path.lower().endswith(('.dicom', '.dcm')):
        return preprocess_dicom(image_path, img_size)
    
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
        img = np.array(img).astype(np.float32)
        
        # Normalize and standardize
        img = img / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # CHW format with batch dimension
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        
        return img, None
        
    except Exception as e:
        return None, str(e)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# ============== CLI Interface ==============

def run_inference(args):
    """Command-line inference."""
    result = {
        "filename": args.image_path,
        "prediction": None,
        "confidence": 0.0,
        "probabilities": {},
        "threshold": args.threshold,
        "error": None,
        "device": "Unknown"
    }
    
    if not os.path.exists(args.image_path):
        result["error"] = f"File not found: {args.image_path}"
        print(json.dumps(result))
        return
    
    # Preprocess
    img, error = preprocess_image(args.image_path)
    if error:
        result["error"] = error
        print(json.dumps(result))
        return
    
    try:
        strategy = get_system_strategy(force_cpu=args.cpu)
        engine = InferenceEngine(args.model_path, strategy)
        
        if not engine.model:
            result["error"] = "Failed to load model"
            print(json.dumps(result))
            return
        
        result["device"] = engine.strategy["description"]
        
        logits = engine.predict(img)
        probs = softmax(logits)
        
        normal_prob = float(probs[0, 0])
        abnormal_prob = float(probs[0, 1])
        
        result["probabilities"] = {"normal": normal_prob, "abnormal": abnormal_prob}
        
        if abnormal_prob >= args.threshold:
            result["prediction"] = "Abnormal"
            result["confidence"] = abnormal_prob
        else:
            result["prediction"] = "Normal"
            result["confidence"] = normal_prob
            
    except Exception as e:
        result["error"] = f"Inference failed: {str(e)}"
    
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TB Inference CLI")
    parser.add_argument('--image_path', required=True, help='Path to input image')
    parser.add_argument('--model_path', default='model/best_model.pth', help='Path to model')
    parser.add_argument('--threshold', default=0.5, type=float, help='Abnormal threshold')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    
    run_inference(parser.parse_args())

