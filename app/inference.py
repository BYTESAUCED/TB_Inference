import argparse
import json
import os
import sys
import time
import numpy as np
import platform
import types
from PIL import Image

# Import Dependencies
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import pydicom
    import cv2
except ImportError:
    pydicom = None
    cv2 = None

# Fallback import for model definition
try:
    # Try relative import first (for package execution)
    from .eva_x import EVA_X
except ImportError:
    try:
        # Try absolute import (for script execution)
        from eva_x import EVA_X
    except ImportError:
        # Try adding current dir to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from eva_x import EVA_X
        except ImportError:
            print("Warning: could not import EVA_X")
            EVA_X = None

# Attempt to import export logic
try:
    from .export_onnx import export_model
except ImportError:
    try:
        from export_onnx import export_model
    except ImportError:
        export_model = None


def get_system_strategy(force_cpu=False):
    """
    Determine the optimal inference strategy based on OS and Hardware.
    Returns: config dict
    """
    system = platform.system()
    strategy = {
        "backend": "pytorch_cpu",
        "provider": "cpu",
        "description": "Default PyTorch CPU"
    }
    
    if force_cpu:
        return strategy

    if system == "Darwin":
        # MacOS
        if torch and torch.backends.mps.is_available():
            strategy = {
                "backend": "pytorch_mps",
                "provider": "mps",
                "description": "PyTorch MPS + torch.compile"
            }
    elif system in ["Linux", "Windows"]:
        # Linux/Windows -> Prefer TensorRT via ONNX
        # Only if we can use ORT-GPU/TRT
        if ort and 'TensorrtExecutionProvider' in ort.get_available_providers():
            strategy = {
                "backend": "onnx_trt",
                "provider": "TensorrtExecutionProvider",
                "description": "ONNX Runtime TensorRT"
            }
        elif torch and torch.cuda.is_available():
             # Fallback to PyTorch CUDA if ORT-TRT not available? 
             # User requested "default to cpu use default pytorch not onnx"
             # So if TRT fails/missing, we might skip to CPU per instructions, 
             # but normally CUDA is better. I will stick to user instruction: "windows and linux just torch.... default to cpu"
             # wait, user said "windows and linux ... compile with backend tensorrt ... and default to cpu use default pytorch"
             # I'll stick to TRT -> CPU fallback as requested.
             pass
    
    return strategy

class InferenceEngine:
    def __init__(self, model_path_inp, strategy):
        self.strategy = strategy
        self.model = None
        self.session = None
        self.device = "cpu"
        self.is_onnx = False
        self.model_path_inp = model_path_inp
        
        # Determine paths
        base, ext = os.path.splitext(model_path_inp)
        self.pth_path = base + ".pth" if ext != ".pth" else model_path_inp
        self.onnx_path = base + ".onnx" if ext != ".onnx" else model_path_inp
        
        # Load
        if strategy['backend'] == 'onnx_trt':
            if self._load_onnx():
                return
            else:
                print("Falling back to PyTorch CPU...")
                strategy['backend'] = 'pytorch_cpu'
                self.strategy = strategy

        if strategy['backend'] == 'pytorch_mps':
            if not self._load_pytorch("mps"):
                 print("Falling back to PyTorch CPU...")
                 strategy['backend'] = 'pytorch_cpu'
                 self.strategy = strategy
        
        if strategy['backend'] == 'pytorch_cpu':
            self._load_pytorch("cpu")

    def _load_onnx(self):
        if not ort:
            print("ONNX Runtime not installed.")
            return False
            
        # Check if ONNX exists, else export
        if not os.path.exists(self.onnx_path):
            print(f"ONNX model not found at {self.onnx_path}. Attempting export...")
            if export_model and os.path.exists(self.pth_path):
                try:
                    # Mock args
                    args = types.SimpleNamespace(
                        model_path=self.pth_path,
                        output_path=self.onnx_path
                    )
                    export_model(args)
                except Exception as e:
                    print(f"Export failed: {e}")
                    return False
            else:
                print("Cannot export: export_model function missing or source .pth missing.")
                return False
                
        try:
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
            self.is_onnx = True
            print(f"Loaded ONNX model with providers: {self.session.get_providers()}")
            return True
        except Exception as e:
            print(f"Failed to load ONNX session: {e}")
            return False

    def _load_pytorch(self, device_name):
        if not torch or not EVA_X:
            print("PyTorch or EVA_X model definition not found.")
            return False
            
        if not os.path.exists(self.pth_path):
            print(f"Model file not found: {self.pth_path}")
            return False

        try:
            self.device = torch.device(device_name)
            
            # Instantiate Model
            # Note: We must match the config from export_onnx.py
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
            # Head
            num_ftrs = self.model.head.in_features
            self.model.head = nn.Linear(num_ftrs, 2)
            
            # Load Weights
            checkpoint = torch.load(self.pth_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            if device_name == "mps":
                print("Compiling model (MPS)...")
                try:
                    self.model = torch.compile(self.model, backend="mps", mode="max-autotune")
                except Exception as e:
                    print(f"torch.compile failed ignored: {e}")
            
            self.is_onnx = False
            return True
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")
            return False

    def predict(self, img_numpy):
        # img_numpy: [1, 3, 224, 224] float32 normalized
        
        if self.is_onnx:
             input_name = self.session.get_inputs()[0].name
             outputs = self.session.run(None, {input_name: img_numpy})
             return outputs[0] # logits
        elif self.model:
             with torch.no_grad():
                 tensor = torch.from_numpy(img_numpy).to(self.device)
                 logits = self.model(tensor)
                 return logits.cpu().numpy()
        else:
             raise RuntimeError("No model loaded.")

    def predict_with_heatmap(self, img_numpy):
        """
        Runs inference and generates a heatmap using Grad-CAM on the final block.
        Returns: logits, heatmap_grid (224x224 numpy array)
        """
        if self.is_onnx:
            # Grad-CAM not supported easily on ONNX Runtime without complex graph modification
            # or model support. Returning None for heatmap.
            print("Warning: Heatmap generation not supported for ONNX backend yet.")
            logits = self.predict(img_numpy)
            return logits, None
        
        if not self.model:
             raise RuntimeError("No model loaded.")

        # Gradient Capture
        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        # Hook into the last block's norm or appropriate layer for ViT
        # For EVA-ViT, usually `model.blocks[-1].norm1` or similar
        # checking model structure from eva_x.py: blocks are timm blocks
        # We'll try targetting the last block's output
        target_layer = self.model.blocks[-1].norm1 
        
        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        
        try:
            tensor = torch.from_numpy(img_numpy).to(self.device).requires_grad_(True)
            self.model.zero_grad()
            
            # Forward
            logits = self.model(tensor)
            
            # Target identifying 'Abnormal' class (index 1) or Max class
            # Let's target the predicted class for relevance
            pred_idx = logits.argmax(dim=1).item()
            score = logits[0, pred_idx]
            
            # Backward
            score.backward()
            
            # Grad-CAM Calculation
            # Gradients: [1, N_tokens, Dim] -> [1, 197, 768] (Tiny/Base)
            grads = gradients[0].cpu().data.numpy()[0] # [197, 768]
            acts = activations[0].cpu().data.numpy()[0] # [197, 768]
            
            # Global Average Pooling of Gradients -> Weights
            weights = np.mean(grads, axis=0) # [768]
            
            # Weighted Activations
            cam = np.dot(acts, weights) # [197]
            
            # Reshape (removing cls token if present)
            # ViT usually has 1 CLS token + 14x14 grid = 197 tokens
            num_tokens = cam.shape[0]
            grid_size = int(np.sqrt(num_tokens - 1))
            
            if num_tokens == grid_size**2 + 1:
                cam = cam[1:] # Skip CLS
                cam = cam.reshape(grid_size, grid_size)
            elif num_tokens == grid_size**2:
                 cam = cam.reshape(grid_size, grid_size)
            else:
                # Fallback or error?
                print(f"Heatmap shape mismatch: {num_tokens} tokens.")
                return logits.detach().cpu().numpy(), None

            # ReLU
            cam = np.maximum(cam, 0)
            
            # Normalize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Resize to 224x224
            cam_img = Image.fromarray(cam)
            cam_img = cam_img.resize((224, 224), Image.Resampling.BILINEAR)
            heatmap = np.array(cam_img)
            
            return logits.detach().cpu().numpy(), heatmap
            
        except Exception as e:
            print(f"Heatmap generation failed: {e}")
            return logits.detach().cpu().numpy() if 'logits' in locals() else None, None
        finally:
            h1.remove()
            h2.remove()


def preprocess_dicom(image_path, img_size=224):
    if pydicom is None or cv2 is None:
        return None, "pydicom or opencv-python not installed, cannot process DICOM"
        
    try:
        ds = pydicom.dcmread(image_path)
        pixel_array = ds.pixel_array
        
        # Windowing (Min-Max to 0-255)
        pixel_array = pixel_array.astype(float)
        if np.max(pixel_array) != np.min(pixel_array):
            pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255.0
        pixel_array = pixel_array.astype(np.uint8)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        pixel_array = clahe.apply(pixel_array)
        
        # Resize
        pixel_array = cv2.resize(pixel_array, (img_size, img_size))
        
        # Convert to 3 channels (RGB)
        img = np.stack((pixel_array,)*3, axis=-1)
        
        # Normalize (0-1)
        img = img.astype(np.float32) / 255.0
        
        # Transpose to CHW
        img = img.transpose((2, 0, 1))
        
        # Standardization (ImageNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        
        img = (img - mean) / std
        
        # Batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img, None
        
    except Exception as e:
        return None, f"DICOM processing error: {str(e)}"

def preprocess_image(image_path, img_size=224):
    if image_path.lower().endswith(('.dicom', '.dcm')):
        return preprocess_dicom(image_path, img_size)

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return None, str(e)

    # Resize
    image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    
    # To Numpy
    img_data = np.array(image).astype(np.float32)
    
    # Normalize (0-1) and then Standardize (ImageNet stats)
    img_data = img_data / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std =  np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_data = (img_data - mean) / std
    
    # CHW format
    img_data = img_data.transpose((2, 0, 1))
    
    # Add batch dimension
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data, None

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def run_inference(args):
    result = {
        "filename": args.image_path,
        "prediction": None,
        "confidence": 0.0,
        "all_probs": {},
        "threshold": args.threshold,
        "error": None,
        "device": "Unknown"
    }

    if not os.path.exists(args.image_path):
        result["error"] = f"Image file not found: {args.image_path}"
        print(json.dumps(result))
        return

    # Preprocess
    input_data, error = preprocess_image(args.image_path)
    if error:
        result["error"] = error
        print(json.dumps(result))
        return

    try:
        strategy = get_system_strategy(force_cpu=args.cpu)
        engine = InferenceEngine(args.model_path, strategy)
        
        if not engine.model and not engine.session:
             result["error"] = "Failed to load model on any backend."
             print(json.dumps(result))
             return
             
        result["device"] = engine.strategy['description']
        
        logits = engine.predict(input_data)
        probs = softmax(logits)
        
        # Class 0: Normal, Class 1: Abnormal
        abnormal_prob = float(probs[0, 1])
        normal_prob = float(probs[0, 0])
        
        result["all_probs"] = {
            "normal": normal_prob,
            "abnormal": abnormal_prob
        }
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, type=str, help='Path to input image')
    # Default to .pth so the script can choose .pth or derive .onnx
    parser.add_argument('--model_path', default='model/best_model.pth', type=str, help='Path to model (pth or onnx)')
    parser.add_argument('--threshold', default=0.61, type=float, help='Decision threshold for Abnormal class')
    parser.add_argument('--cpu', action='store_true', help='Force CPU execution')
    args = parser.parse_args()
    
    run_inference(args)

