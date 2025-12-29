from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import glob
import sys
import numpy as np

# Adjust path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.inference import get_system_strategy, InferenceEngine, preprocess_image, softmax

app = FastAPI(title="TB Inference Server", description="Batch inference for Tuberculosis scans with Heatmaps.")

# Global Model Engine
engine = None
MODEL_PATH = "model/best_model.pth"

class AnalysisRequest(BaseModel):
    patient_ids: List[str]

class FileResult(BaseModel):
    filename: str
    prediction: str
    confidence: float
    abnormal_probability: float
    heatmap_grid: Optional[List[List[float]]] = None
    error: Optional[str] = None

class PatientResult(BaseModel):
    patient_id: str
    files: List[FileResult]
    summary: str # "Abnormal" if any file is abnormal, else "Normal"

@app.on_event("startup")
async def startup_event():
    global engine
    print(f"Starting UP... Loading Model from {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        strategy = get_system_strategy()
        engine = InferenceEngine(MODEL_PATH, strategy)
        if engine.model or engine.session:
            print(f"Model loaded successfully on {engine.strategy['description']}")
        else:
            print("Failed to load model on startup.")
    else:
        print(f"Model path {MODEL_PATH} not found.")

@app.post("/analyze", response_model=List[PatientResult])
async def analyze_patients(request: AnalysisRequest = Body(...)):
    if not engine or (not engine.model and not engine.session):
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    
    # Base directory for organized scans
    # Assumption: This server runs with access to the same volume/path
    # Path relative to project root usually
    ORGANIZED_SCANS_DIR = os.path.abspath(os.path.join(current_dir, "../organized_scans"))
    
    for pid in request.patient_ids:
        patient_res = PatientResult(patient_id=pid, files=[], summary="Normal")
        
        # Find folder starting with PID
        # e.g. PID-12345_Name
        search_pattern = os.path.join(ORGANIZED_SCANS_DIR, f"{pid}*")
        folders = glob.glob(search_pattern)
        
        if not folders:
            # No folder found for this patient
            results.append(patient_res)
            continue
            
        target_folder = folders[0] # Take first match
        
        # Scan DICOM files
        dicom_files = glob.glob(os.path.join(target_folder, "*.dicom")) + \
                      glob.glob(os.path.join(target_folder, "*.dcm"))
                      
        has_abnormal = False
        
        for dcm_file in dicom_files:
            file_res = FileResult(
                filename=os.path.basename(dcm_file),
                prediction="Unknown", 
                confidence=0.0,
                abnormal_probability=0.0
            )
            
            # Preprocess
            input_data, error = preprocess_image(dcm_file)
            
            if error:
                file_res.error = error
                patient_res.files.append(file_res)
                continue
                
            try:
                # Inference + Heatmap
                # Use heatmap generation if available (PyTorch)
                if engine.model:
                     logits, heatmap = engine.predict_with_heatmap(input_data)
                else:
                     logits = engine.predict(input_data)
                     heatmap = None
                
                probs = softmax(logits)
                abnormal_prob = float(probs[0, 1])
                normal_prob = float(probs[0, 0])
                
                file_res.abnormal_probability = abnormal_prob
                
                threshold = 0.61
                if abnormal_prob >= threshold:
                    file_res.prediction = "Abnormal"
                    file_res.confidence = abnormal_prob
                    has_abnormal = True
                else:
                    file_res.prediction = "Normal"
                    file_res.confidence = normal_prob
                
                if heatmap is not None:
                    # Convert numpy array to list of lists for JSON
                    # Precision: float32, maybe round to 4 decimals to save bandwidth
                    file_res.heatmap_grid = np.round(heatmap.astype(float), 4).tolist()
                    
            except Exception as e:
                file_res.error = str(e)
            
            patient_res.files.append(file_res)
            
        if has_abnormal:
            patient_res.summary = "Abnormal"
            
        results.append(patient_res)
        
    return results

@app.get("/health")
def health_check():
    status = "healthy" if engine and (engine.model or engine.session) else "degraded"
    return {"status": status, "device": engine.device if engine else "none"}
