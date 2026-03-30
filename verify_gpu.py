import torch
import sys
from ultralytics import YOLO

def verify_gpu():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Test with YOLO
        print("\nVerifying Ultralytics YOLO11...")
        model = YOLO("yolo11n.pt")
        # Check if the model is on CUDA
        model.to('cuda')
        print(f"YOLO11 Model on Device: {next(model.model.parameters()).device}")
        
        # Simple inference test
        print("Running a sample inference on GPU...")
        import numpy as np
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model.predict(dummy_img, device='cuda', verbose=False)
        print("GPU Inference Successful!")
    else:
        print("CUDA is NOT available. Please check drivers and CUDA toolkit installation.")

if __name__ == "__main__":
    verify_gpu()
