from ultralytics import YOLO
import torch

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device.upper()}")

    model = YOLO('yolov8n.pt').to(device)
    
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        augment=True,
        lr0=0.01,
        device=device
    )

if __name__ == "__main__":
    train()