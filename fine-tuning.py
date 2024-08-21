from ultralytics import YOLO
import torch
from torch.utils.tensorboard import SummaryWriter

path = 'C:/path'
def train_model():
    # Load a pretrained model (recommended for training)
    model = YOLO(f"{path}/best.pt")

    # For COCO Model
    # model = YOLO("yolov8n.pt")

    # For training from scratch
    # model = YOLO("yolov8n.yaml")

    # Check if GPU is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir="runs/yolov8_training")

    # Train the model
    results = model.train(data="polyp.yaml",
                          pretrained=True,
                          epochs=50,
                          imgsz=640,
                          device=device,
                          project="runs",
                          name=f"{path}/fine-tuned",
                          plots=True,
                          verbose=True,
                          )

    # Close the TensorBoard writer
    writer.close()

if __name__=="__main__":
    train_model()
