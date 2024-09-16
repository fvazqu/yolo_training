from ultralytics import YOLO
import torch


def test_model():
    # Check if GPU is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLO(f"C:path/best.pt") # Remember to Change the Model to the one you want to test

    # Validate the model
    metrics = model.val(data="data.yaml",           # Change this to the .yaml file with the validation set you want to test
                        plots=True,
                        device=device,
                        name=f"/path to save results")              # Change this to the name of the model you are testing
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

if __name__=="__main__":
    test_model()
