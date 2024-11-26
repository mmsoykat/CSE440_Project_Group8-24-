from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11m-seg.pt")  # Replace with the correct path to your weights file.

# Train the model
model.train(
    data="dataset_custom.yaml",  # Path to the dataset YAML configuration file.
    imgsz=640,                  # Image size.
    batch=8,                    # Batch size.
    epochs=100,                 # Number of training epochs.
    workers=0,                  # Number of workers.
    device="cpu"               # Use 'cpu' for CPU or 'cuda' for GPU training.
)
