from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model on the dataset for 100 epochs
if __name__ == "__main__":
    train_results = model.train(
        data="config.yml",  # Path to dataset configuration file
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size for training
        device=0,  # Device to run on (e.g., 'cpu', 0(GPU))
        batch=16,  # Number of images to process at once
)