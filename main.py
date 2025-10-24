from ultralytics import YOLO

if __name__ == "__main__":
    # # Create a new YOLO model from scratch
    # model = YOLO("yolo11n.yaml")

    # # Load a pretrained YOLO model (recommended for training)
    # model = YOLO("yolo11n.pt")

    # Load a model
    model = YOLO("./runs/detect/train2/weights/best.pt")  # load a partially trained model

    # # Train the model using the 'coco8.yaml' dataset for 3 epochs
    # results = model.train(data="./data/dataset.yaml", epochs=100)

    # Resume training
    # results = model.train(resume=True, data="./data/dataset.yaml", epochs=100)

    # Evaluate the model's performance on the validation set
    results = model.val(data="./data/dataset.yaml", save_json=True)
    with open("val_results.txt", "w") as f:
        f.write(str(results.box.map))

    # Perform object detection on an image using the model
    # results = model("https://ultralytics.com/images/bus.jpg")

    # Export the model to ONNX format
    # success = model.export(format="onnx")
