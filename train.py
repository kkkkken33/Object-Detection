def train_yolo_model(epochs=50, batch_size=16, img_size=640, lr0=0.01):
    # Check for CUDA availability
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Define timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f'train_{timestamp}'
    
    # Load the model
    try:
        model = YOLO('yolo12n.pt')
        model_type = 'yolo12n'
    except Exception:
        model = YOLO('yolov8n.pt')
        model_type = 'yolov8n'
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=10,
        save=True,
        device=device,
        project=os.path.join(base_dir, 'runs'),
        name=run_name,
        lr0=lr0,
        lrf=0.01,
        plots=True,
        save_period=5
    )
    
    # Save the model
    model_save_path = os.path.join(model_save_dir, f"{model_type}_{timestamp}.pt")
    
    try:
        model.model.save(model_save_path)
    except AttributeError:
        try:
            model.save(model_save_path)
        except Exception:
            best_model_path = os.path.join(base_dir, 'runs', run_name, 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, model_save_path)
    
    return model