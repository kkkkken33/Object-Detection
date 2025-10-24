## Dataset
1. Download dataset provided in kaggle: https://www.kaggle.com/competitions/hkustgz-aiaa-4220-2025-fall-project-2/data
2. move "data" dir to root dir

## Usage
1. Install requirement
   ```sh
   pip install -r requirement.txt
   ```
2. Train and validate model with Ultralytics
   ```sh
   python main.py
   ```
3. Inference on test dataset
   ```sh
   python inference.py
   ```
4. Convert result to submission format
   ```sh
   python convert_to_submission.py --pred inference_results.json --output submission.csv
   ```
## Logs
  In dir "runs/" logs the training and validation history, including model weights, visual samples, confusion matrix, etc.

## Models
  Latest trained model is in "runs/train2/weights/best.pt". Modify main.py to train a new one, val, or resume training.
