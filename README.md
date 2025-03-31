# Crop Disease Detection System

This repository contains two main files:
- **`train.py`** – Script to train the crop disease detection model.
- **`app.py`** – Streamlit application for running the model and performing disease detection.

## Getting Started

### 1. Download the Dataset

Download the dataset from [Mendeley Data](https://data.mendeley.com/datasets/bwh3zbpkpv/1).

> **About the Dataset:**  
> This dataset contains high-quality images of crop leaves, including both healthy and diseased samples. The images are captured under controlled conditions and annotated with various disease labels, making it a valuable resource for training robust crop disease detection models.

> **Important:** Please ensure that the dataset file names match the naming convention used in `train.py`. If they are not in the correct format, rename them accordingly to avoid errors during training.

### 2. Train the Model

Run the training script to train your model and generate the best model file:

```bash
python train.py
```
> This process will create a file (e.g., best_model.pth) that the application will use for prediction.


### 3. Run the Application

Once the model is trained, launch the Streamlit application:

```bash
streamlit run app.py
```

This will start the Crop Disease Detection System.

### Dependencies
Install the required packages using:

```bash
pip install streamlit torch torchvision Pillow numpy opencv-python matplotlib seaborn pandas
```

