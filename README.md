# AI-Generated Image Classification

## Project Overview
This repository contains a deep learning model that detects whether an image has been AI-generated or is authentic. The model achieves **97% accuracy** by combining convolutional neural networks (CNNs) with Vision Transformer (ViT) architectures, trained on the CIFAKE dataset.

## Usage
1. Pull the Docker Image:
```bash
docker pull yuvraj131204/ai-generated-image-classification:latest
```

2. Run the docker Image:
```bash
docker run --name containername -p 3625:5000 -d yuvraj131204/ai-generated-image-classification:latest
```

stopping the container & deleting container & image after usage:
```bash
docker stop containername
docker rm containername
docker rmi yuvraj131204/ai-generated-image-classification:latest
```

## Training
To retrain the model:

1. Clone the repository:
```bash
git clone https://github.com/YuvrajBalagoni13/AI_Image_Classification.git
cd AI_Image_Classification
```

2. Run this for training CNN model:
```bash
python Scripts/sweep_runner.py 
```

Run this for CNN + Vision Transformer model:
```bash
python HybridModel/sweep_runner.py
```
