# Image Caption Generator
This repository contains the code and resources for an Image Caption Generator that uses a Convolutional Neural Network (CNN) for image feature extraction and a Recurrent Neural Network (RNN) for generating descriptive captions. The model is trained on the Flickr8K dataset and utilizes a VGG16 model for feature extraction and an LSTM network for sequence prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

## Project Overview

The goal of this project is to create a system that can generate captions for images automatically. The system uses a deep learning approach, leveraging CNNs for feature extraction from images and LSTMs for generating textual descriptions. This project is useful in various applications such as generating captions for social media, aiding visually impaired individuals, and more.

## Dataset

The model is trained on the Flickr8K dataset, which consists of 8,000 images with five different captions each. The images and their corresponding captions are used to train the model.

- **Image dataset location:** `../input/flickr8k/Images`
- **Caption file location:** `../input/flickr8k-text/Flickr8k.token.txt`

## Model Architecture

1. **Feature Extraction:**  
   - VGG16 model is used for extracting features from the images. The last layer of the VGG16 model is removed, and the output is a feature vector of 4096 dimensions.
   
2. **Caption Generation:**  
   - An LSTM-based network is used for generating captions. The model takes in the image feature vector and the input text (caption sequence) and predicts the next word in the sequence.

3. **Training:**  
   - The model is trained by feeding the image feature vector and partial captions, and the output is the next word in the caption. The process is repeated for each word in the caption.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/image-caption-generator.git
   cd image-caption-generator
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Place the Flickr8K images and captions in the respective directories mentioned above.

## Usage

1. **Training the Model:**
   - To train the model, run the script:
     ```bash
     python train.py
     ```

2. **Generating Captions:**
   - After training, you can generate captions for new images by running:
     ```bash
     python generate_caption.py --image_path path_to_your_image
     ```

3. **Evaluating the Model:**
   - Evaluate the model on the test dataset by running:
     ```bash
     python evaluate.py
     ```

## Results

The model is capable of generating coherent captions for the images. Some sample outputs can be found in the `results/` directory. 

## Evaluation

The model's performance is evaluated using the BLEU score, which compares the generated captions to the reference captions. The BLEU score is calculated for the test set, and the average score is reported.

Example:
```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'sample', 'caption']]
candidate = ['this', 'is', 'a', 'generated', 'caption']
score = sentence_bleu(reference, candidate)
print(f"BLEU score: {score}")
```

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.
