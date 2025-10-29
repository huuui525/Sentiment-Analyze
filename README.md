# MovieSentimentBERT

Fine-tuning BERT for IMDb movie review sentiment analysis with 92%+ accuracy.

## About

This project demonstrates how to fine-tune the BERT transformer model for binary sentiment classification on the IMDb movie review dataset. The system processes raw text reviews and predicts whether they express positive or negative sentiment using state-of-the-art natural language processing techniques.

## Features

-  **State-of-the-art BERT architecture** - Leverages pre-trained transformer models
-  **90%+ test accuracy** - High-performance sentiment classification
-  **Complete ML pipeline** - Data exploration → preprocessing → training → inference
-  **Easy deployment** - Ready-to-use inference API
-  **Comprehensive analysis** - Detailed EDA and model evaluation

## Project Structure

sentiment_analysis/
├── src/
│ ├── data_exploration.py # Data analysis and visualization
│ ├── data_preprocessing.py # Dataset and preprocessing
│ ├── train.py # Model training and evaluation
│ └── inference.py # Prediction interface
├── models/ # Saved models 
├──data/ #Data analysis
└── requirements.txt # Python dependencies

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/huuui525/sentiment_analysis.git
cd sentiment_analysis

# Install dependencies
pip install -r requirements.txt

## Basic Usage

# Run data exploration (generates analysis plots)
python src/data_exploration.py

# Train the model (saves to models/ directory)
python src/train.py

# Make predictions using trained model
python src/inference.py

## Technical Details

·Model: BERT-base-uncased (fine-tuned)
·Dataset: IMDb Movie Reviews (50,000 samples)
·Framework: PyTorch + Hugging Face Transformers
·Task: Binary sentiment classification (Positive/Negative)

## Dependencies

Python 3.8+
PyTorch 2.0+
Transformers 4.30+
Datasets 2.12+
See requirements.txt for complete dependency list.
