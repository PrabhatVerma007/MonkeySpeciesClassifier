# 🐵 Monkey Species Classification using Vision Transformer (ViT)

This project implements a monkey species classifier using the Vision Transformer (ViT) architecture. It leverages TensorFlow/Keras for model training and evaluation, with robust preprocessing and augmentation strategies. The model is trained on a dataset comprising images of various monkey species.

## 📁 Project Structure

- `keras-validation-monkey-vit.ipynb`: Notebook for dataset loading, preprocessing, training, validation, and result visualization using Keras.
- `vit-transformer.ipynb`: Notebook for building and training a Vision Transformer (ViT) model from scratch using TensorFlow/Keras.

## 🧠 Model Overview

### Vision Transformer (ViT)

The ViT model splits images into patches, encodes them via positional embeddings, and applies Transformer blocks for feature extraction, followed by classification.

**Key Components**:
- Patch extraction and embedding
- Transformer encoder layers (Multi-head attention + MLP)
- Classification head

## 📊 Dataset

- **Classes**: 18 monkey species
- **Images**: 24,000+ images (training, validation, and test splits)
- **Format**: Image folders with labeled subdirectories

## 🔧 Setup & Requirements

Install dependencies using pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```
## 🚀 Running the Notebooks

### Train & Evaluate with ViT:
- Open `vit-transformer.ipynb` to build and train the Vision Transformer model.
- Includes patch embedding, transformer encoding, and training loop.

### Validation with Keras:
- Open `keras-validation-monkey-vit.ipynb` to validate the model using Keras utilities.
- Includes visualization of confusion matrix and accuracy curves.

## 📈 Evaluation Metrics
- Accuracy  
- Confusion Matrix  
- Loss/Accuracy Curves  

## 📷 Sample Results
- Training and validation accuracy over epochs  
- Misclassified examples visualization  
- Confusion matrix heatmap  

## 📌 Notes
- Vision Transformer offers strong performance on structured image classification tasks.
- Hyperparameters like patch size, transformer depth, and number of attention heads can be tuned for improved results.

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
