# Deep Learning PR 5: Sentiment Analysis with RNN, LSTM and GRU

A hands-on Deep Learning project that classifies IMDB movie reviews as **Positive** or **Negative** using Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU).

---

## 📌 Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) task used to identify the emotional tone of text.

In this project, we use the IMDB Movie Reviews dataset to predict whether a movie review expresses positive or negative sentiment.

We compare the following deep learning models:

1. ANN Baseline
2. Simple Recurrent Neural Network (SimpleRNN)
3. Long Short-Term Memory (LSTM)
4. Gated Recurrent Unit (GRU)
5. Stacked LSTM

## 🎯 Objectives

- Load and inspect the IMDB dataset.
- Clean and preprocess movie reviews.
- Remove duplicate reviews.
- Perform text tokenisation.
- Convert text into numerical sequences.
- Apply padding and truncation.
- Understand ANN and RNN architectures.
- Implement RNN forward propagation using NumPy.
- Understand Backpropagation Through Time (BPTT).
- Study vanishing and exploding gradients.
- Implement LSTM and GRU architectures.
- Train and evaluate different models.
- Compare model performance using evaluation metrics.
- Perform error analysis and predict sentiment for new reviews.

## 📂 Dataset

**Dataset Name:** IMDB Dataset of 50K Movie Reviews

**Source:** Kaggle

[Download IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

The dataset contains 50,000 movie reviews.

| Column | Description |
|---|---|
| `review` | Movie review text |
| `sentiment` | Positive or Negative sentiment |

### Sentiment Labels

| Sentiment | Numeric Value |
|---|---:|
| Negative | 0 |
| Positive | 1 |

### Download Dataset Using KaggleHub

Install KaggleHub:

```python
%pip install kagglehub
```

Download and load the dataset:

```python
import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download(
    "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
)

print("Dataset path:", path)

DATA_PATH = os.path.join(path, "IMDB Dataset.csv")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
display(df.head())
```

## 📁 Project Structure

```text
PR-5-DL/
│
├── Exam 5/
│   └── DL_PR5_Complete_Deep_Exam_Expanded.ipynb
│
└── README.md
```

The dataset is downloaded separately using KaggleHub.

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Practical implementation |
| TensorFlow | Deep learning framework |
| Keras | Neural network development |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Scikit-learn | Model evaluation |
| KaggleHub | Dataset downloading |

## ⚙️ Installation and Setup

### 1. Install Python

Install Python and configure it with Jupyter Notebook or VS Code.

### 2. Install Required Libraries

Run the following command:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn kagglehub jupyter
```

For Jupyter Notebook, you can also use:

```python
%pip install tensorflow numpy pandas matplotlib seaborn scikit-learn kagglehub
```

### 3. Import Libraries

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("TensorFlow Version:", tf.__version__)
```

## 🚀 Project Workflow

### Task 0: Imports, Reproducibility and Configuration

- Import required libraries.
- Set random seeds.
- Configure vocabulary size and embedding dimensions.
- Set maximum sequence length.
- Configure batch size and training epochs.

### Task 1: Text Loading and Preprocessing

- Load and inspect the dataset.
- Check missing values.
- Inspect sentiment distribution.
- Remove exact duplicate review-label pairs.
- Clean review text without removing sentiment-bearing words.
- Calculate review lengths.
- Visualise review-length distribution.
- Split training and testing data.
- Fit the tokenizer on training data only.
- Convert text into sequences.
- Apply padding and truncation.

**Important:** Words such as `not`, `never`, `bad`, and `excellent` should be preserved because they carry sentiment information. The test dataset must not be used to fit the tokenizer.

### Task 2: RNN Introduction and ANN Baseline

- Understand the difference between ANN and RNN.
- Build an ANN baseline model.
- Understand sequential data.
- Study RNN architecture.
- Implement manual RNN forward propagation using NumPy.
- Compare manual calculations with Keras results.
- Calculate model parameters.

### Task 3: SimpleRNN Sentiment Classification

- Understand different types of RNN.
- Build an embedding-based SimpleRNN model.
- Train the model.
- Evaluate model performance.
- Display training and validation curves.
- Calculate classification metrics.

### Task 4: Backpropagation Through Time (BPTT)

- Understand BPTT.
- Study vanishing gradients.
- Study exploding gradients.
- Inspect gradient flow.
- Experiment with different sequence lengths.
- Understand gradient clipping.

### Task 5: LSTM (Long Short-Term Memory)

- Understand LSTM architecture.
- Study forget gate.
- Study input gate.
- Study candidate memory.
- Study output gate.
- Understand cell state.
- Implement a manual LSTM calculation.
- Build and train an LSTM model.
- Experiment with stacked LSTM.
- Evaluate model performance.

### Task 6: GRU (Gated Recurrent Unit)

- Understand GRU architecture.
- Study update gate.
- Study reset gate.
- Compare GRU and LSTM.
- Calculate model parameters.
- Build and train a GRU model.
- Evaluate model performance.
- Experiment with different hidden units.

### Task 7: Final Model Comparison

- Compare ANN, SimpleRNN, LSTM and GRU.
- Compare accuracy, precision, recall and F1-score.
- Compare ROC-AUC scores.
- Display ROC curves.
- Analyse false-positive and false-negative predictions.
- Test models using new movie reviews.
- Recommend the best model based on actual results.

## 🧠 Models and Architectures

### 1. ANN Baseline

```text
Input Review
     |
 Embedding
     |
  Flatten
     |
Dense (64, ReLU)
     |
Dense (1, Sigmoid)
     |
Positive / Negative
```

### 2. SimpleRNN

```text
Input Review
     |
 Embedding
     |
SimpleRNN (64 Units)
     |
Dense (1, Sigmoid)
     |
Positive / Negative
```

### 3. LSTM

```text
Input Review
     |
 Embedding
     |
 LSTM (64 Units)
     |
Dense (1, Sigmoid)
     |
Positive / Negative
```

### 4. GRU

```text
Input Review
     |
 Embedding
     |
 GRU (64 Units)
     |
Dense (1, Sigmoid)
     |
Positive / Negative
```

### 5. Stacked LSTM

```text
Input Review
     |
 Embedding
     |
 LSTM (64 Units)
     |
 LSTM (32 Units)
     |
Dense (1, Sigmoid)
     |
Positive / Negative
```

## 📊 Evaluation Metrics

The following evaluation metrics are used:

| Metric | Description |
|---|---|
| Accuracy | Measures overall correct predictions |
| Precision | Measures correct positive predictions |
| Recall | Measures correctly identified positive samples |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Measures classification performance across thresholds |
| Confusion Matrix | Displays correct and incorrect predictions |

Additional analysis:

- Training and validation accuracy.
- Training and validation loss.
- ROC curves.
- Gradient analysis.
- Review-length robustness.
- False-positive and false-negative analysis.
- Model parameter comparison.
- Training-time comparison.

## 📈 Expected Outputs

After successfully executing the notebook, the following outputs are generated:

- Dataset inspection.
- Missing-value analysis.
- Duplicate review analysis.
- Cleaned text examples.
- Review-length distribution graphs.
- Tokenised and padded sequences.
- ANN baseline results.
- SimpleRNN model results.
- LSTM model results.
- GRU model results.
- Stacked LSTM results.
- Training and validation graphs.
- Confusion matrices.
- ROC curves.
- Classification reports.
- Model comparison table.
- New review sentiment predictions.

## 📋 Results

Fill in the following table after executing the notebook.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| ANN | Actual result | Actual result | Actual result | Actual result | Actual result |
| SimpleRNN | Actual result | Actual result | Actual result | Actual result | Actual result |
| LSTM | Actual result | Actual result | Actual result | Actual result | Actual result |
| GRU | Actual result | Actual result | Actual result | Actual result | Actual result |
| Stacked LSTM | Actual result | Actual result | Actual result | Actual result | Actual result |

**Note:** Actual evaluation results must be added after running the notebook. No estimated scores are presented as experimental results.

## 📚 Key Concepts Covered

- Deep Learning
- Natural Language Processing (NLP)
- Sentiment Analysis
- Text Preprocessing
- Tokenisation
- Padding and Truncation
- Word Embeddings
- Artificial Neural Networks
- Recurrent Neural Networks
- SimpleRNN
- Forward Propagation
- Backpropagation Through Time
- Vanishing and Exploding Gradients
- Gradient Clipping
- Long Short-Term Memory
- LSTM Gates
- Gated Recurrent Unit
- GRU Gates
- Binary Classification
- Confusion Matrix
- Precision, Recall and F1-score
- ROC-AUC
- Model Comparison

## 📝 Conclusion

This project demonstrates how Deep Learning techniques can be used for sentiment analysis of movie reviews.

The project starts with text preprocessing and an ANN baseline, followed by SimpleRNN, LSTM and GRU architectures.

SimpleRNN introduces the basic concept of sequential learning, while LSTM and GRU use gating mechanisms to manage information over longer sequences.

By comparing different architectures and evaluation metrics, the project helps us understand the advantages and limitations of different recurrent neural network models.

The final model recommendation is based on the actual experimental results.

## 🔧 Troubleshooting

### 1. NameError: name 'df' is not defined

Run the dataset-loading cell before executing preprocessing or duplicate-removal code.

### 2. FileNotFoundError

Check that the dataset path points to the correct `IMDB Dataset.csv` file.

### 3. KaggleHub Import Error

Install KaggleHub:

```python
%pip install kagglehub
```

### 4. TensorFlow Import Error

Install TensorFlow:

```python
%pip install tensorflow
```

Restart the Jupyter kernel if necessary.

### 5. Memory Error

- Reduce batch size.
- Reduce maximum sequence length.
- Reduce vocabulary size.
- Reduce the number of model units.
- Close unnecessary applications.

### 6. Missing Variables

Run the notebook cells from beginning to end in the correct order. Restarting the kernel removes variables and trained models from memory.

## 👨‍💻 Author

**Name:** Abhiraj Medhat

**Course:** Deep Learning – Practical 5

**Project:** Sentiment Analysis with RNN, LSTM and GRU

**Programme:** B.Tech in Computer Engineering (AI & ML)

**Institution:** Uka Tarsadia University, Gujarat, India

## 🔗 References

1. [Kaggle – IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. [TensorFlow Documentation](https://www.tensorflow.org/)
3. [Keras Documentation](https://keras.io/)
4. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
5. [NumPy Documentation](https://numpy.org/doc/)

---

**Project:** Deep Learning Practical 5  
**Topic:** Sentiment Analysis with RNN, LSTM and GRU
