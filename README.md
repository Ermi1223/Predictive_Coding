# Predictive Coding MNIST

This project implements a minimal **Predictive Coding Neural Network** for MNIST digit classification using PyTorch.  
The model learns by iteratively minimizing prediction errors layer-by-layer, inspired by predictive coding theory in neuroscience.

---

## Project Structure

```
predictive_coding_mnist/
├── models/
│   ├── __init__.py
│   ├── predictive_layer.py
│   └── predictive_net.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   └── visualization.py
├── train.py
├── validate.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Features

- Predictive coding layers and network architecture
- Training loop for predictive coding with MNIST dataset
- Validation with confusion matrix and misclassified digit visualization
- Metrics reporting (precision, recall, F1-score per class)
- Saving the best model checkpoint based on validation accuracy

---

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/Ermi1223/Predictive_Coding.git
cd Predictive_Coding
```

2. Create and activate a Python virtual environment (recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script to train and validate the model:

```bash
python main.py
```

The script downloads the MNIST dataset automatically, trains the predictive coding network for 5 epochs,  
validates after each epoch, displays accuracy, confusion matrix, and saves the best model as `best_model.pth`.

---

## Notes

- Trained models, output images, and datasets are excluded from version control (`.gitignore`).
- Visualization windows will pop up during validation showing confusion matrix and misclassified images.
- Modify parameters like batch size, learning rate, or epochs inside `main.py` as needed.

---

## License

This project is licensed under the MIT License.

---
