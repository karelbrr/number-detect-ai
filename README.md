# MNIST Classifier – Neural Network in NumPy

This project implements a two-layer fully connected neural network from scratch using only Python's **NumPy** library. The model is designed to train on and classify handwritten digits from the popular MNIST dataset.

## Key Features

*   **Custom Implementation:** Complete forward propagation and backpropagation built entirely without deep learning frameworks.
*   **Model Persistence:** Automatically saves weights and biases to a `.npz` file upon successful training. On subsequent runs, it loads the saved model to skip retraining.
*   **Vectorization:** Uses NumPy matrix operations for efficient computations.

## Network Architecture

The network consists of three layers configured as follows:

| Layer | Type | Neurons | Activation Function |
|---|---|---|---|
| **Input** | Input Data (28x28 pixel images) | 784 | N/A |
| **Hidden** | Fully Connected (Dense) | 64 | ReLU |
| **Output** | Fully Connected (Dense) | 10 | Softmax |

## Requirements

To run this project, you need Python and NumPy installed.

```bash
pip install -r requirements.txt
```

## Data Structure

The script expects a dataset file in the same directory:
*   `mnist_train_copy.csv`

The CSV file must contain a header row followed by pixel data, where the first column is the label (digit 0–9) and the remaining 784 columns represent pixel values (0–255). Data is normalized during loading by dividing by 255.0.

## Usage

Run the main Python script from your terminal:

```bash
python main.py
```

### Execution Flow:

1.  **Saved Model Check:** The script checks if `my_trained_data.npz` exists in the project directory.
2.  **Loading:** If found, it loads the pre-trained parameter matrices.
3.  **Training:** If no saved model is found, it loads the CSV dataset, initializes parameters, and runs Gradient Descent for 500 iterations at a learning rate of 0.1. Accuracy is printed every 10 epochs.
4.  **Saving:** Once training completes, the parameters are saved to `my_trained_data.npz` for future reuse.
