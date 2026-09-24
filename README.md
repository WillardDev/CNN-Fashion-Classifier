# CNN on Fashion MNIST

A Jupyter notebook demonstrating a Convolutional Neural Network (CNN) for image
classification on the Fashion MNIST dataset, following a general deep-learning
workflow guideline.

## What is this?

Fashion MNIST is a dataset of 70,000 grayscale 28×28 images of clothing across
10 classes:

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

It is designed as a drop-in replacement for the classic MNIST digits dataset but
is a more challenging, realistic classification task.

## Why a CNN?

Use a CNN when:
- The dataset is large with complex, highly non-linear patterns — especially
  unstructured data (images, audio, text).
- Hand-crafted features are hard to design.

## Repository contents

- `fashion_mnist_cnn.ipynb` — the main notebook implementing the full workflow.

## Notebook workflow

The notebook follows this guideline checklist:

1. **Data cleaning** — check for missing values, normalise pixels to [0, 1],
   add a channel dimension for the CNN.
2. **EDA** — verify data volume and class balance, sanity-check labels with
   random image samples.
3. **Train/validation/test split** — stratified split with a dedicated
   validation set used for **early stopping** (training is expensive, so CV
   alone is not used).
4. **Data augmentation** — mild translation/flip to grow the effective dataset
   and reduce overfitting.
5. **Model training** — build a CNN (Conv + BN + Pool + Dropout blocks) trained
   with the Adam optimiser; tune architecture, learning rate, batch size and
   regularisation.
6. **Evaluation metrics** — Accuracy, Precision, Recall, F1, ROC-AUC, plus
   train/validation loss and accuracy curves.
7. **Error analysis** — inspect misclassified examples and the confusion matrix
   to decide whether labels, augmentation or architecture need adjustment.
8. **Summary** — pros/cons and real-world examples.

## Requirements

- Python 3.9+
- TensorFlow (>= 2.6)
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn

Install dependencies:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

## Running the notebook

```bash
jupyter notebook fashion_mnist_cnn.ipynb
```

or, in VS Code, open the notebook and run all cells. The dataset is downloaded
automatically by TensorFlow/Keras on first run.

## Expected results

With the included architecture you should reach roughly **~92–94% test
accuracy**. Most misclassifications happen between similar-looking classes
(e.g. Pullover vs Coat, T-shirt vs Shirt).

## Notes

- Results may vary slightly due to model weight initialisation.
- The notebook uses a fixed random seed for reproducibility.