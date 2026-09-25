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
- `app.py` — Streamlit web app for classifying uploaded clothing photos.
- `requirements.txt` — all dependencies for both the notebook and the app.
- `samples/` — example clothing photos to try.

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
7. **Error analysis & reduction** — rank confusion pairs, check prediction
   confidence, inspect misclassified examples, then reduce errors with
   test-time augmentation and targeted fine-tuning.
8. **Summary** — pros/cons and real-world examples.

## Requirements

- Python 3.9+
- TensorFlow (>= 2.15)
- Streamlit (>= 1.35)
- NumPy, Pandas, Pillow, Altair, Matplotlib, Seaborn
- scikit-learn

Homebrew's Python is externally managed (PEP 668), so install everything into a
project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel so the notebook can use it:

```bash
python -m ipykernel install --user --name fashion-cnn --display-name "Fashion CNN (.venv)"
```

## Running the notebook

```bash
source .venv/bin/activate
jupyter notebook fashion_mnist_cnn.ipynb
```

or, in VS Code, open the notebook and select the **Fashion CNN (.venv)**
kernel. The dataset is downloaded automatically by TensorFlow/Keras on first
run.

## Streamlit app

The app lets you upload a clothing photo (or pick one from `samples/`) and
shows the predicted class, its confidence, and a bar chart of the top-3 class
probabilities.

1. Train the model in the notebook and run the **Save the trained model** cell
   (creates `fashion_cnn.keras`).
2. Start the app:

```bash
source .venv/bin/activate
streamlit run app.py
```

3. Open http://localhost:8501 in your browser.

The app automatically crops the item, pads it to a square, resizes to 28×28 and
inverts bright backgrounds so external photos match the Fashion MNIST training
style. If TensorFlow or the saved model is missing, the app shows setup
instructions instead of crashing.

## Expected results

With the included architecture you should reach roughly **~92–94% test
accuracy**. Most misclassifications happen between similar-looking classes
(e.g. Pullover vs Coat, T-shirt vs Shirt).

## Notes

- Results may vary slightly due to model weight initialisation.
- The notebook uses a fixed random seed for reproducibility.