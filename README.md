from pathlib import Path

# Load the uploaded notebook file
notebook_path = Path("/mnt/data/Garbage Classification using Transfer learning code.ipynb")

# Read and parse the notebook
import nbformat

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

# Extract all code cells for analysis
code_cells = [cell['source'] for cell in notebook.cells if cell.cell_type == 'code']

# Show a preview of the first few code cells for initial understanding
code_cells[:5]

import numpy as np, matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf, keras, sklearn, gradio


dataset_dir= r"C:\\Users\\Edunet Foundation\\Downloads\\project\\garbage\\TrashType_Image_Dataset"
image_size = (124, 124)
batch_size = 32
seed = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)

val_batches = tf.data.experimental.cardinality(val_ds)  

test_ds = val_ds.take(val_batches // 2)  
val_dat = val_ds.skip(val_batches // 2)  

test_ds_eval = test_ds.cache().prefetch(tf.data.AUTOTUNE)
