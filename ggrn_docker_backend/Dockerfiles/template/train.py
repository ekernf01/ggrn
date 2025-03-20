import scanpy as sc
import os 
import json 
import pandas as pd
from anndata import AnnData
import numpy as np

# Read the inputs from PEREGGRN: data, desired predictions, and arguments.
out_dir = "from_to_docker"
train = sc.read_h5ad("from_to_docker/train.h5ad")

predictions_metadata = pd.read_csv("from_to_docker/predictions_metadata.csv")

with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)

with open(os.path.join("from_to_docker/ggrn_args.json")) as f:
    ggrn_args = json.load(f)

defaults = {
  "pca_dim": 10,
  "ridge_penalty": 0.01,
}
for k in defaults:
    if k not in kwargs:
        kwargs[k] = defaults[k]
# ================================ YOUR CODE HERE ================================


pred = AnnData(obs = predictions_metadata, var = train.var, X = np.zeros((predictions_metadata.shape[0], train.shape[1])))
# pred = yourmethod.train(train).predict(predictions_metadata)


# ================================================================================
pred.write_h5ad(os.path.join(out_dir, "predictions.h5ad"))
