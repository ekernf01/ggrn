import os
import json
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
train = sc.read_h5ad("from_to_docker/train.h5ad")
predictions_metadata = pd.read_csv("predictions_metadata.csv")
assert all(predictions_metadata.columns == np.array(['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation", 'prediction_timescale']))
with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)
# Since it's just a demo, return all zeroes.
predictions = anndata.AnnData(
    X = np.zeros((predictions_metadata.shape[0], train.n_vars)),
    obs = predictions_metadata,
    var = train.var,
)
predictions.write_h5ad("from_to_docker/predictions.h5ad")