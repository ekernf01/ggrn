import os
import json
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
train = sc.read_h5ad("from_to_docker/train.h5ad")
with open(os.path.join("from_to_docker/perturbations.json")) as f:
    perturbations = json.load(f)
with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)
# Since it's just a demo, return all zeroes.
predictions = anndata.AnnData(
    X = np.zeros((len(perturbations), train.n_vars)),
    obs = pd.DataFrame({
        "perturbation":                       [p[0] for p in perturbations], 
        "expression_level_after_perturbation":[p[1] for p in perturbations], 
    }), 
    var = train.var,
)
predictions.write_h5ad("from_to_docker/predictions.h5ad")