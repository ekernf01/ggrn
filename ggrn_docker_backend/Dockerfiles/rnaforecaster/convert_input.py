import os 
import json 
import pandas as pd
from anndata import AnnData
from anndata.io import read_h5ad

# Read the inputs from PEREGGRN: data, desired predictions, and arguments.
out_dir = "from_to_docker"
train = read_h5ad("from_to_docker/train.h5ad")

predictions_metadata = pd.read_csv("from_to_docker/predictions_metadata.csv")

with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)

if kwargs["matching_method"]=="steady_state":
    raise ValueError(f"steady state matching may not be used with rnaforecaster. If you really want to do this, please file a github issue.")

with open(os.path.join("from_to_docker/ggrn_args.json")) as f:
    ggrn_args = json.load(f)

defaults = {
  "pca_dim": 10,
  "ridge_penalty": 0.01,
}
for k in defaults:
    if k not in kwargs:
        kwargs[k] = defaults[k]

# RNAForecaster wants data in genes by cells format
has_matched_control = pd.notnull(train.obs["matched_control"])
try:
    train.X = train.X.toarray()
except AttributeError:
    pass
pd.DataFrame(train[train.obs.loc[has_matched_control, "matched_control"], :].X.T).to_csv("from_to_docker/0.csv", index=False)
pd.DataFrame(train[train.obs.loc[has_matched_control, :].index,           :].X.T).to_csv("from_to_docker/1.csv", index=False)
train.var.to_csv("from_to_docker/gene_metadata.csv", index=True)
train.obs.to_csv("from_to_docker/training_metadata.csv", index=True)
