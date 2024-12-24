import os 
import pandas as pd
from anndata import AnnData
from anndata.io import read_h5ad

out_dir = "from_to_docker"
train = read_h5ad("from_to_docker/train.h5ad")
predictions_metadata = pd.read_csv("from_to_docker/predictions_metadata.csv")
predicted_expression = pd.read_csv("from_to_docker/predicted_expression.csv") # RNAForecaster records data as genes by cells
pred = AnnData(obs = predictions_metadata, var = train.var, X = predicted_expression.values.T)
pred.write_h5ad(os.path.join(out_dir, "predictions.h5ad"))