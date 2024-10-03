import anndata
import json
import os
import numpy as np
import pandas as pd
import scanpy as sc
import sys
sys.path.append(".")

train = sc.read_h5ad("from_to_docker/train.h5ad")

predictions_metadata = pd.read_csv("from_to_docker/predictions_metadata.csv")
for c in ['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation", 'prediction_timescale']:
    assert c in predictions_metadata.columns, f"For the timeseries baseline that returns the unperturbed timeseries, predictions_metadata.columns is required to contain {c}."

# This should be taken care of early on in ggrn.api.fit().
assert "matched_control" in train.obs.columns, "For the timeseries baseline that returns the unperturbed timeseries, the dataset must have matched controls for each timepoint except the first."

with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)

with open(os.path.join("from_to_docker/ggrn_args.json")) as f:
    ggrn_args = json.load(f)


defaults = {
    "num_cells_to_simulate": 100,
}
for k in defaults:
    if k not in kwargs:
        kwargs[k] = defaults[k]

cell_type_means = {
    ct: np.array(train[train.obs["cell_type"]==ct, :].X.mean(axis = 0))
    for ct in train.obs["cell_type"].unique()
}
def predict_many(tf: str, expression_level_after_perturbation: float, cell_type: str, timepoint: int, num_steps: int):
    """Predict expression after TF perturbation

    Args:
        tf (str): index of the TF such that train.var.index[tf_index] is the name of the TF
        expression_level_after_perturbation (_type_): expression level of the perturbed TF after perturbation
        cell_type (str): starting cell type
        timepoint (int): starting timepoint
        num_steps (int): how many time-steps ahead to predict
    """
    current_cells = train.obs_names[(train.obs["cell_type"]==cell_type) & (train.obs["timepoint"]==timepoint)]
    for _ in range(num_steps):
        next_cells = np.unique([c for c in train.obs_names if train.obs.loc[c, "matched_control"] in current_cells])
        if len(next_cells) == 0:
            return cell_type_means[cell_type]
        next_cell_type = train.obs.loc[next_cells, "cell_type"].value_counts().idxmax()
        return cell_type_means[next_cell_type]

print("Running simulations")
predictions = anndata.AnnData(
    X = np.zeros((predictions_metadata.shape[0], train.n_vars)),
    obs = predictions_metadata,
    var = train.var,
)
predictions.obs["error_message"] = ""
for _, current_prediction_metadata in predictions.obs[['perturbation', "expression_level_after_perturbation", 'prediction_timescale', 'timepoint', 'cell_type']].drop_duplicates().iterrows():
    prediction_index = \
        (predictions.obs["cell_type"]==current_prediction_metadata["cell_type"]) & \
        (predictions.obs["timepoint"]==current_prediction_metadata["timepoint"]) & \
        (predictions.obs["perturbation"]==current_prediction_metadata["perturbation"]) & \
        (predictions.obs["expression_level_after_perturbation"]==current_prediction_metadata["expression_level_after_perturbation"]) & \
        (predictions.obs["prediction_timescale"]==current_prediction_metadata["prediction_timescale"]) 
    train_index = (train.obs["cell_type"]==current_prediction_metadata["cell_type"]) & (train.obs["timepoint"]==current_prediction_metadata["timepoint"])
    goi, level, prediction_timescale = current_prediction_metadata[['perturbation', "expression_level_after_perturbation", 'prediction_timescale']]
    for i in prediction_index[prediction_index].index:
        try:
            predictions[i, :].X = predict_many(goi, level, current_prediction_metadata["cell_type"], current_prediction_metadata["timepoint"], prediction_timescale) 
        except Exception as e:
            predictions.obs.loc[i, "error_message"] = repr(e)
            predictions[i, :].X = np.nan

print("Summary of errors:")
print(predictions.obs["error_message"].value_counts())
print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")