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

def predict_one(tf: str, expression_level_after_perturbation: float, cell_type: str, timepoint: int, num_steps: int):
    """Predict expression after TF perturbation

    Args:
        tf (str): index of the TF such that train.var.index[tf_index] is the name of the TF
        expression_level_after_perturbation (_type_): expression level of the perturbed TF after perturbation
        cell_type (str): starting cell type
        timepoint (int): starting timepoint
        num_steps (int): how many time-steps ahead to predict
    """
    current_cell = train.obs_names[(train.obs["cell_type"]==cell_type) & (train.obs["timepoint"]==timepoint)]
    current_cell = np.random.choice(current_cell, size = 1)
    for _ in range(num_steps):
        # Find descendents and select one at random
        next_cells = [c for c in train.obs_names if train.obs.loc[c, "matched_control"] in current_cell]
        if len(next_cells) == 0:
            return np.nan
        current_cell = np.random.choice(next_cells, size = 1)
    return train[current_cell, :].X
    
def predict_many(num_cells_to_simulate: int, tf: str, expression_level_after_perturbation: float, cell_type: str, timepoint: int, num_steps: int):
    average_prediction = np.zeros(train.n_vars)
    for j in range(num_cells_to_simulate):
        a_single_prediction = predict_one(tf, expression_level_after_perturbation, cell_type, timepoint, num_steps)
        if any([pd.isnull(x) for x in a_single_prediction]):
            num_cells_to_simulate -= 1
        else:
            average_prediction += a_single_prediction
    return average_prediction / num_cells_to_simulate

print("Running simulations")
predictions = anndata.AnnData(
    X = np.zeros((predictions_metadata.shape[0], train.n_vars)),
    obs = predictions_metadata,
    var = train.var,
)

def get_unique_rows(df, factors): 
    return df[factors].groupby(factors).mean().reset_index()

for _, current_prediction_metadata in get_unique_rows(predictions.obs, ['perturbation', "expression_level_after_perturbation", 'prediction_timescale', 'timepoint', 'cell_type']).iterrows():
    print("Predicting " + current_prediction_metadata.to_csv(sep=" "))
    prediction_index = \
        (predictions.obs["cell_type"]==current_prediction_metadata["cell_type"]) & \
        (predictions.obs["timepoint"]==current_prediction_metadata["timepoint"]) & \
        (predictions.obs["perturbation"]==current_prediction_metadata["perturbation"]) & \
        (predictions.obs["expression_level_after_perturbation"]==current_prediction_metadata["expression_level_after_perturbation"]) & \
        (predictions.obs["prediction_timescale"]==current_prediction_metadata["prediction_timescale"]) 
    train_index = (train.obs["cell_type"]==current_prediction_metadata["cell_type"]) & (train.obs["timepoint"]==current_prediction_metadata["timepoint"])
    goi, level, prediction_timescale = current_prediction_metadata[['perturbation', "expression_level_after_perturbation", 'prediction_timescale']]
    for i in np.where(prediction_index)[0]:
        predictions[i, :].X = predict_many(kwargs["num_cells_to_simulate"], goi, level, current_prediction_metadata["cell_type"], current_prediction_metadata["timepoint"], prediction_timescale) 
    
print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")