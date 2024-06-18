import anndata
import json
import os
import numpy as np
import pandas as pd
import scanpy as sc
import sys
sys.path.append(".")
from sckinetics import EM, tf_targets
from pereggrn_networks import LightNetwork


train = sc.read_h5ad("from_to_docker/train.h5ad")
try:
    train.X = train.X.toarray()
except AttributeError:
    pass
is_expressed=train.X.sum(axis=0) != 0
train_all_genes = train.copy()
train = train[:,is_expressed]
train.layers['norm_counts'] = train.X.copy()

predictions_metadata = pd.read_csv("from_to_docker/predictions_metadata.csv")
for c in ['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation", 'prediction_timescale']:
    assert c in predictions_metadata.columns, f"For the ggrn prescient backend, predictions_metadata.columns is required to contain {c}."

with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)

with open(os.path.join("from_to_docker/ggrn_args.json")) as f:
    ggrn_args = json.load(f)

try:
    network = LightNetwork(files = ["from_to_docker/network.parquet"])
except FileNotFoundError as e:
    raise ValueError(
"""
Could not read in the network (from_to_docker/network.parquet). GGRN cannot use the scKINETICS backend without a user-provided base network. 
Base networks are usually derived from motif analysis of cell-type-specific ATAC data. Many such networks are available in out collection. 
You have two options: 
   - If you are using GGRN directly, construct a LightNetwork object using pereggrn_networks.LightNetwork() and pass it to the ggrn.api.GRN constructor.
   - If you are using GGRN indirectly via our benchmarking framework, add a network to your experiment's metadata.json.
More information should be available in the documentation for GGRN and for the benchmarking framework.
"""
    )

# Network reformatting
# This is based on the network contruction code provided with the original paper in tf_targets.py.
# Their description of the network format:
# > each target's TFs can be accessed as G[target_name].transcription_factors
# > indices in the count matrix are stored in G[target_name].target_index, and .transcription_factors_index
# > the corresponding imputed, scaled count matrix is added to peak_annotation.adata.obsm as "X_cluster"
# > cluster_peaks_filtered = np.intersect1d(peak_annotation.pairs.index,cluster_peaks[cluster])
network_reformatted = {}
train.var["numeric_index"] = range(train.n_vars)
for _,record in network.get_all().iterrows():
    if record["target"] not in train.var.index:
        continue
    if record["regulator"] not in train.var.index:
        continue
    target = tf_targets.TargetRecord(   record['target'], index=train.var.loc[record["target"],    "numeric_index"])
    target.add_transcription_factor( record["regulator"], index=train.var.loc[record["regulator"], "numeric_index"])
    try: #if this target was already created
        network_reformatted[record['target']].update(target)
    except: #if first instance of target
        network_reformatted[record['target']] = target

# Expression reformatting
# This is based on the function prepare_data provided with the original paper in tf_targets.py.
sc.pp.pca(train, n_comps=50)
temp_data = sc.external.pp.magic(train, n_pca=30, random_state=0, copy = True)
try: 
    temp_data.X = temp_data.X.toarray()
except AttributeError:
    pass
train.obsm['X_transformed'] = pd.DataFrame(
    temp_data.X,
    columns=train.var_names,
    index=train.obs_names
)
for ct in train.obs["cell_type"].unique():
    try: 
        train.uns['X_{}'.format(ct)] = pd.DataFrame(train.X.toarray(), index = train.obs_names, columns=train.var_names)
    except AttributeError:
        train.uns['X_{}'.format(ct)] = pd.DataFrame(train.X, index = train.obs_names, columns=train.var_names)

# Train it up!
model = EM.ExpectationMaximization(threads=15, maxiter=20)
model.fit(
    train,
    {ct:network_reformatted.copy() for ct in train.obs["cell_type"].unique()},
    celltype_basis = "cell_type"
)

# Predictions
# This is based on the in silico KO screening code provided with the original paper. 
# https://github.com/dpeerlab/scKINETICS/blob/40588aca0fea25a1c4c397e3579f355556dc7fd6/sckinetics/tf_perturbation.py#L13
# The modifications from the original are:
#     - we perturb only user-specified TF's, not all TF's.
#     - we allow user-specified dosage of TF's, e.g. for overexpression or (incomplete) knockdown experiments.
#     - we return expression on the same scale as the training data, not cosine similarity of perturbed and unperturbed velocity estimates.
#           Cosine similarity or any other metric can be computed after the fact.
#     - we predict expression starting from a specific time-point and after a specific number of steps, not for the whole dataset.
def predict_one(tf: str, expression_level_after_perturbation: float, cell_type: str, timepoint: int, num_steps: int):
    """Predict expression after TF perturbation

    Args:
        tf (str): index of the TF such that train.var.index[tf_index] is the name of the TF
        expression_level_after_perturbation (_type_): expression level of the perturbed TF after perturbation
        cell_type (str): starting cell type
        timepoint (int): starting timepoint
        num_steps (int): how many time-steps ahead to predict
    """
    A=model.A_[cell_type].astype(np.float64)
    A=np.nan_to_num(A)
    X = train_all_genes[(train_all_genes.obs["cell_type"]==cell_type) & (train_all_genes.obs["timepoint"]==timepoint), :].X.mean(0).copy()
    try:
        tf_index = train_all_genes.var.index.get_loc(tf)
    except KeyError:
        raise ValueError(f"TF {tf} not found in the training data.")
    for _ in range(num_steps):
        X[tf_index] = expression_level_after_perturbation
        V = A.dot(X[is_expressed])
        X[is_expressed] = X[is_expressed]+V.flatten()
        X[tf_index] = expression_level_after_perturbation
    assert X.shape[0]==train_all_genes.n_vars
    return X

print("Running simulations")
predictions = anndata.AnnData(
    X = np.zeros((predictions_metadata.shape[0], train_all_genes.n_vars)),
    obs = predictions_metadata,
    var = train_all_genes.var,
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
    try:
        train_index = (train.obs["cell_type"]==current_prediction_metadata["cell_type"]) & (train.obs["timepoint"]==current_prediction_metadata["timepoint"])
        goi, level, prediction_timescale = current_prediction_metadata[['perturbation', "expression_level_after_perturbation", 'prediction_timescale']]
        for i in np.where(prediction_index)[0]:
            predictions[i, :].X = predict_one(goi, level, current_prediction_metadata["cell_type"], current_prediction_metadata["timepoint"], prediction_timescale).flatten()
    except ValueError as e:
        predictions[prediction_index, :].X = np.nan
        print("Prediction failed. Error text: " + str(e))

    
print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")