import anndata
import json
import os
import numpy as np
import pandas as pd
import scanpy as sc
from pereggrn_networks import LightNetwork, pivotNetworkLongToWide
import dictys.network
print("Training a model via dictys.", flush=True)
train = sc.read_h5ad("from_to_docker/train.h5ad")
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
Could not read in the network (from_to_docker/network.parquet). GGRN cannot use the dictys backend without a user-provided base network. 
Base networks are usually derived from footprint analysis of cell-type-specific ATAC data. Many similar networks are available in out collection. 
You have two options: 
   - If you are using GGRN directly, construct a LightNetwork object using pereggrn_networks.LightNetwork() and pass it to the ggrn.api.GRN constructor.
   - If you are using GGRN indirectly via PEREGGRN, add a network to your experiment's metadata.json.
More information should be available in the documentation for GGRN and for PEREGGRN.
"""
)

# We apply dictys based on the full tutorial at https://github.com/pinellolab/dictys/blob/master/doc/tutorials/full-multiome/notebooks/3-static-inference.ipynb
# 
# The code to read expression and network structure looks like this. 
# dt0=pd.read_csv(fi_exp,header=0,index_col=0,sep='\t')
# mask=pd.read_csv(fi_mask,header=0,index_col=0,sep='\t')
# assert len(set(mask.index)-set(dt0.index))==0
# assert len(set(mask.columns)-set(dt0.index))==0
#

print("Converting inputs to dictys' preferred format.", flush=True)
# Intersect the network with the expression data
network_features = set(network.get_all_regulators()).union(set(network.get_all_one_field("target")))
common_features = list(train.var.index.intersection(network_features))
network = network.get_all().query("regulator in @common_features and target in @common_features")
train = train[:, common_features]
# Convert to dictys' preferred format via celloracle's preferred format
network = pivotNetworkLongToWide(network) 
network.index = network["gene_short_name"]
del network["gene_short_name"]
del network["peak_id"]
# Input network must be square. Make it square by zero-padding (specifying no targets for any non-TF).
network_non_regulators = list(set(train.var.index) - set(network.columns))
network.loc[:,network_non_regulators] = 0 # This causes a rank-deficiency error in cholesky decomposition. Replacing with random values (probably linearly independent??) solves it.
# Remove self loops. Without this, we trigger an assertion error in dictys reconstruct line 790, which is documented behavior.
for g in network.columns:
    network.loc[g,g]=0

# exporttttt
try:
    pd.DataFrame(train.raw[:, common_features].X.toarray(), index = train.raw.obs_names, columns = common_features).T.to_csv("exp.tsv", sep="\t")
except AttributeError:
    pd.DataFrame(train.raw[:, common_features].X, index = train.obs_names, columns = common_features).T.to_csv("exp.tsv", sep="\t")

# Dictys expects one TF per row and CO uses one per column. We transpose the network to match the expected format.
network.T.to_csv("mask.tsv", sep="\t")

print("Running network reconstruction.", flush=True)
dictys.network.reconstruct(
    fi_exp="exp.tsv",
    fi_mask="mask.tsv",
    fo_weight="weight.tsv",
    fo_meanvar="meanvar.tsv",
    fo_covfactor="covfactor.tsv",
    fo_loss="loss.tsv",
    fo_stats="stats.tsv", 
    npc = 0
)
#   fi_exp                Path of input tsv file of expression matrix.
#   fi_mask               Path of input tsv file of mask matrix indicating which
#                         edges are allowed. Can be output of dictys chromatin
#                         binlinking.
#   fo_weight             Path of output tsv file of edge weight matrix
#   fo_meanvar            Path of output tsv file of mean and variance of each
#                         gene's relative log expression
#   fo_covfactor          Path of output tsv file of factors for the off-
#                         diagonal component of gene covariance matrix
#   fo_loss               Path of output tsv file of sublosses in each training
#                         step
#   fo_stats              Path of output tsv file of basic stats of each
#                         variable during training

# print("Rescaling coefficients.")
# dictys.network.normalize(
#     fi_weight="weight.tsv",
#     fi_meanvar="meanvar.tsv",
#     fi_covfactor="covfactor.tsv",
#     fo_nweight="nweight.tsv"
# )
# #   fi_weight     Path of input tsv file of edge weight matrix
# #   fi_meanvar    Path of iput tsv file of mean and variance of each gene's
# #                 relative log expression
# #   fi_covfactor  Path of iput tsv file of factors for the off-diagonal
# #                 component of gene covariance matrix
# #   fo_nweight    Path of output tsv file of normalized edge weight matrix

print("Calculating indirect effects.")
dictys.network.indirect(
    fi_weight="weight.tsv",
    fi_covfactor="covfactor.tsv",
    fi_meanvar="meanvar.tsv",
    fo_iweight="iweight.tsv",
    norm = 1, # This puts the result on the scale of change in target per change in regulator. https://github.com/pinellolab/dictys/issues/61
)
#   fi_weight             Path of input tsv file of edge weight matrix
#   fi_covfactor          Path of iput tsv file of factors for the off-diagonal
#                         component of gene covariance matrix
#   fi_meanvar            Path of iput tsv file of mean and variance of each gene's
#                         relative log expression
#   fo_iweight            Path of output tsv file of steady-state indirect
#                         effect edge weight matrix

indirect_effects = pd.read_csv("iweight.tsv", sep="\t", index_col=0)
def predict_one(perturbed_gene: str, level: float, cell_type: str, timepoint: int):
    """Predict log-scale post-perturbation expression

    Args:
        perturbed_gene (str): gene of interest, the perturbation
        level (float): the expression level of perturbed_gene after the perturbation
        cell_type (str): starting cell type to perturb
        timepoint (int): starting timepoint to perturb

    Returns:
        np.array: the predicted log-scale expression of all genes
    """
    train_index = (train.obs["cell_type"]==current_prediction_metadata["cell_type"]) & (train.obs["timepoint"]==current_prediction_metadata["timepoint"])
    control_expression = train[train_index, :].X.mean()
    change_in_perturbed_gene = level - control_expression[perturbed_gene]
    return indirect_effects[perturbed_gene, :]*change_in_perturbed_gene  # Index name is "regulator" so we want to grab a row, not a column

predictions_metadata["prediction_timescale"]

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
    try:
        perturbed_gene, level, prediction_timescale = current_prediction_metadata[['perturbation', "expression_level_after_perturbation", 'prediction_timescale']]
        for i in np.where(prediction_index)[0]:
            predictions[i, :].X = predict_one(perturbed_gene, level, current_prediction_metadata["cell_type"], current_prediction_metadata["timepoint"], prediction_timescale)
    except ValueError as e:
        predictions[prediction_index, :].X = np.nan
        print("Prediction failed. Error text: " + str(e))

    
print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")