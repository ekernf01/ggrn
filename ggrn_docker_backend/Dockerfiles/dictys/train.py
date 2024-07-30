import anndata
import json
import os
import numpy as np
import pandas as pd
import scanpy as sc
from pereggrn_networks import LightNetwork, pivotNetworkLongToWide
import dictys.network
import contextlib
import os

# Thanks to https://stackoverflow.com/a/75049113
@contextlib.contextmanager
def temporaryWorkingDirectory(path):
    _oldCWD = os.getcwd()
    os.chdir(os.path.abspath(path))
    try:
        yield
    finally:
        os.chdir(_oldCWD)

def make_unique(original_list):
    unique_list = []
    for item in original_list:
        counter = 0
        new_item = item
        while new_item in unique_list:
            counter += 1
            new_item = "{}_{}".format(item, counter)
        unique_list.append(new_item)
    return unique_list

print("Training a model via dictys.", flush=True)
os.chdir("from_to_docker")
train = sc.read_h5ad("train.h5ad")
predictions_metadata = pd.read_csv("predictions_metadata.csv")
for c in ['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation", 'prediction_timescale']:
    assert c in predictions_metadata.columns, f"For the ggrn prescient backend, predictions_metadata.columns is required to contain {c}."

with open(os.path.join("kwargs.json")) as f:
    kwargs = json.load(f)

with open(os.path.join("ggrn_args.json")) as f:
    ggrn_args = json.load(f)

try:
    network = LightNetwork(files = ["network.parquet"])
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

defaults = {
    'lr': 0.01,
    'lrd': 0.999,
    'nstep': 4000,
    'npc': 0,
    'fi_cov': None,
    'model': 'ou',
    'nstep_report': 100,
    'rseed': 12345,
    'device': 'cpu',
    'dtype': 'float',
    'loss': 'Trace_ELBO_site',
    'nth': 15,
    'varmean': 'N_0val',
    'varstd': None,
    'fo_weightz': None,
    'scale_lyapunov': 1E5,
    'minimum_expression': 0.125
}
for k in defaults:
    if k not in kwargs:
        kwargs[k] = defaults[k]

# Dictys requires a minimum expression cutoff.
print("Filtering genes for minimum expression.")
ise = train.X.sum(axis = 0) >= kwargs['minimum_expression']*train.X.shape[0]
train.var["is_sufficiently_expressed"] = ise.T
for g in predictions_metadata["perturbation"]:
    if g in train.var_names and not train.var.loc[g, "is_sufficiently_expressed"]:
        print(f"Gene {g} falls below minimum_expression ({kwargs['minimum_expression']}). Perturbations of it will not be predicted.")
        predictions_metadata = predictions_metadata.query("perturbation!=@g")
train_all_features = train.copy()
train = train[:, train.var["is_sufficiently_expressed"]]

print("Converting inputs to dictys' preferred format.", flush=True)
# Intersect the network with the expression data
# If there is a regulator in the network that does not appear as a target in the network (has no incoming edges),
# This causes a pattern of zeros that dictys.network.reconstruct cannot handle.
# So we remove any regulators that are not also targets, focusing on targets that are also in the expression data.
network_features = set(network.get_all_one_field("target")).union(set(network.get_all_one_field("regulator")))
common_features = list(train.var.index.intersection(network_features))
common_features_int = np.array([train.var.index.get_loc(f) for f in common_features])
network = network.get_all().query("regulator in @common_features and target in @common_features")
train = train[:, common_features]
# Convert to dictys' preferred format via celloracle's preferred format
network = pivotNetworkLongToWide(network) 
network.index = network["gene_short_name"]
del network["gene_short_name"]
del network["peak_id"]
# Input network must be square. Make it square by zero-padding:
#   - specify no targets for any non-TF.
#   - If a gene is listed as a TF but not a target, add it to the targets.
network_non_regulators = list(set(train.var.index) - set(network.columns))
network.loc[:,network_non_regulators] = 0
network_non_targets = list(set(train.var.index) - set(network.index))
network = pd.concat([network, pd.DataFrame({tf:0 for tf in network.columns}, index = network_non_targets)])
network = network.loc[common_features, common_features]
# Remove self loops. Without this, we trigger an assertion error in dictys reconstruct line 790, which is documented behavior.
for g in network.columns:
    network.loc[g,g]=0

if any(pd.isnull(network).any()) or \
    any(common_features!=network.columns.astype("str")) or \
        any(common_features!=network.index.values.astype("str") ):
    print(set(common_features).difference(network.columns))
    print(set(common_features).difference(network.index))
    print(set(network.index).difference(common_features))
    print(set(network.columns).difference(common_features))
    raise ValueError("There are NaNs or incorrect genes or missing genes in the input network. This will cause dictys to fail.")

def augment_data(X, desired_total):
    """Add resampled dummy cells to the data to make it square

    Args:
        X (pd.DataFrame): expression data

    Returns:
        pd.DataFrame: expression data with resampled dummy cells added
    """
    num_new_cells = desired_total - X.shape[0]
    average_expression = X.sum(axis = 0)
    newX = pd.DataFrame(index = range(num_new_cells), columns = X.columns)
    for i in range(num_new_cells):
        newX.loc[i, :] = [np.random.binomial(np.random.poisson(x), 1/X.shape[0]) for x in average_expression]
    return pd.concat([X, newX])

indirect_effects = dict()
for ct in train.obs["cell_type"].unique():
    os.makedirs(ct, exist_ok=True)
    with temporaryWorkingDirectory(ct):
        # exporttttt
        cells_to_use = train.obs.query("cell_type==@ct").index        
        try:
            X = pd.DataFrame(train.raw[cells_to_use, common_features].X.toarray(), index = make_unique(cells_to_use), columns = common_features)
        except AttributeError:
            X = pd.DataFrame(train.raw[cells_to_use, common_features].X,           index = make_unique(cells_to_use), columns = common_features)
        # Dictys expects one TF per row and CO uses one per column. We transpose the network to match the expected format.
        network.T.to_csv("mask.tsv", sep="\t")
        # dictys.network.reconstruct errs unless n>p, so we add dummy cells in that case. 
        if X.shape[0] < network.shape[0]: 
            print("Augmenting data since we have fewer cells than genes.")
            X = augment_data(X, network.shape[0])
        # Dictys expects one gene per row, one cell per column, unlike typical ML or stats data tables. So, transpose before saving.
        X.T.to_csv("exp.tsv", sep="\t")
        print("Running network reconstruction.", flush=True)
        dictys.network.reconstruct(
            fi_exp="exp.tsv",
            fi_mask="mask.tsv",
            fo_weight="weight.tsv",
            fo_meanvar="meanvar.tsv",
            fo_covfactor="covfactor.tsv",
            fo_loss="loss.tsv",
            fo_stats="stats.tsv", 
            lr = kwargs['lr'],
            lrd = kwargs['lrd'],
            nstep = kwargs['nstep'],
            npc = kwargs['npc'],
            fi_cov = kwargs['fi_cov'],
            model = kwargs['model'],
            nstep_report = kwargs['nstep_report'],
            rseed = kwargs['rseed'],
            device = kwargs['device'],
            dtype = kwargs['dtype'],
            loss = kwargs['loss'],
            nth = kwargs['nth'],
            varmean = kwargs['varmean'],
            varstd = kwargs['varstd'],
            fo_weightz = kwargs['fo_weightz'],
            scale_lyapunov = kwargs['scale_lyapunov']
        )
        print("Calculating indirect effects.")
        dictys.network.indirect(
            fi_weight="weight.tsv",
            fi_covfactor="covfactor.tsv",
            fi_meanvar="meanvar.tsv",
            fo_iweight="iweight.tsv",
            norm = 1, # This puts the result on the scale of change in target per change in regulator. https://github.com/pinellolab/dictys/issues/61
        )
        indirect_effects[ct] = pd.read_csv("iweight.tsv", sep="\t", index_col=0)

def predict_expression(perturbed_gene: str, level: float, cell_type: str, timepoint: int):
    """Predict log-scale post-perturbation expression

    Args:
        perturbed_gene (str): gene of interest, the perturbation
        level (float): the expression level of perturbed_gene after the perturbation
        cell_type (str): starting cell type to perturb
        timepoint (int): starting timepoint to perturb

    Returns:
        np.array: the predicted log-scale expression of all genes. For genes not in the dictys model, returns control expression.
    """
    is_baseline = (
        (train.obs["cell_type"]==current_prediction_metadata["cell_type"]) &
        (train.obs["timepoint"]==current_prediction_metadata["timepoint"]) &
        train.obs["is_control"]
    )
    if not any(is_baseline):
        raise ValueError(f"No controls were found at time {current_prediction_metadata['timepoint']} for celltype {current_prediction_metadata['cell_type']}. Cannot predict perturbed expression.")
    control_expression = train_all_features[is_baseline, :].X.mean(axis = 0)
    try: 
        control_expression = control_expression.A1
    except AttributeError:
        pass
    if perturbed_gene in train.var.index:
        change_in_perturbed_gene = level - train[is_baseline, perturbed_gene].X.mean()
        logfc = indirect_effects[cell_type].loc[perturbed_gene, :]*change_in_perturbed_gene  # Index name is "regulator" so we want to grab a row, not a column
        logfc = logfc.values
    else:
        logfc = 0
    control_expression[common_features_int] = control_expression[common_features_int] + logfc
    return control_expression



print("Running simulations")
predictions = anndata.AnnData(
    X = np.zeros((predictions_metadata.shape[0], train_all_features.n_vars)),
    obs = predictions_metadata,
    var = train_all_features.var,
)
for _, current_prediction_metadata in predictions.obs[['perturbation', "expression_level_after_perturbation", 'prediction_timescale', 'timepoint', 'cell_type']].drop_duplicates().iterrows():
    prediction_index = \
        (predictions.obs["cell_type"]==current_prediction_metadata["cell_type"]) & \
        (predictions.obs["timepoint"]==current_prediction_metadata["timepoint"]) & \
        (predictions.obs["perturbation"]==current_prediction_metadata["perturbation"]) & \
        (predictions.obs["expression_level_after_perturbation"]==current_prediction_metadata["expression_level_after_perturbation"]) & \
        (predictions.obs["prediction_timescale"]==current_prediction_metadata["prediction_timescale"]) 
    x = predict_expression(
        perturbed_gene=current_prediction_metadata["perturbation"], 
        level = current_prediction_metadata["expression_level_after_perturbation"], 
        cell_type = current_prediction_metadata["cell_type"],
        timepoint = current_prediction_metadata["timepoint"]
    )
    for i in np.where(prediction_index)[0]:
        predictions[i, :].X = x
    

print("Saving results.")
predictions.write_h5ad("predictions.h5ad")