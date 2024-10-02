import scanpy as sc
import os 
from pereggrn.experimenter import averageWithinPerturbation
import json 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from anndata import AnnData
import numpy as np

# Read the inputs from PEREGGRN: data, desired predictions, and arguments.
out_dir = "from_to_docker"
sce = sc.read_h5ad("from_to_docker/train.h5ad")

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

# We will get W from eqn. 1 of Ahlmann-Eltze et al. using this code.
def solve_y_axb_for_x(Y, A, B, A_ridge = kwargs["ridge_penalty"], B_ridge = kwargs["ridge_penalty"]):
    assert A.shape[1] == Y.shape[0]
    assert B.shape[1] == Y.shape[1]
    AtAiA = np.linalg.solve(A.dot(A.T) + np.identity(A.shape[0]) * A_ridge, A)
    BtBiB = np.linalg.solve(B.dot(B.T) + np.identity(B.shape[0]) * B_ridge, B)
    return AtAiA.dot(Y).dot(BtBiB.T)

# Compute mean expression before pseudo-bulking
baseline = sce[sce.obs["is_control"], :].X.mean(axis = 0)
# We can only train on or predict perturbations for genes that are measured, because we need the expression-based embedding. 
print(f"Train data number of (samples, perturbations): {sce.n_obs}, {sce.obs['perturbation'].nunique()}")
print("Removing perturbations of genes that are not measured.")
keepers = np.array([
    all([
        g in sce.uns["perturbed_and_measured_genes"] 
        for g in p.split(",")
    ]) 
    for p in sce.obs["perturbation"]
])
pd.DataFrame(sce.var_names).to_csv("from_to_docker/genes.csv", index = False)
print(pd.value_counts(keepers))
sce.obs.loc[ keepers, "perturbation"].value_counts().to_csv("from_to_docker/kept_perturbations.csv")
sce.obs.loc[~keepers, "perturbation"].value_counts().to_csv("from_to_docker/removed_perturbations.csv")
sce = sce[keepers, :]
print(f"Final train data number of (samples, perturbations): {sce.n_obs}, {sce.obs['perturbation'].nunique()}")

print(f"Test data number of (samples, perturbations): {predictions_metadata.shape[0]}, {predictions_metadata['perturbation'].nunique()}")
print("From the requested predictions, removing perturbations of genes that are not measured.")
keepers_test = np.array([
    all([
        g in sce.var_names
        for g in p.split(",")
    ]) 
    for p in predictions_metadata["perturbation"]
])
print(pd.value_counts(keepers_test))
predictions_metadata = predictions_metadata.loc[keepers_test | predictions_metadata["is_control"], :]
print(f"Final test data number of (samples, perturbations): {predictions_metadata.shape[0]}, {predictions_metadata['perturbation'].nunique()}")

# Pseudobulk, center, do PCA
psce = averageWithinPerturbation(sce)
centered_data = psce.X - baseline # this works as intended: https://numpy.org/doc/stable/user/basics.broadcasting.html
gene_pca = PCA(n_components = kwargs["pca_dim"]).fit(centered_data)
gene_emb = gene_pca.components_

# Features used: the PCA embeddings of the perturbed genes
genezzzzz = list(psce.var_names)
def get_embedding(p, is_control):
    """Get embeddings, handling edge cases.

    Args:
        p (_type_): perturbation in the format "SOX9" or "SOX9,SOX6"
        is_control (bool): whether the perturbation is a control

    Returns:
        np.array: for multi-gene perturbations, take the mean of the gene embeddings. For single-gene perturbations, the gene embedding. For controls, a zero vector.

    """
    if is_control:
        return np.zeros(kwargs["pca_dim"])
    return gene_emb.copy()[:,[genezzzzz.index(g) for g in p.split(",")]].mean(1)

pert_emb_train = np.vstack(
    [
        get_embedding(p, i)
        for p,i in psce.obs[["perturbation", "is_control"]].itertuples(index=False, name=None)
    ]
).T
pert_emb_test   = np.vstack(
    [
        get_embedding(p, i) 
        for p,i in predictions_metadata[["perturbation", "is_control"]].itertuples(index=False, name=None)
    ]
).T

# Here we have one sample per row, one feature per column, as usual in ML and in scanpy.
# Original paper Eqn 1 oriented it the other way: genes by samples. 
# So we are solving for W in X = PWG, not X = GWP.
W = solve_y_axb_for_x(Y = centered_data, 
                      A = pert_emb_train, 
                      B = gene_emb)

pred = AnnData( 
    X = (pert_emb_test.T.dot(W).dot(gene_emb)) + baseline, 
    obs = predictions_metadata, 
    var = psce.var, 
)
pred.write_h5ad(os.path.join(out_dir, "predictions.h5ad"))