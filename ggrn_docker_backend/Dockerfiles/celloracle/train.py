import os
import json
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
import celloracle as co
import warnings
from load_networks import LightNetwork, pivotNetworkLongToWide

train = sc.read_h5ad("from_to_docker/train.h5ad")
# network = co.data.load_human_promoter_base_GRN() 
network = LightNetwork(files = ["from_to_docker/network.parquet"])
network = pivotNetworkLongToWide(network.get_all())
with open(os.path.join("from_to_docker/perturbations.json")) as f:
    perturbations = json.load(f)
with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)
with open(os.path.join("from_to_docker/ggrn_args.json")) as f:
    ggrn_args = json.load(f)

print("Importing data and preprocessing")
oracle = co.Oracle()
sc.neighbors.neighbors(train)
sc.tl.louvain(train)
oracle.import_anndata_as_raw_count(
    adata=train,
    cluster_column_name="louvain",
    embedding_name="X_pca"
)
oracle.import_TF_data(TF_info_matrix=network)
oracle.perform_PCA()
n_comps = 50
k = 1
oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                        b_maxl=k*4, n_jobs=4)

print("Running ridge regression and pruning")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("regressing")
    links = oracle.get_links(cluster_name_for_GRN_unit="louvain", 
                                alpha=10, 
                                model_method = "bayesian_ridge",
                                verbose_level=10,    
                                test_mode=False, 
                                n_jobs=14)
    print("filtering")
    links.filter_links(p=0.001, 
                       weight="coef_abs", 
                       threshold_number=int(ggrn_args["pruning_parameter"]))
print("filteringing")
links.links_dict = links.filtered_links.copy()
print("refitting")
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)

print("Running simulations")
predictions = anndata.AnnData(
    X = np.zeros((len(perturbations), train.n_vars)),
    obs = pd.DataFrame({
        "perturbation":                       [p[0] for p in perturbations], 
        "expression_level_after_perturbation":[p[1] for p in perturbations], 
    }), 
    var = train.var,
)
for i, goilevel in enumerate(perturbations):
    goi, level = goilevel
    print("Predicting " + goi)
    try:
        oracle.simulate_shift(perturb_condition={goi: level}, n_propagation=3, ignore_warning = True)
        predictions[i, :].X = oracle.adata[oracle.adata.obs["is_control"],:].layers['simulated_count'].squeeze().mean(0)
    except ValueError as e:
        predictions[i, :].X = np.nan
        print("Prediction failed for " + goi + " with error " + str(e))

print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")