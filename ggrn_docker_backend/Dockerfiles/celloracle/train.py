import os
import json
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
import celloracle as co
import warnings
from load_networks import LightNetwork, pivotNetworkLongToWide
import itertools

train = sc.read_h5ad("from_to_docker/train.h5ad")
# network = co.data.load_human_promoter_base_GRN() 
network = LightNetwork(files = ["from_to_docker/network.parquet"])
network = pivotNetworkLongToWide(network.get_all())
predictions_metadata = pd.read_csv("predictions_metadata.csv")
assert all(predictions_metadata.columns == np.array(['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation", 'prediction_timescale']))
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
print("filtering")
links.links_dict = links.filtered_links.copy()
print("refitting")
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)

print("Running simulations")
predictions = anndata.AnnData(
    X = np.zeros((len(combinations), train.n_vars)),
    obs = predictions_metadata,
    var = train.var,
)
for gene_level_steps in predictions.obs[['perturbation', "expression_level_after_perturbation", 'prediction_timescale']].unique():
    print("Predicting " + gene_level_steps["perturbation"])
    try:
        oracle.simulate_shift(
            perturb_condition={
                gene_level_steps["perturbation"]: gene_level_steps["expression_level_after_perturbation"]
            }, 
            n_propagation=gene_level_steps["prediction_timescale"], 
            ignore_warning = True
        )
        for starting_state in predictions.obs[['timepoint', 'cell_type']].unique():
            prediction_index = \
                predictions.obs["cell_type"]==starting_state["cell_type"] & \
                predictions.obs["timepoint"]==starting_state["timepoint"] & \
                predictions.obs["perturbation"]==gene_level_steps["perturbation"] & \
                predictions.obs["expression_level_after_perturbation"]==gene_level_steps["expression_level_after_perturbation"] & \
                predictions.obs["prediction_timescale"]==gene_level_steps["prediction_timescale"] 
            train_index = oracle.adata.obs["cell_type"]==starting_state["cell_type"] & oracle.adata.obs["timepoint"]==starting_state["timepoint"]
            predictions[prediction_index, :].X = oracle.adata[train_index,:].layers['simulated_count'].squeeze().mean(0)
    except ValueError as e:
        for starting_state in predictions.obs[['timepoint', 'cell_type']].unique():
            prediction_index = \
                predictions.obs["cell_type"]==starting_state["cell_type"] & \
                predictions.obs["timepoint"]==starting_state["timepoint"] & \
                predictions.obs["perturbation"]==gene_level_steps["perturbation"] & \
                predictions.obs["expression_level_after_perturbation"]==gene_level_steps["expression_level_after_perturbation"] & \
                predictions.obs["prediction_timescale"]==gene_level_steps["prediction_timescale"] 
            predictions[prediction_index, :].X = np.nan
        print("Prediction failed for " + gene_level_steps.to_csv() + " with error " + str(e))

print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")