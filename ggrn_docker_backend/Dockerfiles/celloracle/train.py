import os
import json
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
import celloracle as co
import warnings
import traceback
from pereggrn_networks import LightNetwork, pivotNetworkLongToWide

train = sc.read_h5ad("from_to_docker/train.h5ad")
try:
    network = LightNetwork(files = ["from_to_docker/network.parquet"])
except FileNotFoundError as e:
    raise ValueError("Could not read in the network (from_to_docker/network.parquet). GGRN cannot use the celloracle backend without a user-provided base network. CellOracle base networks are usually derived from motif analysis of cell-type-specific ATAC data. Many such networks are available in out collection. Construct a LightNetwork object using pereggrn_networks.LightNetwork() and pass it to the ggrn.api.GRN constructor.")

network = pivotNetworkLongToWide(network.get_all())
predictions_metadata = pd.read_csv("from_to_docker/predictions_metadata.csv", index_col=0)
expected_columns = np.array(['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation", 'prediction_timescale'])
for c in expected_columns:
    assert c in predictions_metadata.columns, f"For the ggrn celloracle backend, predictions_metadata.columns is required to contain {c}."
predictions_metadata["error_message"] = ""
with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)
with open(os.path.join("from_to_docker/ggrn_args.json")) as f:
    ggrn_args = json.load(f)
if "pruning_parameter" not in ggrn_args or ggrn_args["pruning_parameter"] is None or np.isnan(ggrn_args["pruning_parameter"]):
    ggrn_args["pruning_parameter"] = 2000

print("Importing data and preprocessing")
sc.tl.pca(train, n_comps=50)
oracle = co.Oracle()
oracle.import_anndata_as_raw_count(
    adata=train,
    cluster_column_name="cell_type",
    embedding_name="X_pca"
)
oracle.import_TF_data(TF_info_matrix=network)
oracle.perform_PCA()
n_comps = 50
k = 1
print("Running imputation")
oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                        b_maxl=k*4, n_jobs=4)

print("Running ridge regression and pruning")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("regressing")
    links = oracle.get_links(cluster_name_for_GRN_unit="cell_type", 
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
    X = np.zeros((predictions_metadata.shape[0], train.n_vars)),
    obs = predictions_metadata,
    var = train.var,
)

for _, gene_level_steps in predictions.obs[['perturbation', "expression_level_after_perturbation", 'prediction_timescale']].drop_duplicates().iterrows(): 
    try:
        oracle.simulate_shift(
            # For multi-gene perturbations, the format provided by ggrn is like "OCT4,NANOG" and "2.11135,5.09493".
            perturb_condition={
                g:x 
                for g,x in zip(
                                           gene_level_steps["perturbation"]                        .split(","), 
                    [float(z) for z in str(gene_level_steps["expression_level_after_perturbation"]).split(",")]
                )
            }, 
            n_propagation=gene_level_steps["prediction_timescale"], 
            ignore_warning = True
        )
        for _, starting_state in predictions.obs[['timepoint', 'cell_type']].drop_duplicates().iterrows(): 
            prediction_index = \
                (predictions.obs["cell_type"]==starting_state["cell_type"]) & \
                (predictions.obs["timepoint"]==starting_state["timepoint"]) & \
                (predictions.obs["perturbation"]==gene_level_steps["perturbation"]) & \
                (predictions.obs["expression_level_after_perturbation"]==gene_level_steps["expression_level_after_perturbation"]) & \
                (predictions.obs["prediction_timescale"]==gene_level_steps["prediction_timescale"]) 
            train_index = (oracle.adata.obs["cell_type"]==starting_state["cell_type"]) & (oracle.adata.obs["timepoint"]==starting_state["timepoint"])
            for i in np.where(prediction_index)[0]:
                predictions[i, :].X = oracle.adata[train_index,:].layers['simulated_count'].squeeze().mean(0)
    except ValueError as e:
        for _, starting_state in predictions.obs[['timepoint', 'cell_type']].drop_duplicates().iterrows(): 
            prediction_index = \
                (predictions.obs["cell_type"]==starting_state["cell_type"] ) & \
                (predictions.obs["timepoint"]==starting_state["timepoint"] ) & \
                (predictions.obs["perturbation"]==gene_level_steps["perturbation"] ) & \
                (predictions.obs["expression_level_after_perturbation"]==gene_level_steps["expression_level_after_perturbation"] ) & \
                (predictions.obs["prediction_timescale"]==gene_level_steps["prediction_timescale"] )
            predictions[prediction_index, :].X = np.nan
            predictions.obs.loc[prediction_index, "error_message"] = repr(e)
        print("Prediction failed for " + gene_level_steps.to_csv() + " with error " + repr(e))

print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")