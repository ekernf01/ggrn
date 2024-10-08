import numpy as np 
import pandas as pd 
import onesc
import scanpy as sc
import anndata
import json 
import os

predictions_metadata = pd.read_csv("from_to_docker/predictions_metadata.csv")
for c in ['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation", 'prediction_timescale']:
    assert c in predictions_metadata.columns, f"For the ggrn onesc backend, predictions_metadata.columns is required to contain {c}."

adata = sc.read_h5ad("from_to_docker/train.h5ad")

with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)

with open(os.path.join("from_to_docker/ggrn_args.json")) as f:
    ggrn_args = json.load(f)

defaults = {
    "num_cells_to_simulate": 100
}
for c in defaults.keys():
    if c not in kwargs.keys():
        kwargs[c] = defaults[c]

print("Constructing cell type graph.")
adata.obs["type_and_time"] = [f"{a}-{b}" for a, b in zip(adata.obs["cell_type"], adata.obs["timepoint"])]
adata.obs["type_and_time"] = [tnt.replace("_", "-") for tnt in adata.obs["type_and_time"]]
predictions_metadata["type_and_time"] = [f"{a}-{b}" for a, b in zip(predictions_metadata["cell_type"], predictions_metadata["timepoint"])]
predictions_metadata["type_and_time"] = [tnt.replace("_", "-") for tnt in predictions_metadata["type_and_time"]]
first_timepoint  = adata.obs["timepoint"].min()
initial_clusters = list(adata.obs.loc[adata.obs["timepoint"]==first_timepoint, "type_and_time"].unique())
last_timepoint   = adata.obs["timepoint"].max()
end_clusters     = list(adata.obs.loc[adata.obs["timepoint"]==last_timepoint, "type_and_time"].unique())
cellstate_graph = onesc.construct_cluster_graph_adata(
    adata, 
    initial_clusters = initial_clusters, 
    terminal_clusters = end_clusters,
    cluster_col = "type_and_time", 
    pseudo_col = "timepoint"
)

# Another way to construct the state graph is to manually define the edge list.
# We could do that by generating a coarse summary of the matching done upstream by GGRN. 
# We won't now, but here's the relevant code snippet.
# edge_list = [("CMP", "MK"), ("CMP", "MEP"), ("MEP", "Erythrocytes"), ("CMP", "GMP"), ("GMP", "Granulocytes"), ("GMP", "Monocytes")]
# cellstate_graph = nx.DiGraph(edge_list)

print("Inferring network.")
start_end_states = {'start': initial_clusters, 'end': end_clusters}
iGRN = onesc.infer_grn(
    cellstate_graph, 
    start_end_states, 
    adata, 
    # TODO: we could expose all of these to the user via kwargs.json.
    num_generations = 20, 
    sol_per_pop = 30, 
    reduce_auto_reg=True, 
    GA_seed_list = [1, 3], 
    init_pop_seed_list = [21, 25],
    # Do not change these.
    cluster_col='type_and_time', 
    pseudoTime_col='timepoint'
)
grn_ig = onesc.dataframe_to_igraph(iGRN)

print("Simulating")
predictions = anndata.AnnData(
    X = np.zeros((predictions_metadata.shape[0], adata.n_vars)),
    obs = predictions_metadata,
    var = adata.var,
)
predictions.obs["error_message"] = ""
for _, starting_state in predictions.obs[['timepoint', 'type_and_time']].drop_duplicates().iterrows():
    initial_state = adata[adata.obs['type_and_time']==starting_state['type_and_time'], :].copy()
    xstates = onesc.define_states_adata(initial_state, min_mean = 0.05, min_percent_cells = 0.20)
    for _, gene_level_steps in predictions.obs[['perturbation', "expression_level_after_perturbation", 'prediction_timescale']].drop_duplicates().iterrows(): 
        goi = gene_level_steps["perturbation"]
        netname = 'onesc_network'
        netsim = onesc.network_structure()
        netsim.fit_grn(iGRN)
        sim = onesc.OneSC_simulator()
        sim.add_network_compilation(netname, netsim)
        perturb_dict = dict()
        perturb_dict[goi] = np.sign(gene_level_steps["expression_level_after_perturbation"] - initial_state[:, goi].X.mean())
        predicted_expression = onesc.simulate_parallel_adata(
            sim, 
            xstates, 
            netname, 
            perturb_dict = perturb_dict, 
            n_cores = 8, 
            num_sim = gene_level_steps["prediction_timescale"]*10,
            num_runs = kwargs["num_cells_to_simulate"],
            t_interval = 0.1, 
            noise_amp = 0.5
        )
        prediction_index = \
            (predictions.obs["type_and_time"]==starting_state["type_and_time"]) & \
            (predictions.obs["timepoint"]==starting_state["timepoint"]) & \
            (predictions.obs["perturbation"]==gene_level_steps["perturbation"]) & \
            (predictions.obs["expression_level_after_perturbation"]==gene_level_steps["expression_level_after_perturbation"]) & \
            (predictions.obs["prediction_timescale"]==gene_level_steps["prediction_timescale"]) 
        for i in np.where(prediction_index)[0]:
            predictions[i, :].X = 0
            for j in range(kwargs["num_cells_to_simulate"]):
                predictions[i, :].X = predictions[i, :].X + predicted_expression[j][gene_level_steps["prediction_timescale"], :].X
            predictions[i, :].X = predictions[i, :].X / kwargs["num_cells_to_simulate"]

print("Summary of error messages encounered during prediction:")
print(predictions.obs["error_message"].value_counts())
print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")