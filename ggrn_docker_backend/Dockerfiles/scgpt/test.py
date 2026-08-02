# This is a test script for using scGPT via GGRN + docker (not in the sense of train vs test data).
import pereggrn_perturbations
import ggrn.api as ggrn
import numpy as np
import pereggrn_networks
pereggrn_networks.set_grn_location("../../../../network_collection/networks") # you may need to change this to the path where the networks are stored on your computer.
pereggrn_perturbations.set_data_path("../../../../perturbation_data/perturbations") # you may need to change this to the path where the perturbations are stored on your computer.

# scGPT's perturbation task is GEARS-style: it learns to predict the (single-cell) expression
# profile of control cells after a genetic perturbation. It does not need a base GRN and it does
# not use timeseries structure, so we test it on the non-timeseries "nakatake" dataset, as the
# template does.
test = pereggrn_perturbations.load_perturbation("nakatake", is_timeseries=False)
top_genes = test.var.query("highly_variable_rank <= 2000").index
test  =  test[:, list(set(list(test.uns["perturbed_and_measured_genes"]) + list(top_genes)))]

grn = ggrn.GRN(test)
name = "scgpt"
grn.fit(
    method = f"docker____ekernf01/ggrn_docker_backend_{name}",
    kwargs = {"epochs": 2},   # keep the fine-tuning short for a smoke test
)
ad_out = grn.predict(
    predictions_metadata = test.obs.loc[
        np.random.choice(test.obs_names, 50),
        ['perturbation', "expression_level_after_perturbation", "is_control"]
    ],
    prediction_timescale = 0,
)
