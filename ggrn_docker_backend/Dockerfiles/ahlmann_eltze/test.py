# This is a test script for using sckinetics via GGRN + docker 
import pereggrn_perturbations
import ggrn.api as ggrn
import numpy as np
import pereggrn_networks
pereggrn_networks.set_grn_location("../../../../network_collection/networks") # you may need to change this to the path where the networks are stored on your computer.
pereggrn_perturbations.set_data_path("../../../../perturbation_data/perturbations") # you may need to change this to the path where the perturbations are stored on your computer.
test = pereggrn_perturbations.load_perturbation("replogle2_tf_only", is_timeseries=False)
top_genes = test.var.query("highly_variable_rank <= 2000").index
test  =  test[:, list(set(list(test.uns["perturbed_and_measured_genes"]) + list(top_genes)))]
grn = ggrn.GRN(test)
grn.fit(
    method = "docker____ekernf01/ggrn_docker_backend_ahlmann_eltze",
)
ad_out = grn.predict(predictions_metadata = test.obs.loc[np.random.choice(test.obs_names, 50), ['perturbation', "expression_level_after_perturbation", "is_control"]], prediction_timescale = 0)
ad_out = grn.predict(predictions_metadata = test.obs.loc[np.random.choice(test.obs_names, 50), ['perturbation', "expression_level_after_perturbation", "is_control"]], prediction_timescale = 1)
ad_out = grn.predict(predictions_metadata = test.obs.loc[np.random.choice(test.obs_names, 50), ['perturbation', "expression_level_after_perturbation", "is_control"]], prediction_timescale = 2)
