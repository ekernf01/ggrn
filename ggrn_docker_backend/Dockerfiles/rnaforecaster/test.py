# This is a test script for using sckinetics via GGRN + docker 
import pereggrn_perturbations
import ggrn.api as ggrn
import numpy as np
import pereggrn_networks
pereggrn_networks.set_grn_location("../../../../network_collection/networks") # you may need to change this to the path where the networks are stored on your computer.
pereggrn_perturbations.set_data_path("../../../../perturbation_data/perturbations") # you may need to change this to the path where the perturbations are stored on your computer.
test = pereggrn_perturbations.load_perturbation("definitive_endoderm_downsampled", is_timeseries=False)
train = pereggrn_perturbations.load_perturbation("definitive_endoderm_downsampled", is_timeseries=True)
top_genes = train.var.query("highly_variable_rank <= 500").index
train = train[:, top_genes]
test = test[:, top_genes]   
grn = ggrn.GRN(train)
name = "rnaforecaster"
grn.fit(
    method = f"docker____ekernf01/ggrn_docker_backend_{name}",
)
ad_out = grn.predict(
    predictions_metadata = test.obs.loc[np.random.choice(test.obs_names, 50), ['perturbation', "expression_level_after_perturbation", "is_control"]], 
    prediction_timescale = [1,2],
)
