# This is a test script for using dictys via GGRN + docker 
import pereggrn_perturbations
import ggrn.api as ggrn
import numpy as np
import pereggrn_networks
import pandas as pd
pereggrn_networks.set_grn_location("../../../../network_collection/networks") # you may need to change this to the path where the networks are stored on your computer.
pereggrn_perturbations.set_data_path("../../../../perturbation_data/perturbations") # you may need to change this to the path where the perturbations are stored on your computer.
train = pereggrn_perturbations.load_perturbation("definitive_endoderm", is_timeseries=True)
test = pereggrn_perturbations.load_perturbation("definitive_endoderm", is_timeseries=False)
top_genes = train.var.query("highly_variable_rank <= 2000").index
train = train[:, list(set(list(test.uns["perturbed_and_measured_genes"]) + list(top_genes)))]
test  =  test[:, list(set(list(test.uns["perturbed_and_measured_genes"]) + list(top_genes)))]
grn = ggrn.GRN(train,  network=pereggrn_networks.LightNetwork("celloracle_human"))
grn.fit(
    method = "docker____ekernf01/ggrn_docker_backend_dictys", 
    # These args are suitable for fast debugging but nstep=4000 is the default for actual use
    kwargs = {
        'nstep': 40,
        'npc': 0,
        'fi_cov': None,
        'model': 'ou',
        'nstep_report': 1,
        'rseed': 12345,
        'device': 'cpu',
        'dtype': 'float',
        'loss': 'Trace_ELBO_site',
        'nth': 15,
        'varmean': 'N_0val',
        'varstd': None,
        'fo_weightz': None,
        'scale_lyapunov': 1E5
    },
)
predictions_metadata = pd.merge(
                    train.obs[['timepoint', 'cell_type']].drop_duplicates(), 
                    test.obs[['perturbation', 'is_control', 'perturbation_type', "expression_level_after_perturbation"]],
                    how = "cross", 
                )
ad_out = grn.predict(predictions_metadata = predictions_metadata.loc[np.random.choice(predictions_metadata.index, 50),:])