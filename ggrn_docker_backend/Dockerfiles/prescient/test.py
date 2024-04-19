# This is a test script for using prescient via GGRN + docker (not in the sense of train vs test data).
import pereggrn_perturbations
import ggrn.api as ggrn
import shutil
import numpy as np
pereggrn_perturbations.set_data_path("../../../../perturbation_data/perturbations") # you may need to change this to the path where the perturbations are stored on your computer.
train = pereggrn_perturbations.load_perturbation("definitive_endoderm", is_timeseries=True)
test = pereggrn_perturbations.load_perturbation("definitive_endoderm", is_timeseries=False)
top_genes = train.var.query("highly_variable_rank <= 2000").index
train = train[:, list(set(list(test.uns["perturbed_and_measured_genes"]) + list(top_genes)))]
test  =  test[:, list(set(list(test.uns["perturbed_and_measured_genes"]) + list(top_genes)))]
grn = ggrn.GRN(train)
grn.fit(
    method = "docker____ekernf01/ggrn_docker_backend_prescient", 
    kwargs = {    
        "pretrain_epochs": 2, # in practice, 500
        "train_epochs": 3, # in practice, 2500
        "save": 3, # Suggest you set this to same value as train_epochs
    },
)
ad_out = grn.predict(predictions_metadata = test.obs.loc[np.random.choice(test.obs_names, 50), ['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation"]])
shutil.rmtree("from_to_docker")