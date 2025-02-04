# This is a test script for using prescient via GGRN + docker (not in the sense of train vs test data).
import pereggrn_perturbations
import ggrn.api as ggrn
import numpy as np
import pandas as pd
from pereggrn.experimenter import find_controls
pereggrn_perturbations.set_data_path("../../../../perturbation_data/perturbations") # you may need to change this to the path where the perturbations are stored on your computer.
train = pereggrn_perturbations.load_perturbation("definitive_endoderm_downsampled", is_timeseries=True)
test = pereggrn_perturbations.load_perturbation("definitive_endoderm_downsampled", is_timeseries=False)
top_genes = train.var.query("highly_variable_rank <= 2000").index
train = train[:, list(set(list(test.uns["perturbed_and_measured_genes"]) + list(top_genes)))]
test  =  test[:, list(set(list(test.uns["perturbed_and_measured_genes"]) + list(top_genes)))]
grn = ggrn.GRN(train, species="human")
grn.fit(
    method = "docker____ekernf01/ggrn_docker_backend_prescient", 
    kwargs = {    
        "pretrain_epochs": 2, # in practice, 500
        "train_epochs": 3, # in practice, 2500
        "save": 3, # Suggest you set this to same value as train_epochs, 
    }
)
# This is different from the test.py for most Docker backends, because does_simulation_progress is set to True.  
predictions_metadata = pd.merge(
    test.obs.head(50)[['perturbation', "expression_level_after_perturbation", "perturbation_type", "is_control"]],
    train.obs[['cell_type', "timepoint"]].head(1),
    how = "cross",
)
predictions_metadata = predictions_metadata.head(50)
print(predictions_metadata.to_csv())
ad_out = grn.predict( predictions_metadata = predictions_metadata, prediction_timescale = [1,2,3,4] )
ad_out = find_controls(ad_out)
assert ad_out.obs["is_control"].any()