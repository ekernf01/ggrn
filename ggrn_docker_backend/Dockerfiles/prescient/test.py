# This is a test script for using prescient via GGRN + docker (not in the sense of train vs test data).
import pereggrn_perturbations
import ggrn.api as ggrn
import shutil
pereggrn_perturbations.set_data_path("../../../../perturbation_data/perturbations") # you may need to change this to the path where the perturbations are stored on your computer.
ad = pereggrn_perturbations.load_perturbation("definitive_endoderm")
ad = ad[:, ad.uns["perturbed_and_measured_genes"]]
grn = ggrn.GRN(ad)
grn.fit(method = "docker____ekernf01/ggrn_docker_backend_prescient")
ad_out = grn.predict(predictions_metadata = ad.obs[['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation"]])
shutil.unlink("from_to_docker")