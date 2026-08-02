# This is a test script for using scGPT via GGRN + docker (not in the sense of train vs test data).
import os
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

# Persistently cache the ~1.5GB scGPT checkpoint on the host so it is downloaded only once, not on
# every run. GGRN only auto-mounts from_to_docker/ (and wipes it each fit), so we pass an extra bind
# mount (and --gpus all) to `docker run` via the method string, whose format is docker__<args>__<image>.
# The container's default checkpoint dir is /scgpt_weights/scGPT_human, so we mount our host cache at
# /scgpt_weights. First run downloads into the cache; later runs reuse it.
cache_dir = os.path.expanduser("~/.cache/scgpt_weights")
os.makedirs(cache_dir, exist_ok=True)
import subprocess
has_gpu = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
gpu_flag = "--gpus all " if has_gpu else ""
docker_args = f"{gpu_flag}-v {cache_dir}:/scgpt_weights"
grn.fit(
    method = f"docker__{docker_args}__ekernf01/ggrn_docker_backend_{name}",
    kwargs = {"epochs": 2},   # keep the fine-tuning short for a smoke test
)
ad_out = grn.predict(
    predictions_metadata = test.obs.loc[
        np.random.choice(test.obs_names, 50),
        ['perturbation', "expression_level_after_perturbation", "is_control"]
    ],
    prediction_timescale = 0,
)
