import anndata
import json
import os
import numpy as np
import pandas as pd
import scanpy as sc
import subprocess
import torch

train = sc.read_h5ad("from_to_docker/train.h5ad")
with open(os.path.join("from_to_docker/perturbations.json")) as f:
    perturbations = json.load(f)
with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)
    assert kwargs
with open(os.path.join("from_to_docker/ggrn_args.json")) as f:
    ggrn_args = json.load(f)

# Set defaults
if "pretrain_epochs" not in kwargs:
    kwargs["pretrain_epochs"] = "50"

subprocess.call(["python3", "estimate-growth-rates.py", 
                 "--input_h5ad", "from_to_docker/train.h5ad", 
                 "--outfile", "growth_rates.pt"])

subprocess.call([
    "prescient", "process_data", 
    "-d", "from_to_docker/train.h5ad", 
    "-o", "train", 
    "--tp_col", "timepoint", 
    "--fix_non_consecutive", 
    "--celltype_col", "cell_type", 
    "--growth_path", "growth_rates.pt",
    "--num_pcs", "50", # near future: grab this from ggrn_args. Note it also affects the prediction step circa line 77
])

subprocess.call([
    "prescient", "train_model", 
    "-i", "traindata.pt", 
    "--out_dir", "prescient_trained/", 
    # The following code will be useful if we decide to pass through any of the usual args for prescient process_data.
    # Note: altering these values would also affect the paths provided to torch.load and prescient perturbation_analysis. 
    # "--no-cuda",
    # "--gpu", "GPU",             
    "--pretrain_epochs", str(kwargs["pretrain_epochs"]),
    "--train_epochs", "50", # kwargs["train_epochs"],
    # "--train_lr", kwargs["train_lr"],
    # "--train_dt", kwargs["train_dt"],
    # "--train_sd", kwargs["train_sd"],
    # "--train_tau", kwargs["train_tau"],
    # "--train_batch", kwargs["train_batch"],
    # "--train_clip", kwargs["train_clip"],
    # "--save", kwargs["save"],
    # "--pretrain", kwargs["pretrain"],
    # "--train", kwargs["train"],
    # "--config", kwargs["config"],
    "--weight_name", "kegg-growth"
])

print(os.listdir("prescient_trained"))

predictions = anndata.AnnData(
    X = np.zeros((len(perturbations), train.n_vars)),
    obs = pd.DataFrame({
        "perturbation":                       [p[0] for p in perturbations], 
        "expression_level_after_perturbation":[p[1] for p in perturbations], 
    }), 
    var = train.var,
)
for i, goilevel in enumerate(perturbations):
    goi, level = goilevel
    print("Predicting " + goi)
    assert "," not in goi, "PRESCIENT and GGRN can each handle multi-gene perturbations, but the interface between them currently cannot. Sorry."
    control = train[train.obs["is_control"], goi].X.mean()
    if level < control:
        z = -5
    if level > control:
        z = 5
    os.makedirs("model")
    subprocess.call([
        "prescient", "perturbation_analysis", 
        "-i", "traindata.pt", 
        "-p", f"'{goi}'", 
        "-z", str(z), 
        "--num_pcs", "50", # grab this from GGRN args
        "--model_path", f"prescient_trained/kegg-growth-softplus_1_{str(kwargs['pretrain_epochs'])}-1e-06", # This is affected by PRESCIENT kwargs 
        "--num_steps", "10", 
        "--seed", "2", 
        "-o", "experiments/",
    ])
    # result = torch.load("experiments/seed_2_train.epoch_002500_num.sims_10_num.cells_200_num.steps_10_subsets_None_None_perturb_simulation.pt")
    print(os.listdir("experiments"))
    result = torch.load("experiments/seed_2_train.epoch_000050_num.sims_10_num.cells_200_num.steps_10_subsets_None_None_perturb_simulation.pt")
    print(type(result))
    # TODO: find the results.
    # TODO: Convert the results from PCA-space back to expression-space
    predictions[i, :].X = 1


print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")