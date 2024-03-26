import anndata
import json
import os
import numpy as np
import pandas as pd
import scanpy as sc
import subprocess
import torch
import sklearn
import itertools 
train = sc.read_h5ad("from_to_docker/train.h5ad")
predictions_metadata = pd.read_csv("predictions_metadata.csv")
assert all(predictions_metadata.columns == np.array(['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation", 'prediction_timescale']))

with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)
    assert kwargs


with open(os.path.join("from_to_docker/ggrn_args.json")) as f:
    ggrn_args = json.load(f)


# Set defaults
defaults = {
    "pretrain_epochs": 500,
    "train_epochs": 2500,
    "save": 5, # Suggest you set this to same value as train_epochs
    "train_dt": 0.1,
}
for k in defaults:
    if k not in kwargs:
        kwargs[k] = defaults[k]


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
    "--num_pcs", str(ggrn_args["low_dimensional_value"]), 
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
    "--train_epochs", str(kwargs["train_epochs"]), # kwargs["train_epochs"],
    # "--train_lr", kwargs["train_lr"],
    "--train_dt", str(kwargs["train_dt"]), # This affects the prediction timescale too!!
    # "--train_sd", kwargs["train_sd"],
    # "--train_tau", kwargs["train_tau"],
    # "--train_batch", kwargs["train_batch"],
    # "--train_clip", kwargs["train_clip"],
    "--save", str(kwargs["save"]),
    # "--pretrain", kwargs["pretrain"],
    # "--train", kwargs["train"],
    # "--config", kwargs["config"],
    "--weight_name", "kegg-growth"
])


# Below, this PCA and scaler will help us transform predictions back to the scale of the data in from_to_docker/train.h5ad.
original_expression = torch.load("traindata.pt")["data"]
pca_model = torch.load("traindata.pt")["pca"]
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit_transform(original_expression)


# Make predictions
#
# Let's talk about prescient timescales.
# 
# First, before we talk about the model, a basic thing to specify is the starting cell state, via  "--celltype_subset" and "--tp_subset".
# 
# Second, input timepoints are transformed to be consecutive starting from 0 during process_data (flag --fix_non_consecutive).
# So if training samples are labeled 0, 2, 12, 24, and I want a prediction at time 18, I would proceed as follows. 
# 12 is mapped to 2 and 24 is mapped to 3.
# I want predictions at 18, so that would be mapped to 2.5 during process_data. I currently do piecewise linear interpolation.
# 
# Third, it's a differential equation method. It's solved via time discretization, but still the natural scale to run the model on is
# a finer discretization of the space than the original data.
# [prescient docs](https://cgs.csail.mit.edu/prescient/documentation/) say there's a default dt of 0.1, so ten steps per time-point.
# So if I want expression at time 2.5, I should feed 25 to num_steps below.
# 
# Third, as of 2024 March 13, interpolation with PRESCIENT is also tricky, and I have not read the details yet. https://github.com/gifford-lab/prescient/issues/3

tps = train.obs["timepoint"]
time_points = {
    "train_original": np.sort(np.unique(tps)), 
    "train_consecutive": np.arange(0, len(np.unique(tps))),
    "predict_original": predictions_metadata["prediction_timescale"].unique(),
}
assert max(time_points["predict_original"]) <= max(time_points["train_original"]), "GGRN's PRESCIENT backend cannot currently extrapolate outside the original time-window."
assert min(time_points["predict_original"]) >= min(time_points["train_original"]), "GGRN's PRESCIENT backend cannot currently extrapolate outside the original time-window."
time_points["predict_consecutive"] = np.interp(time_points["predict_original"], time_points["train_original"], time_points["train_consecutive"])
time_points["predict_steps"] = [int(round(i/kwargs["train_dt"])) for i in time_points["predict_consecutive"]]
print("Timescale: ")
print(time_points)

print("Running simulations")
# Note the timepoint metadata is on the original scale e.g. 0, 2, 12, 24.
# The num_steps is on prescient's internal scale, accounting for the value of dt, e.g. 0, 10, 20, 30 if dt is 0.1.
predictions = anndata.AnnData(
    X = np.zeros((predictions_metadata.shape[0], train.n_vars)),
    obs = predictions_metadata,
    var = train.var,
)
for gene_level_steps in predictions.obs[['perturbation', "expression_level_after_perturbation", 'num_steps']].unique():
    print("Predicting " + gene_level_steps["perturbation"])
    for starting_state in predictions.obs[['timepoint', 'cell_type']].unique():
        prediction_index = \
            predictions.obs["cell_type"]==starting_state["cell_type"] & \
            predictions.obs["timepoint"]==starting_state["timepoint"] & \
            predictions.obs["perturbation"]==gene_level_steps["perturbation"] & \
            predictions.obs["expression_level_after_perturbation"]==gene_level_steps["expression_level_after_perturbation"] & \
            predictions.obs["num_steps"]==gene_level_steps["num_steps"] 
        try:
            train_index = train.obs["cell_type"]==starting_state["cell_type"] & train.obs["timepoint"]==starting_state["timepoint"]
            goi, level, num_steps = gene_level_steps
            assert "," not in goi, "PRESCIENT and GGRN can each handle multi-gene perturbations, but the interface between them currently cannot. Sorry."
            control = train[train_index, goi].X.mean()
            if level < control:
                z = -5
            if level > control:
                z = 5
            subprocess.call([
                "prescient", "perturbation_analysis", 
                "-i", "traindata.pt", 
                "-p", f"'{goi}'", 
                "-z", str(z), 
                "--epoch", f"{int(kwargs['train_epochs']):06}",
                "--num_pcs",  str(ggrn_args["low_dimensional_value"]),
                "--model_path", "prescient_trained/kegg-growth-softplus_1_500-1e-06", 
                "--num_steps", str(max(time_points["predict_steps"])),
                "--num_cells", "1",
                "--num_sims", "1",
                "--celltype_subset", starting_state["cell_type"], 
                "--tp_subset", starting_state["timepoint"], 
                "--seed", "2", 
                "-o", "prescient_trained",
            ])
            result = torch.load("prescient_trained/kegg-growth-softplus_1_500-1e-06/"
                                "seed_2"
                                "_train.epoch_" f"{int(kwargs['train_epochs']):06}"
                                "_num.sims_" "1"
                                "_num.cells_" "1"
                                "_num.steps_" "10"
                                "_subsets_" "None_None"
                                "_perturb_simulation.pt")
            pca_predicted_embeddings = result["perturbed_sim"][0].squeeze()[time_points["predict_steps"],:] 
            # Now we un-do all of PRESCIENT's preprocessing in order to return data on the scale of the training data input. 
            scaled_expression = pca_model.inverse_transform(pca_predicted_embeddings)
            predictions[prediction_index, :].X = scaler.inverse_transform(scaled_expression) 
        except ValueError as e:
            predictions[prediction_index, :].X = np.nan
            print("Prediction failed for the following genes and cell types: \n" + gene_level_steps.to_csv() + starting_state.to_csv() + "\nError text: " + str(e))

    
print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")