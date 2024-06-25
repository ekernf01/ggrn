import anndata
import json
import os
import numpy as np
import pandas as pd
import scanpy as sc
import subprocess
import torch
from sklearn import preprocessing

train = sc.read_h5ad("from_to_docker/train.h5ad")
predictions_metadata = pd.read_csv("from_to_docker/predictions_metadata.csv")
assert predictions_metadata.shape[0] > 0, "predictions_metadata is empty."
for c in ['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation", 'prediction_timescale']:
    assert c in predictions_metadata.columns, f"For the ggrn prescient backend, predictions_metadata.columns is required to contain {c}."

with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)


with open(os.path.join("from_to_docker/ggrn_args.json")) as f:
    ggrn_args = json.load(f)

# Set defaults
defaults = {
    "pretrain_epochs": 500,
    "train_epochs": 2500,
    "save": 2500, # Suggest you set this to same value as train_epochs
    "train_dt": 0.1,
    "num_cells_to_simulate": 100,
}
ggrn_defaults = {
    "low_dimensional_value": 50
}
for k in defaults:
    if k not in kwargs:
        kwargs[k] = defaults[k]
for k in ggrn_defaults:
    if k not in ggrn_args:
        ggrn_args[k] = ggrn_defaults[k]


# Let's talk about prescient timescales. Internally, we use 3 (three) different timescales.
# 
# Input timepoints are transformed to be consecutive starting from 0 during process_data (flag --fix_non_consecutive).
# So if training samples are labeled 0, 2, 12, 24, we would re-label them 0,1,2,3,4. 
# So now we have two timescales, "original" and "consecutive".
# 
# Next, it's a differential equation method. It's solved via time discretization. 
# The [prescient docs](https://cgs.csail.mit.edu/prescient/documentation/) say there's a default dt of 0.1, so ten steps per time-point.
# So if I want expression at time 2, I should feed 20 to prescient's num_steps arg.
# So now we have three timescales, "original" and "consecutive" and "steps". 
# 
# Finally: as of 2024 March 13, interpolation with PRESCIENT is also tricky, and I have not read the details yet. https://github.com/gifford-lab/prescient/issues/3
# 
# Guidance to the user: 
# 
# - the "timepoint" in the `obs` of `train.h5ad` and in `predictions_metadata` should be on the original scale e.g. 0, 2, 12, 24. 
# - `prediction_timescale` in `predictions_metadata` should be on the "consecutive" scale, so a 2 would bring you from 0 to 12 or from 2 to 24.
tps = train.obs["timepoint"]
time_scales = {
    "train_original": np.sort(np.unique(tps)), 
    "train_consecutive": np.arange(0, len(np.unique(tps))),
}
time_scales["train_steps"] = [int(round(i/kwargs["train_dt"])) for i in time_scales["train_consecutive"]]
print("Timescale: ")
print(time_scales)
time_scales["convert_original_to_consecutive"] = {time_scales["train_original"   ][i]:time_scales["train_consecutive"][i] for i in range(len(time_scales["train_original"]))}
time_scales["convert_consecutive_to_steps"]    = {time_scales["train_consecutive"][i]:time_scales["train_steps"      ][i] for i in range(len(time_scales["train_original"]))}
time_scales["convert_consecutive_to_original"] = {time_scales["train_consecutive"][i]:time_scales["train_original"   ][i] for i in range(len(time_scales["train_original"]))}
time_scales["convert_steps_to_consecutive"]    = {time_scales["train_steps"      ][i]:time_scales["train_consecutive"][i] for i in range(len(time_scales["train_original"]))}
# Remove any trajectory running off the end. 
predictions_metadata["timepoint_original"] = predictions_metadata["timepoint"].copy()
predictions_metadata["timepoint_consecutive"] = predictions_metadata["timepoint_original"].map(time_scales["convert_original_to_consecutive"])
predictions_metadata["takedown_timepoint_consecutive"] = predictions_metadata["timepoint_consecutive"] + predictions_metadata["prediction_timescale"]
final_timepoint_consecutive = time_scales['train_consecutive'].max()
if any( predictions_metadata["takedown_timepoint_consecutive"] > final_timepoint_consecutive):
    print(f"WARNING: some requested predictions are beyond the end of the training data (time point {final_timepoint_consecutive}). These will be removed.")
    print("requested time points (timepoint, num_observations):")
    print(predictions_metadata["takedown_timepoint_consecutive"].value_counts())
    predictions_metadata = predictions_metadata.query("takedown_timepoint_consecutive <= @final_timepoint_consecutive")
    print("time points that will be predicted (timepoint, num_observations):")
    print(predictions_metadata["takedown_timepoint_consecutive"].value_counts())
    assert predictions_metadata.shape[0] > 0, "All predictions were beyond the end of the training data. Please adjust the prediction_timescale in predictions_metadata.csv."
    
# Direct input to PRESCIENT: consecutive-ized timepoints and number of steps
predictions_metadata["timepoint"] = predictions_metadata["timepoint_consecutive"].copy()
predictions_metadata["prediction_timescale_steps"] = [time_scales["convert_consecutive_to_steps"][t]    for t in predictions_metadata["prediction_timescale"]]


print("Estimating growth rates.", flush=True)
subprocess.call(["python3", "estimate-growth-rates.py", 
                 "--input_h5ad", "from_to_docker/train.h5ad", 
                 "--outfile", "growth_rates.pt"])

print("Processing input data.", flush=True)
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

print("Training model.", flush=True)
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
print("Decoding output.", flush=True)
original_expression = torch.load("traindata.pt")["data"]
pca_model = torch.load("traindata.pt")["pca"]
scaler = preprocessing.StandardScaler()
scaler.fit_transform(original_expression)


# Make predictions
print("Running simulations", flush = True)
predictions = anndata.AnnData(
    X = np.zeros((predictions_metadata.shape[0], train.n_vars)),
    obs = predictions_metadata,
    var = train.var,
)

for _, current_prediction_metadata in predictions.obs[['perturbation', "expression_level_after_perturbation", 'prediction_timescale_steps', 'timepoint', 'cell_type']].drop_duplicates().iterrows():
    prediction_index = \
        (predictions.obs["cell_type"]                          ==current_prediction_metadata["cell_type"]) & \
        (predictions.obs["timepoint"]                          ==current_prediction_metadata["timepoint"]) & \
        (predictions.obs["perturbation"]                       ==current_prediction_metadata["perturbation"]) & \
        (predictions.obs["expression_level_after_perturbation"]==current_prediction_metadata["expression_level_after_perturbation"]) & \
        (predictions.obs["prediction_timescale_steps"]         ==current_prediction_metadata["prediction_timescale_steps"]) 
    assert prediction_index.sum() >= 1, "Prediction metadata are out of sync with themselves. Please report this bug."
    try:
        train_index = (train.obs["cell_type"]==current_prediction_metadata["cell_type"]) & (train.obs["timepoint"]==current_prediction_metadata["timepoint"])
        assert train_index.sum() >= 1, "No training samples have the specified combination of cell_type and timepoint."
        goi, level, predict_steps = current_prediction_metadata[['perturbation', "expression_level_after_perturbation", 'prediction_timescale_steps']]
        assert "," not in goi, "PRESCIENT and GGRN can each handle multi-gene perturbations, but the interface between them currently cannot. Sorry."
        # handle samples labeled e.g. "control" or "scramble"
        if goi in train.var_names:
            control = train[train_index, goi].X.mean()
            if level < control:
                z = -5
            if level > control:
                z = 5
        else:
            z = 0
            goi = train.var_names[0]
        subprocess.call([
            "prescient", "perturbation_analysis", 
            "-i", "traindata.pt", 
            "-p", f"'{goi}'", 
            "-z", str(z), 
            "--epoch", f"{int(kwargs['train_epochs']):06}",
            "--num_pcs",  str(ggrn_args["low_dimensional_value"]),
            "--model_path", "prescient_trained/kegg-growth-softplus_1_500-1e-06", 
            "--num_steps", str(predict_steps),
            "--num_cells", str(kwargs["num_cells_to_simulate"]),
            "--num_sims", "1",
            "--celltype_subset", str(current_prediction_metadata["cell_type"]), 
            "--tp_subset", str(current_prediction_metadata["timepoint"]), 
            "--seed", "2", 
            "-o", "prescient_trained",
        ])
        result = torch.load("prescient_trained/kegg-growth-softplus_1_500-1e-06/"
                            "seed_2"
                            "_train.epoch_" f"{int(kwargs['train_epochs']):06}"
                            "_num.sims_" "1"
                            "_num.cells_" f"{kwargs['num_cells_to_simulate']}"
                            "_num.steps_" f"{current_prediction_metadata['prediction_timescale_steps']}" 
                            "_subsets_" f"{str(current_prediction_metadata['timepoint'])}_{current_prediction_metadata['cell_type']}"
                            "_perturb_simulation.pt"
                            )
        # Average multiple trajectories to get a single trajectory
        pca_predicted_embeddings = result["perturbed_sim"][0][0:predict_steps,:,:].mean(axis=(0,1), keepdims = False).reshape(1, -1)
        # Now we un-do all of PRESCIENT's preprocessing in order to return data on the scale of the training data input. 
        scaled_expression = pca_model.inverse_transform(pca_predicted_embeddings)
        for i in np.where(prediction_index)[0]:
            predictions[i, :].X = scaler.inverse_transform(scaled_expression) 
    except Exception as e:
        predictions[prediction_index, :].X = np.nan
        print("Prediction failed. Error text: " + str(e))
        print("Metadata: " + current_prediction_metadata.to_csv(sep=" "))

# Convert back to original timepoint labels before returning the data
predictions.obs["timepoint"] = [time_scales["convert_consecutive_to_original"][t] for t in predictions.obs["timepoint"]]
print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")