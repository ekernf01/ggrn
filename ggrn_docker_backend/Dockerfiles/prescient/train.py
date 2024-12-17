import anndata
import json
import os
import numpy as np
import pandas as pd
import scanpy as sc
import subprocess
import torch
from sklearn import preprocessing
import warnings
# suppress an annoying FutureWarning whose instructions are vague and I could not satisfy them
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    "num_cells_to_simulate": 20,
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

# We can only predict KO consequences for detected genes. 
predictions_metadata["perturbed_gene_is_measured"] = [
    all([g in train.var_names for g in p.split(",")]) 
    for p in predictions_metadata["perturbation"]
]
print(f"PRESCIENT can only predict KO consequences for measured genes. Will not make predictions for {np.sum(1-predictions_metadata['perturbed_gene_is_measured'])} unmeasured genes.")
predictions_metadata = predictions_metadata.query("perturbed_gene_is_measured")

# SEE THE README ON TIMESCALES!!
tps = train.obs["timepoint"]
time_scales = {
    "train_original": np.sort(np.unique(tps)), 
    "train_consecutive": np.arange(0, len(np.unique(tps))),
}
time_scales["train_steps"] = [int(round(i/kwargs["train_dt"])) for i in time_scales["train_consecutive"]]
print("Timescale: ")
print(pd.DataFrame(time_scales))
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
    print(predictions_metadata["takedown_timepoint_consecutive"].value_counts().sort_index(inplace=False))
    predictions_metadata = predictions_metadata.query("takedown_timepoint_consecutive <= @final_timepoint_consecutive")
    print("time points that will be predicted (timepoint, num_observations):")
    print(predictions_metadata["takedown_timepoint_consecutive"].value_counts().sort_index(inplace=False))
    assert predictions_metadata.shape[0] > 0, "All predictions were beyond the end of the training data. Please adjust the prediction_timescale in predictions_metadata.csv."
    
# Direct input to PRESCIENT: consecutive-ized timepoints and number of steps
predictions_metadata["timepoint"] = predictions_metadata["timepoint_consecutive"].copy()
predictions_metadata["prediction_timescale_steps"] = [time_scales["convert_consecutive_to_steps"][t]    for t in predictions_metadata["prediction_timescale"]]

print(f"Estimating growth rates assuming species is {kwargs['species']}.", flush=True)
assert kwargs["species"] + "_birth.csv" in os.listdir(), f"Proliferation-associated genes for {kwargs['species']} not found. Cannot run PRESCIENT."
assert kwargs["species"] + "_death.csv" in os.listdir(), f"Apoptosis-associated genes for {kwargs['species']} not found. Cannot run PRESCIENT."
_ = subprocess.run(["/opt/venv/bin/python", "estimate-growth-rates.py", 
                "--input_h5ad", "from_to_docker/train.h5ad", 
                "--birth_gst", kwargs["species"] + "_birth.csv",
                "--death_gst", kwargs["species"] + "_death.csv",
                "--outfile", "growth_rates.pt"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)

print("Processing input data.", flush=True)
_ = subprocess.run(
    [
    "/opt/venv/bin/python", "-m", 
    "prescient", "process_data", 
    "-d", "from_to_docker/train.h5ad", 
    "-o", "train", 
    "--tp_col", "timepoint", 
    "--fix_non_consecutive", 
    "--celltype_col", "cell_type", 
    "--growth_path", "growth_rates.pt",
    "--num_pcs", str(ggrn_args["low_dimensional_value"]), 
], stderr=subprocess.PIPE, stdout=subprocess.PIPE)

print("Training model.", flush=True)
_ = subprocess.run([
    "/opt/venv/bin/python", "-m", 
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
], stderr=subprocess.PIPE, stdout=subprocess.PIPE)


# Below, this PCA and scaler will help us transform predictions back to the scale of the data in from_to_docker/train.h5ad.
print("Decoding output.", flush=True)
original_expression = torch.load("traindata.pt")["data"]
pca_model = torch.load("traindata.pt")["pca"]
scaler = preprocessing.StandardScaler()
scaler.fit_transform(original_expression)


# Make predictions
print(f"Running {predictions_metadata.shape[0]} simulations", flush = True)
predictions = anndata.AnnData(
    X = np.zeros((predictions_metadata.shape[0]*kwargs["num_cells_to_simulate"], train.n_vars)),
    obs = pd.concat([predictions_metadata for _ in range(kwargs["num_cells_to_simulate"])]),
    var = train.var,
)
predictions.obs_names_make_unique()
del predictions_metadata #this is now stored in predictions.obs

predictions.obs["error_message"] = "no prediction made"
for _, current_prediction_metadata in predictions.obs.drop_duplicates().iterrows():
    prediction_index = \
        (predictions.obs["cell_type"]                           ==current_prediction_metadata["cell_type"]) & \
        (predictions.obs["timepoint_consecutive"]               ==current_prediction_metadata["timepoint"]) & \
        (predictions.obs["perturbation"]                        ==current_prediction_metadata["perturbation"]) & \
        (predictions.obs["expression_level_after_perturbation"] ==current_prediction_metadata["expression_level_after_perturbation"]) & \
        (predictions.obs["prediction_timescale_steps"]          ==current_prediction_metadata["prediction_timescale_steps"]) 
    prediction_index = prediction_index[prediction_index].index
    if len(prediction_index) < 1:
        print("No predictions to make for this metadatum.")
        print(current_prediction_metadata.to_csv(sep=" "))
        continue
    try:
        train_index = (train.obs["cell_type"]==current_prediction_metadata["cell_type"]) & (train.obs["timepoint"]==current_prediction_metadata["timepoint_original"])
        assert train_index.sum() >= 1, "No training samples have the specified combination of cell_type and timepoint."
        goi, level, predict_steps = current_prediction_metadata[['perturbation', "expression_level_after_perturbation", 'prediction_timescale_steps']]
        assert "," not in goi, "PRESCIENT and GGRN can each handle multi-gene perturbations, but the interface between them currently cannot. Sorry."
        if goi in train.var_names:
            # handle observations labeled with a gene name
            control = train[train_index, goi].X.mean()
            if float(level) < control:
                z = -5
            if float(level) > control:
                z = 5
            if float(level) == control:
                z = 0
        else:
            # handle samples labeled e.g. "control" or "scramble" by selecting any gene and then not perturbing it.
            z = 0
            goi = train.var_names[0]
        
        _ = subprocess.run([
            "/opt/venv/bin/python", "-m", 
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
        ], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        result = torch.load("prescient_trained/kegg-growth-softplus_1_500-1e-06/"
                            "seed_2"
                            "_train.epoch_" f"{int(kwargs['train_epochs']):06}"
                            "_num.sims_" "1"
                            "_num.cells_" f"{kwargs['num_cells_to_simulate']}"
                            "_num.steps_" f"{current_prediction_metadata['prediction_timescale_steps']}" 
                            "_subsets_" f"{str(current_prediction_metadata['timepoint'])}_{current_prediction_metadata['cell_type']}"
                            "_perturb_simulation.pt", 
                            )
        # Average multiple trajectories to get a single trajectory
        pca_predicted_embeddings = result["perturbed_sim"][0][predict_steps,:,:]
        # Now we un-do all of PRESCIENT's preprocessing in order to return data on the scale of the training data input. 
        scaled_expression = pca_model.inverse_transform(pca_predicted_embeddings)
        predictions[prediction_index, :].X = scaler.inverse_transform(scaled_expression) 
        predictions.obs.loc[prediction_index, "error_message"] = "no errors"
    except Exception as e:
        predictions[prediction_index, :].X = np.nan
        predictions.obs.loc[prediction_index, "error_message"] =  repr(e)



print("Summary of error messages encounered during prediction:")
print(predictions.obs["error_message"].value_counts())
# Convert back to original timepoint labels before returning the data
predictions.obs["timepoint"] = [time_scales["convert_consecutive_to_original"][t] for t in predictions.obs["timepoint"]]
print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")