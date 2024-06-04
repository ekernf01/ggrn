import anndata
import json
import os
import numpy as np
import pandas as pd
import scanpy as sc
import subprocess
import torch
from sklearn import preprocessing


# Thanks chatgpt for this lovely snippet.
def show_tree(dir_path, indent=''):
    print(indent + os.path.basename(dir_path) + '/')
    indent += '    '
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            show_tree(item_path, indent)
        else:
            print(indent + item)

train = sc.read_h5ad("from_to_docker/train.h5ad")
predictions_metadata = pd.read_csv("from_to_docker/predictions_metadata.csv")
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
#
# Let's talk about prescient timescales.
# 
# First, before we talk about the model, a basic thing to specify is the starting cell state, via  "--celltype_subset" and "--tp_subset".
# 
# Second, input timepoints are transformed to be consecutive starting from 0 during process_data (flag --fix_non_consecutive).
# So if training samples are labeled 0, 2, 12, 24, and I want a prediction at time 18, I would proceed as follows. 
# 12 is labeled as 2 and 24 is labeled as 3.
# I want predictions at 18, so that would be about a 2.5 during process_data. (I currently do piecewise linear interpolation.)
# 
# Third, it's a differential equation method. It's solved via time discretization, but still the natural scale to run the model on is
# a finer discretization of the space than the original data.
# The [prescient docs](https://cgs.csail.mit.edu/prescient/documentation/) say there's a default dt of 0.1, so ten steps per time-point.
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
time_points["converter"] = {time_points["predict_original"][i]:time_points["predict_steps"][i] for i in range(len(time_points["predict_original"]))}
print("Timescale: ")
print(time_points)
predictions_metadata["predict_steps"] = [time_points["converter"][t] for t in predictions_metadata["prediction_timescale"]]

print("Running simulations", flush = True)
# Note the timepoint metadata is on the original scale e.g. 0, 2, 12, 24.
# The predict_steps is on prescient's internal scale, accounting for the value of dt, e.g. 0, 10, 20, 30 if dt is 0.1.
predictions = anndata.AnnData(
    X = np.zeros((predictions_metadata.shape[0], train.n_vars)),
    obs = predictions_metadata,
    var = train.var,
)

def get_unique_rows(df, factors): 
    return df[factors].groupby(factors).mean().reset_index()

for _, current_prediction_metadata in get_unique_rows(predictions.obs, ['perturbation', "expression_level_after_perturbation", 'predict_steps', 'timepoint', 'cell_type']).iterrows():
    print("Predicting " + current_prediction_metadata.to_csv(sep=" "))
    prediction_index = \
        (predictions.obs["cell_type"]==current_prediction_metadata["cell_type"]) & \
        (predictions.obs["timepoint"]==current_prediction_metadata["timepoint"]) & \
        (predictions.obs["perturbation"]==current_prediction_metadata["perturbation"]) & \
        (predictions.obs["expression_level_after_perturbation"]==current_prediction_metadata["expression_level_after_perturbation"]) & \
        (predictions.obs["predict_steps"]==current_prediction_metadata["predict_steps"]) 
    try:
        train_index = (train.obs["cell_type"]==current_prediction_metadata["cell_type"]) & (train.obs["timepoint"]==current_prediction_metadata["timepoint"])
        goi, level, predict_steps = current_prediction_metadata[['perturbation', "expression_level_after_perturbation", 'predict_steps']]
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
                            "_num.steps_" f"{current_prediction_metadata['predict_steps']}" 
                            "_subsets_" f"{str(current_prediction_metadata['timepoint'])}_{current_prediction_metadata['cell_type']}"
                            "_perturb_simulation.pt"
                            )
        print(result["perturbed_sim"][0].shape)
        pca_predicted_embeddings = result["perturbed_sim"][0][time_points["predict_steps"],:,:].mean(axis=(0,1), keepdims = False).reshape(1, -1)
        print(pca_predicted_embeddings.shape)
        # Now we un-do all of PRESCIENT's preprocessing in order to return data on the scale of the training data input. 
        scaled_expression = pca_model.inverse_transform(pca_predicted_embeddings)
        for i in np.where(prediction_index)[0]:
            predictions[i, :].X = scaler.inverse_transform(scaled_expression) 
    except ValueError as e:
        predictions[prediction_index, :].X = np.nan
        print("Prediction failed. Error text: " + str(e))

    
print("Saving results.")
predictions.write_h5ad("from_to_docker/predictions.h5ad")