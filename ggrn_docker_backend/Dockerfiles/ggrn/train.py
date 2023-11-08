import os
import json
import scanpy as sc
import ggrn.api as ggrn
train = sc.read_h5ad("from_to_docker/train.h5ad")
with open(os.path.join("from_to_docker/perturbations.json")) as f:
    perturbations = json.load(f)
with open(os.path.join("from_to_docker/kwargs.json")) as f:
    kwargs = json.load(f)
with open(os.path.join("from_to_docker/ggrn_args.json")) as f:
    ggrn_args = json.load(f)
train.obs["is_control"] = train.obs["is_control"].astype(bool)
grn = ggrn.GRN(
    train                = train, 
    network              = None,
    eligible_regulators  = ggrn_args["eligible_regulators"],
    feature_extraction   = ggrn_args["feature_extraction"],
    validate_immediately = True
) 
grn.fit(
    method                               = ggrn_args["regression_method"], 
    cell_type_labels                     = None,
    cell_type_sharing_strategy           = "identical",
    network_prior                        = ggrn_args["network_prior"],
    pruning_strategy                     = ggrn_args["pruning_strategy"],
    pruning_parameter                    = ggrn_args["pruning_parameter"],
    predict_self                         = ggrn_args["predict_self"],
    matching_method                      = ggrn_args["matching_method"],
    low_dimensional_structure            = ggrn_args["low_dimensional_structure"],
    low_dimensional_training             = ggrn_args["low_dimensional_training"],
    prediction_timescale                 = ggrn_args["prediction_timescale"],
    kwargs                               = kwargs,
)
predictions   = grn.predict(
    perturbations,
    feature_extraction_requires_raw_data = (grn.feature_extraction == "geneformer"),
)
predictions.write_h5ad("from_to_docker/predictions.h5ad")