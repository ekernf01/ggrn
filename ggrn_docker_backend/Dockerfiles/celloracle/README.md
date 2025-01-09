A Docker container to let GGRN interface to CellOracle. Here is how the standard GGRN inputs are used. 

- `matching_method`: ignored (fixed to "steady_state")
- `do_perturbations_persist`: ignored (fixed to True)
- `prediction_timescale`: this is mapped to "n_propagation" in "oracle.simulate_shift()".
- `regression_method`: ignored (CO uses Bayesian ridge regression)
- `eligible_regulators`: ignored (all TF's in the input network are used)
- `predict_self`: ignored (fixed to False)
- `feature_extraction`: ignored (fixed to "mRNA")
- `low_dimensional_structure`: ignored (fixed to "none")
- `low_dimensional_training`: ignored 
- `pruning_strategy`: ignored -- CellOracle alwasy does pruning.
- `pruning_parameter`: this is mapped to "threshold_number" in "links.filter_links"
- `network`: This is mapped to "TF_info_matrix" in "oracle.import_TF_data"
- `network_prior`: fixed to "restrictive"
- `cell_type_sharing_strategy`: ignored (set to "distinct")