A Docker container with a simple baseline that ignores genetic perturbations and returns cells of the specified `cell_type` and `timepoint`, simulated forward by the desired number of steps (`prediction_timescale`). Simulations are run using the matching done by GGRN -- we take the specified cells, find all the matches, find the most abundant cell type, and return all cells of that type. We repeat this for `prediction_timescale` steps.

- `matching_method`: accepts all GGRN options, but using `steady_state` would kind of defeat the purpose.
- `do_perturbations_persist`: ignored; there are no perturbations. 
- `prediction_timescale`: see above
- `regression_method`: ignored
- `eligible_regulators`: ignored 
- `predict_self`: ignored 
- `feature_extraction`: ignored 
- `low_dimensional_structure`: ignored 
- `low_dimensional_training`: ignored 
- `pruning_strategy`: ignored 
- `pruning_parameter`: ignored
- `network`: ignored
- `network_prior`: ignored
- `cell_type_sharing_strategy`: ignored