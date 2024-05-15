A Docker container to let GGRN interface to PRESCIENT. Here is how the standard GGRN inputs are used. 

- `matching_method`: fixed to "optimal_transport"
- `do_perturbations_persist`: fixed to False
- `prediction_timescale`: this is given in the same units as the time-point labels in the training data. 
- `regression_method`: ignored 
- `eligible_regulators`: ignored
- `predict_self`: ignored (fixed to True)
- `feature_extraction`: ignored (fixed to "mRNA")
- `low_dimensional_structure`: ignored (fixed to "dynamics")
- `low_dimensional_training`: ignored (fixed to "PCA") 
- `pruning_strategy`: ignored
- `network`: ignored
- `network_prior`: fixed to "ignore"
- `cell_type_sharing_strategy`: ignored (fixed to "identical")
