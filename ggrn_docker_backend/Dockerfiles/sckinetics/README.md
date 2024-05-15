A Docker container used by GGRN to interface with scKINETICS. Here is how the standard GGRN inputs are used. 

- `matching_method`: ignored (scKINETICS has a customized method based on nearest neighbors)
- `do_perturbations_persist`: fixed to True
- `prediction_timescale`: Any positive integer. this determines how many times a velocity vectors is predicted and the cell state is updated. 
- `regression_method`: ignored (scKINETICS uses a linear model) 
- `eligible_regulators`: ignored (scKINETICS uses all TF's in the network)
- `predict_self`: ignored (fixed to True)
- `feature_extraction`: ignored (fixed to "mRNA")
- `low_dimensional_structure`: ignored (fixed to "none")
- `low_dimensional_training`: ignored 
- `pruning_strategy`: ignored
- `network`: passed to the "G" arg in "EM.ExpectationMaximization.fit"
- `network_prior`: fixed to "restrictive"
- `cell_type_sharing_strategy`: ignored (fixed to "distinct")
