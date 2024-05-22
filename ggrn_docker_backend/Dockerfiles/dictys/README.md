A Docker container used by GGRN to interface with dictys. Here is how the standard GGRN inputs are used. 

- `matching_method`: ignored (fixed to "steady_state")
- `do_perturbations_persist`: fixed to True
- `prediction_timescale`: Fixed to infinity -- in the dictys tutorials, only total effects are exposed to the user (dictys network indirect).
- `regression_method`: ignored (dictys uses a linear model) 
- `eligible_regulators`: ignored (dictys uses all TF's in the network)
- `predict_self`: fixed to False (analogous to the default value for thse `selfreg` arg to `dictys.chromatin.binlinking`)
- `feature_extraction`: ignored (fixed to "mRNA")
- `low_dimensional_structure`: ignored (fixed to "none")
- `low_dimensional_training`: ignored 
- `pruning_strategy`: ignored
- `network`: a networks is passed to dictys in lieu of dictys' bespoke motif footprinting pipeline. 
- `network_prior`: fixed to "restrictive"
- `cell_type_sharing_strategy`: fixed to "distinct"
