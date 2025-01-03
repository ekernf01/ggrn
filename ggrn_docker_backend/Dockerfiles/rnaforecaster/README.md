A backend for ggrn using RNAForecasting as an inference engine.

### Interface 

We run RNAForecasting using a timescale similar to PRESCIENT: we feed in each data point matched to the previous timepoint during training, ignoring the actual time labels and acting as if they are consecutive integers. Then we interpret a prediction timescale of `t` to mean `simulate t timepoints ahead`. For example, if data are labeled 3 6 12 24 and you ask for a prediction timescale of `2` starting at timepoint `3`, you'll end up with predictions about time point `12`. 

Other GGRN args are used as follows.

- `matching_method`: all the usual options are allowed. "steady_state" is *not* recommended.
- `do_perturbations_persist`: ignored (fixed to True)
- `prediction_timescale`: See the note above
- `regression_method`: ignored 
- `eligible_regulators`: ignored (fixed to "all")
- `predict_self`: ignored (fixed to True)
- `feature_extraction`: ignored (fixed to "mRNA")
- `low_dimensional_structure`: ignored (fixed to "none")
- `low_dimensional_training`: ignored (fixed to "none") 
- `pruning_strategy`: ignored
- `network`: ignored
- `network_prior`: fixed to "ignore"
- `cell_type_sharing_strategy`: ignored (fixed to "identical")

Keyword args are not yet passed through to RNAForecaster -- sorry. If you need them, you could modify `train.jl` to read `from_to_docker/kwargs.json`.

### Implementation

Container is based on Ubuntu with Python and Julia installed. Julia version is a little old due to RNAForecaster deps. We convert the data format in Python.
