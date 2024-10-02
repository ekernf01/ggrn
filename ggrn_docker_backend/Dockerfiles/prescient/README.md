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

There is a long note in a comment in `train.py` source code about PRESCIENT timescales, which are kinda messy. Please read it lest you predict the wrong timepoints. 

For human data, growth rate estimation is done using the same gene-sets in the PRESCIENT Veres et al. (pancreas) example. 

For mouse data, we use:

- Birth: https://www.genome.jp/entry/pathway+mmu04110
- Death: https://www.genome.jp/entry/pathway+mmu04210

For zebrafish data, we use:

- Birth: https://www.genome.jp/entry/pathway/dre04110
- Death: https://www.genome.jp/entry/pathway/dre04210 (we manually removed some entries with no gene symbol)