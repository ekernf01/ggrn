A Docker container to let GGRN interface to PRESCIENT. 

PRESCIENT is stochastic. By default, this will output 20 cells for each prediction you ask for, so the resulting AnnData will be 20 times bigger than you expect. You can change this by setting `kwargs["num_cells_to_simulate"]`.

Here is how the standard GGRN inputs are used. 

- `matching_method`: ignored (fixed to "optimal_transport").
- `do_perturbations_persist`: fixed to False
- `prediction_timescale`: **THIS IS WEIRD. SEE BELOW.**
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

For human data, growth rate estimation is done using the same gene-sets in the PRESCIENT Veres et al. (pancreas) example. 

For mouse data, we use:

- Birth: https://www.genome.jp/entry/pathway+mmu04110
- Death: https://www.genome.jp/entry/pathway+mmu04210

For zebrafish data, we use:

- Birth: https://www.genome.jp/entry/pathway/dre04110
- Death: https://www.genome.jp/entry/pathway/dre04210 (we manually removed some entries with no gene symbol)

## Timescales

Let's talk about prescient timescales. Internally, we use 3 (three) different timescales.
 
Input timepoints are transformed to be consecutive starting from 0 during process_data (flag --fix_non_consecutive).
So if training samples are labeled 0, 2, 12, 24, we would re-label them 0,1,2,3,4. 
So now we have two timescales, "original" and "consecutive".

Next, PRESCIENT is a differential equation method. Equations are solved approximately via time discretization. 
The [prescient docs](https://cgs.csail.mit.edu/prescient/documentation/) say there's a default dt of 0.1, so ten steps per time-point.
So if I want expression at time 2, I should feed 20 to prescient's `num_steps` arg.
So now we have three timescales, "original" and "consecutive" and "steps". 

Interpolation with PRESCIENT is also tricky, and I have not read the details yet. https://github.com/gifford-lab/prescient/issues/3

Guidance to the user: 

- the "timepoint" in the `obs` of `train.h5ad` and in `predictions_metadata` should be on the original scale e.g. 0, 2, 12, 24. 
- `prediction_timescale` in `predictions_metadata` should be on the "consecutive" scale, so a 2 would skip one time-step, bringing you from e.g. 0 to 12 or 2 to 24.