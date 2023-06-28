## A declarative grammar of inference methods for gene regulatory networks

GGRN is a set of Python modules that together allow users to flexibly combine traits of different inference methods for gene regulatory network inference. A full exposition is provided in the appendix of our preprint (not up yet). 

### Installation

```bash
pip install git+https://github.com/ekernf01/ggrn
```

### Usage

```python

import ggrn.api as ggrn

# Input: adata and (optional) sparse network structure
grn = ggrn.GRN(train, network = None, validate_immediately = False, eligible_regulators = train.var_names) 

# Training data must meet certain expectations: e.g. has to have a column "perturbation" specifying which gene is perturbed
# and a column "perturbation_type" that is "overexpression" or "knockdown" or "knockout".
# Use our validator to check your training data before proceeding.  
grn.check_perturbation_dataset() #Or, set validate_immediately = True above

# Simple feature construction -- use the mRNA levels of the TF's as a proxy for activity
# We may add more feature construction options in the future.
grn.extract_features(method = "tf_rna")

# Specify a fitting method
grn.fit(
    method = "LassoCV",
    cell_type_sharing_strategy = "identical",
    network_prior = "restrictive",
    matching_method = "steady_state",
    predict_self = False,   
    kwargs = {},    # Passed to the backend
)

# anndata.AnnData is returned
predictions = grn.predict([("POU5F1", 0), ("NANOG", 0)])
```

### Detailed documentation

From the grammar set out in our technical documentation, many of the combinations that can be specified either make no sense, or would be very complex to implement. The current software offers only subsets of features. These are implemented via several separate backends. 

With apologies, the `method` arg of the `grn.fit` is currently used to select both the backend AND the regression method. In the future we hope to replace it with two separate args, `backend` and `regression_method`, but this may break backwards compatibility. Where `regression_method` is mentioned below, it currently corresponds to the `method` arg of `GRN.fit`.

#### Backend 1: steady state 

The initial implementation requires steady-state matching; it ignores the passage of time. It offers flexible regression, rudimentary use of prior network structure, rudimentary cell type specificity, and predictions after just one step forward. It has no scheme for matching treatment samples to specific controls, no acyclic penalty, no low-dimensional structure, and no biological noise. It accepts the following inputs.

- `matching_method`: "steady_state"
- `do_perturbations_persist`: ignored (fixed to True)
- `prediction_timescale`: ignored (fixed to 1)
- `method`: A method from sklearn, e.g. `KernelRidge`
- `eligible_regulators`: Any list of gene names
- `feature_extraction`: ignored (fixed to "mRNA")
- `low_dimensional_structure`: ignored (fixed to "none")
- `low_dimensional_training`: ignored
- `pruning_strategy`: "none", "prune_and_refit"
- `network`: An edge list containing regulators, targets, weights (optional), and cell types (optional).  
- `cell_type_sharing_strategy`: "identical", "distinct"

#### Backend 2: DCD-FG 

We incorporate DCD-FG as a second backend for GGRN. We offer a thin wrapper with no functionality beyond the original DCD-FG implementation. Currently, DCD-FG args are provided in an ad-hod way: by setting the `method` arg to a hyphen-separated string specifying DCDFG, the type of acyclic constraint, the model, and whether to use polynomial features (example: `'DCDFG-spectral_radius-mlplr-false'`). In the future, we hope to use the standard GGRN arguments for everything except the acyclic penalties. This would allow the following options.

- `matching_method`: "steady_state"
- `do_perturbations_persist`: ignored (fixed to True) 
- `prediction_timescale`: ignored (fixed to 1)
- `regression_method`: "mlplr" or "linear" or "poly"
- `eligible_regulators`: ignored (all genes are used)
- `feature_extraction`: ignored (fixed to "mRNA")
- `low_dimensional_structure`: "none", "dynamics"
- `low_dimensional_training`: ignored (fixed to "supervised")
- `pruning_strategy`: ignored (DCD-FG's strategy is used)
- `network`: ignored 
- `cell_type_sharing_strategy`: ignored (fixed to "identical").


#### Backend 3: autoregressive with matching

As the third backend of GGRN, we implement an autoregressive model that explicitly uses the passage of time and does not make steady-state assumptions. This offers the following features.

- `matching_method`: "closest", "user", "random"
- `do_perturbations_persist`: boolean
- `prediction_timescale`: positive int
- `regression_method`: ignored (fixed to "linear")
- `eligible_regulators`: ignored (all genes are used)
- `feature_extraction`: ignored (fixed to "mRNA")
- `low_dimensional_structure`: "none", "dynamics"
- `low_dimensional_training`: "user", "svd", "supervised"
- `pruning_strategy`: ignored (fixed to "none")
- `network`: Currently ignored, but we hope to include this to construct the projection when "low_dimensional_training" is "user".
- `cell_type_sharing_strategy`: ignored (set to "identical")

For more information on this model, see [here](http://github.com/ekernf01/ggrn_backend3).

#### Backend 4: GEARS

GEARS is not well described by GGRN; it works from a different mathematical formulation. Nevertheless, we include an interface to GEARS in the GGRN software. The `eligible_regulators` arg behaves as expected and other GEARS arguments can be passed via `kwargs`. A GPU is typically required to use GEARS, but we find it is possible to use on a CPU with minor changes to GEARS source code: replace occurrences of `a += b` with `a = a + b` and `.to('cuda')` or `.to('cuda:0')` with `.to('cpu')`. 

#### Backend 5: GeneFormer

GeneFormer is not well described by GGRN; it works from a different mathematical formulation. Nevertheless, we include an interface to GeneFormer in the GGRN software. As recommended by the authors in [issue 44](https://huggingface.co/ctheodoris/Geneformer/discussions/44), we obtain post-perturbation cell embeddings from GeneFormer, then we train a simple model to convert the post-perturbation cell embeddings to log fold changes. Arguments to GeneFormer's functions for cell perturbations can be passed via `kwargs`. A GPU is required to use GeneFormer, but we find it is possible to use on a CPU if you replace occurrences of `a += b` with `a = a + b` and `.to('cuda')` or `.to('cuda:0')` with `.to('cpu')`. 

#### Adding a new backend

You can add your own backend using Docker: see our [tutorial](https://github.com/ekernf01/ggrn_docker_backend).
