## A grammar of inference methods for gene regulatory networks

GGRN is a set of Python modules that together allow users to flexibly combine traits of different methods for gene regulatory network inference. A full exposition is provided in [`GGRN.md`](https://github.com/ekernf01/ggrn/blob/main/GGRN.md), which is the raw source for the appendix from our  preprint (not up at time of writing). This is motivated by our [benchmarking project](https://github.com/ekernf01/perturbation_benchmarking).

### Installation

Just this package:

```bash
pip install ggrn @ git+https://github.com/ekernf01/ggrn
```

With our other related packages:

```bash
for p in load_networks load_perturbations ggrn_backend2 ggrn perturbation_benchmarking_package geneformer_embeddings
do
    pip install ${p} @ http://github.com/ekernf01/${p}
done
```

### Usage tutorial

Here is a simple example of how to fit a LASSO model to predict each gene from the other genes.

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
    network_prior = "ignore",
    matching_method = "steady_state",
    predict_self = False,   
    kwargs = {},    # Passed to the backend
)

# anndata.AnnData is returned
predictions = grn.predict([("POU5F1", 0), ("NANOG", 0)])
```

### Detailed reference

From the grammar set out in our [formal documentation](https://github.com/ekernf01/ggrn/blob/main/GGRN.md), many of the combinations that can be specified either make no sense, or would be very complex to implement. The current software offers only subsets of features. These are implemented via several separate backends. 

With apologies, the `method` arg of the `grn.fit` is currently used to select both the backend AND the regression method. In the future we hope to replace it with two separate args, `backend` and `regression_method`, but this may break backwards compatibility. Where `regression_method` is mentioned below, it currently corresponds to the `method` arg of `GRN.fit`.

#### Backend 1: steady state 

The initial implementation offers several matching methods; flexible regression; simple feature extraction or feature extraction via geneformer; rudimentary use of prior network structure; rudimentary cell type specificity; and predictions after just one step forward. It has no scheme for matching treatment samples to specific controls, no acyclic penalty, no low-dimensional structure, and no biological noise. It accepts the following inputs.

- `matching_method`: "steady_state", "user", "closest", or "random"
- `do_perturbations_persist`: ignored (fixed to True)
- `prediction_timescale`: ignored (fixed to 1)
- `method`: A method from sklearn, e.g. `KernelRidge`
- `eligible_regulators`: Any list of gene names
- `predict_self`: ignored (fixed to False)
- `feature_extraction`: "mrna" or "geneformer". With "geneformer", you cannot use prior networks.
- `low_dimensional_structure`: ignored (fixed to "none")
- `low_dimensional_training`: ignored
- `pruning_strategy`: "none", "prune_and_refit"
- `network`: An edge list containing regulators, targets, weights (optional), and cell types (optional).  
- `cell_type_sharing_strategy`: "identical", "distinct"

Backend #1 is the way GGRN can use GeneFormer. A few technical notes on our use of GeneFormer: 


- Following [issue 44](https://huggingface.co/ctheodoris/Geneformer/discussions/44) and [issue 53](https://huggingface.co/ctheodoris/Geneformer/discussions/53), we obtain post-perturbation cell embeddings from GeneFormer, then we train simple regression models to convert the post-perturbation cell embeddings to gene expression. 
- "Post-perturbation cell embeddings" could be constructed from training data in two ways. One option is to simply encode the expression data, since perturbation effects are already present in the expression data. A second option is to use GeneFormer's own tactic of altering the rank order to put overexpressed genes first and deleted genes last. Only the second option is available when making predictions, so we only use the second option during training.
- Because the GeneFormer-derived features are not interpretable as individual gene activities, it is not possible to constrain the list of eligible regulators in any way: `eligible_regulators`, `predict_self`, and `network` cannot be used.
- A GPU is required to use GeneFormer, but we find it is possible to use GeneFormer on a CPU if you replace occurrences of `a += b` with `a = a + b` and `.to('cuda')` or `.to('cuda:0')` with `.to('cpu')`. 
- Certain files are saved to disk as a necessary part of tokenizing data for GeneFormer. These are saved in "geneformer_loom_data" and "geneformer_tokenized_data". If a run crashes, these files may not be cleaned up, and they may need to be manually removed. This step could also cause race conditions when multiple processes are calling GGRN's GeneFormer feature extraction at the same time. 

#### Backend 2: DCD-FG 

We incorporate DCD-FG as a second backend for GGRN. We offer a thin wrapper with no functionality beyond the original DCD-FG implementation. Currently, DCD-FG args are provided in an ad-hod way: by setting the `method` arg to a hyphen-separated string specifying DCDFG, the type of acyclic constraint, the model, and whether to use polynomial features (example: `'DCDFG-spectral_radius-mlplr-false'`). In the future, we hope to use the standard GGRN arguments for everything except the acyclic penalties. This would allow the following options.

- `matching_method`: "steady_state"
- `do_perturbations_persist`: ignored (fixed to True) 
- `prediction_timescale`: ignored (fixed to 1)
- `regression_method`: "mlplr" or "linear" or "poly"
- `eligible_regulators`: ignored (all genes are used)
- `predict_self`: ignored (fixed to False)
- `feature_extraction`: ignored (fixed to "mRNA")
- `low_dimensional_structure`: "none", "dynamics"
- `low_dimensional_training`: ignored (fixed to "supervised")
- `pruning_strategy`: ignored (DCD-FG's strategy is used)
- `network`: ignored 
- `cell_type_sharing_strategy`: ignored (fixed to "identical").

For more information on our pip-installable interface to DCD-FG, see [here](https://github.com/ekernf01/ggrn_backend2). 

#### Backend 3: autoregressive with matching

As the third backend of GGRN, we implement an autoregressive model that explicitly uses the passage of time and does not make steady-state assumptions. This offers the following features.

- `matching_method`: "closest", "user", "random"
- `do_perturbations_persist`: boolean
- `prediction_timescale`: positive int. (Optimization becomes difficult rapidly; we recommend setting this to 1.) 
- `regression_method`: ignored (fixed to "linear")
- `eligible_regulators`: ignored (all genes are used)
- `predict_self`: ignored (fixed to True)
- `feature_extraction`: ignored (fixed to "mRNA")
- `low_dimensional_structure`: "none", "dynamics"
- `low_dimensional_training`: "user", "svd", "supervised"
- `pruning_strategy`: ignored (fixed to "none")
- `network`: Currently ignored, but we hope to include this to construct the projection when "low_dimensional_training" is "user".
- `cell_type_sharing_strategy`: ignored (set to "identical")

For more information on this model, see [here](http://github.com/ekernf01/ggrn_backend3).

#### Backend 4: GEARS

GEARS is not well described by GGRN; it works from a different mathematical formulation. Nevertheless, we include an interface to GEARS in the GGRN software. GGRN args are ignored. GEARS arguments, such as embedding dimension or batch size, can be passed to `grn.fit` via `kwargs`. 

Some technical notes: 

- GEARS requires an internet connection (to download the GO graph). 
- Running GEARS will generate certain files and folders as a side effect; we save training data to a folder "ggrn_gears_input". This folder is normally deleted after use, but if a run crashes, it may need to be deleted. Otherwise, it can interfere with future runs. This can also create race conditions, and we do not support calling GEARS from multiple processes that share a working directory. 
- A GPU is typically required to use GEARS, but we find it is possible to use GEARS on a CPU. Depending on the version, this may require minor changes to GEARS source code: replace occurrences of `a += b` with `a = a + b` and `.to('cuda')` or `.to('cuda:0')` with `.to('cpu')`. 

#### Arbitrary backends via Docker:

You can add your own backend using Docker: consult `ggrn_docker_backend/README.md` in this repo.