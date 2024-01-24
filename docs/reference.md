### GGRN software reference

Our implementation of the [grammar of gene regulatory networks](https://github.com/ekernf01/ggrn/blob/main/docs/GGRN.md) is used by constructing a GRN object, then calling its `fit` and `predict` methods. Most of the GGRN args are passed to `ggrn.api.GRN.fit()`, but `network` and `feature_extraction` are passed to the constructor `ggrn.api.GRN()`.

Many of the combinations that can be specified in GGRN either make no sense, or would be very complex to implement. The current software offers only subsets of features. These are implemented via several separate backends. 

##### Two idiosyncracies

- With apologies, the `method` arg of the `grn.fit` is currently used to specify both the backend AND, within backend #1, it is used to specify the regression method. In the future we hope to replace it with two separate args, `backend` and `regression_method`, but this may break backwards compatibility. Where `regression_method` is mentioned below, it currently corresponds to the `method` arg of `GRN.fit`.
- The `network` arg expects a LightNetwork object from our [network loader package](https://github.com/ekernf01/load_networks). If you have your own network data, you can make a LightNetwork object from a pandas dataframe or a parquet file.

#### Backend 1: steady state 

The initial implementation offers several matching methods; flexible regression; simple feature extraction or feature extraction via geneformer; rudimentary use of prior network structure; rudimentary cell type specificity; and predictions after just one step forward. It has no scheme for matching treatment samples to specific controls, no acyclic penalty, no low-dimensional structure, and no biological noise. It accepts the following inputs.

- `matching_method`: "steady_state", "user", "closest", or "random"
- `do_perturbations_persist`: ignored (fixed to True)
- `prediction_timescale`: ignored (fixed to 1)
- `regression_method`: A method from sklearn, e.g. `KernelRidge`
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
- `network`: Currently ignored, but in the future, we hope to include known networks to help construct the projection when "low_dimensional_training" is "user".
- `cell_type_sharing_strategy`: ignored (set to "identical")

For more information on this model, see [here](http://github.com/ekernf01/ggrn_backend3).

#### Backend 4: GEARS

GEARS is not well described by GGRN; it works from a different mathematical formulation. Nevertheless, we include an interface to GEARS in the GGRN software. GGRN args are ignored. GEARS arguments, such as embedding dimension or batch size, can be passed to `grn.fit` via `kwargs`. 

Some technical notes: 

- GEARS requires an internet connection (to download the GO graph). 
- Running GEARS will generate certain files and folders as a side effect; we save training data to a folder "ggrn_gears_input". This folder is normally deleted after use, but if a run crashes, it may need to be deleted. Otherwise, it can interfere with future runs. This can also create race conditions, and we do not support calling GEARS from multiple processes that share a working directory. 
- A GPU is typically required to use GEARS, but we find it is possible to use GEARS on a CPU. Depending on the version, this may require minor changes to GEARS source code: replace occurrences of `a += b` with `a = a + b` and `.to('cuda')` or `.to('cuda:0')` with `.to('cpu')`. 

#### Backend 5: networks alone

We offer another simple option that is not well described by GGRN. This backend is specified by setting `regression_method` to `"regulon"`. This backend requires a network structure, but no training data are needed. In this backend, the targets in the provided network will exactly follow the fold change of their regulators. For example, whenever log-scale FOXA1 expression goes up by 0.5 due to overexpression, FOXA1's targets in the provided network will go up by 0.5 also. 

#### Arbitrary backends via Docker:

You can add your own backend using Docker: consult `ggrn_docker_backend/README.md` in this repo.