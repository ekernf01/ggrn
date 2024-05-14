## A grammar of inference methods for gene regulatory networks

GGRN (Grammar of Gene Regulatory Networks) is a set of Python modules that together allow users to flexibly combine traits of different methods for gene regulatory network inference, motivated by our [benchmarking project](https://github.com/ekernf01/perturbation_benchmarking). The theory behind this effort is outlined in [`docs/GGRN.md`](https://github.com/ekernf01/ggrn/blob/main/docs/GGRN.md), which contains the same content as supplemental file 1 from our [preprint](https://www.biorxiv.org/content/10.1101/2023.07.28.551039v1), or minor updates of it. A software reference fully cataloguing our implementation is in [`docs/reference.md`](https://github.com/ekernf01/ggrn/blob/main/docs/reference.md).

### Installation

Just this package:

```bash
pip install ggrn @ git+https://github.com/ekernf01/ggrn
```

With our other related packages:

```bash
for p in pereggrn_networks pereggrn_perturbations pereggrn ggrn ggrn_backend2 geneformer_embeddings
do
    pip install ${p} @ http://github.com/ekernf01/${p}
done
```

### Usage tutorial

Here is a simple example of how to fit a LASSO model to predict each gene from the other genes. 

```python

import ggrn.api as ggrn
import pereggrn_perturbations
pereggrn_perturbations.set_data_path("../../perturbation_data/perturbations")
train = pereggrn_perturbations.load_perturbation("nakatake")

# Input: adata and (optional) sparse network structure
grn = ggrn.GRN(train, network = None, validate_immediately = False, eligible_regulators = train.var_names) 

# Training data must meet certain expectations: e.g. has to have a column "perturbation" specifying which gene is perturbed
# and a column "perturbation_type" that is "overexpression" or "knockdown" or "knockout".
# Use our validator to check your training data before proceeding.  
grn.check_perturbation_dataset() #Or, set validate_immediately = True above

# Specify a fitting method
grn.fit(
    method = "LassoCV",
    cell_type_sharing_strategy = "identical",
    network_prior = "ignore",
    matching_method = "steady_state",
    predict_self = False,   
    kwargs = {},    # Passed to the backend
)

# Consult help(GGRN.api.GRN.predict) for up-to-date docs -- this may get out of date more easily. 
# Specify what to predict by providing a dataframe with columns columns `cell_type`, `timepoint`, `perturbation_type`, `perturbation`, and `expression_level_after_perturbation`.
#
# `perturbation` and `expression_level_after_perturbation` can contain comma-separated strings for multi-gene perturbations, for 
example "NANOG,POU5F1" for `perturbation` and "5.43,0.0" for `expression_level_after_perturbation`. 
# Anything with expression_level_after_perturbation equal to np.nan will be treated as a control, no matter the name.
# If timepoint or celltype or perturbation_type are missing, the default is to copy them from the top row of self.train.obs.
# If the training data do not have a `cell_type` or `timepoint` column, then those *must* be omitted. Sorry; this is for backwards compatibility.
#
# An anndata.AnnData is returned with the provided dataframe in its .obs slot.
predictions = grn.predict(pd.DataFrame({
    "perturbation": ["POU5F1", "NANOG"],
     "expression_level_after_perturbation": [0,0],
    }))
```

### Explanation and full API reference

See `docs`.
