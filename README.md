## A grammar of inference methods for gene regulatory networks

GGRN is a set of Python modules that together allow users to flexibly combine traits of different methods for gene regulatory network inference, motivated by our [benchmarking project](https://github.com/ekernf01/perturbation_benchmarking). The theory behind this effort is outlined in [`GGRN.md`](https://github.com/ekernf01/ggrn/blob/main/GGRN.md), which contains the same content as supplemental file 1 from our [preprint](https://www.biorxiv.org/content/10.1101/2023.07.28.551039v1), or minor updates of it. A software reference fully cataloguing our implementation is in [`docs/reference.md`](https://github.com/ekernf01/ggrn/blob/main/docs/reference.md).

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

### Explanation and full API reference

See `docs`.
