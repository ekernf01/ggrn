### A declarative grammar of inference methods for gene regulatory networks

GGRN is a set of Python modules that together allow users to flexibly combine traits of different inference methods for gene regulatory network inference. Check out the [writing repo](https://github.com/ekernf01/perturbation_writing) for more information.

#### Installation

WRITEME

#### Usage

```python

import ggrn.api as ggrn

# Input: adata and (optional) sparse network structure
grn = ggrn.GRN(train, network = None, validate_immediately = False) 

# Training data must meet certain expectations: e.g. has to have a column "perturbation" specifying which gene is perturbed
# and a column "perturbation_type" that is "overexpression" or "knockdown" or "knockout".
# Use our validator to check your training data before proceeding.  
grn.check_perturbation_dataset() #Or, set validate_immediately = True above

# Simple feature construction -- use the mRNA levels of the TF's as a proxy for activity
grn.extract_features(method = "tf_rna")

# Specify a fitting method
grn.fit(
    method = "linear", 
    cell_type_sharing = "distinct",    # or "identical" or "similar"
    network_prior = "restrictive",     # or "adaptive" or "informative"
    pruning_strategy = "none",         # or "prune_and_refit" or eventually maybe others
    pruning_parameter = None,          # e.g. lasso penalty or total number of nonzero coefficients
    projection = "factor_graph",       # or "PCA_before_training" or "PCA_after_training"
    kwargs = {},                       # Passed to the backend
)

# anndata.AnnData is returned
predictions = grn.predict([("POU5F1", 0), ("NANOG", 0)])
```

#### Adding a new backend

GGRN primarily serves as a wrapper for different backends that do the real work. For example, GEARS and DCD-FG are available as backends. You can add your own backend using Docker: see our [tutorial](https://github.com/ekernf01/ggrn_docker_backend).
