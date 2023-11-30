import anndata
from joblib import Parallel, delayed, cpu_count, dump
import pandas as pd
import scanpy as sc
import os, shutil
import sklearn.linear_model
import sklearn.ensemble
import sklearn.neural_network
import sklearn.kernel_ridge
from sklearn.neighbors import KDTree
import sklearn.dummy
import numpy as np
import scipy.sparse
import json
import gc
import ray.tune.error
from torch.cuda import is_available as is_gpu_available
try:
    import ggrn_backend2.api as dcdfg_wrapper 
    HAS_DCDFG = True 
except ImportError:
    HAS_DCDFG = False 
try:
    from gears import PertData as GEARSPertData
    from gears import GEARS
    HAS_GEARS = True
except ImportError:
    HAS_GEARS = False
try:
    import ggrn_backend3.api as autoregressive
    HAS_AUTOREGRESSIVE = True
except ImportError:
    HAS_AUTOREGRESSIVE = False
try:
    from geneformer_embeddings import geneformer_embeddings, geneformer_hyperparameter_optimization
    HAS_GENEFORMER = True
except ImportError:
    HAS_GENEFORMER = False

# These govern expectations about data format for regulatory networks and perturbation data.
try:
    import load_networks
    HAS_NETWORKS = True
except:
    HAS_NETWORKS = False
try:
    import load_perturbations
    CAN_VALIDATE = True
except:
    CAN_VALIDATE = False

class GRN:
    """
    Flexible inference of gene regulatory network models.
    """
    def __init__(
        self, train: anndata.AnnData, 
        network = None, 
        eligible_regulators: list = None,
        validate_immediately: bool = True,
        feature_extraction: str = "mrna",
    ):
        """Create a GRN object.

        Args:
            train (anndata.AnnData): Training data. Should conform to the requirements in load_perturbations.check_perturbation_dataset().
            network (load_networks.LightNetwork, optional): LightNetwork object containing prior knowledge about regulators and targets.
            eligible_regulators (List, optional): List of gene names that are allowed to be regulators. Defaults to all genes.
            validate_immediately (bool, optional): check validity input object?
            feature_extraction (str, optional): How to extract features. Defaults to "tf_rna", which uses the mRNA level of the TF. You 
                can also specify "geneformer" to use GeneFormer to extract cell embeddings. 

        """
        self.train = train
        try:
            self.train.X = self.train.X.toarray()
        except AttributeError:
            pass
        assert network is None or type(network)==load_networks.LightNetwork, "Network must be of type LightNetwork (see our load_networks package)."
        self.network = network 
        self.models = [None for _ in self.train.var_names]
        self.training_args = {}
        self.features = None
        self.geneformer_finetuned = None
        self.feature_extraction = "mrna" if (feature_extraction is None or isnan_safe(feature_extraction)) else feature_extraction
        self.eligible_regulators = eligible_regulators
        if validate_immediately:
            assert self.check_perturbation_dataset()
        if self.feature_extraction.lower().startswith("geneformer"):
            assert self.eligible_regulators is None, "You may not constrain the set of regulators when using GeneFormer feature extraction."
        return

    def fit(
        self,
        method: str, 
        confounders: list = [], 
        cell_type_sharing_strategy: str = "identical",   
        cell_type_labels: str = None,
        network_prior: str = None,    
        pruning_strategy: str = "none", 
        pruning_parameter: str = None,          
        matching_method: str = "steady_state",
        feature_extraction: str = "mrna",
        low_dimensional_structure: str = None,
        low_dimensional_training: str = None,
        prediction_timescale: int = 1, 
        predict_self: str = False,   
        do_parallel: bool = True,
        kwargs = dict(),
    ):
        """Fit the model.

        Args:
            method (str): Regression method to use. 
            confounders (list): Not implemented yet.
            cell_type_sharing_strategy (str, optional): Whether to fit one model across all training data ('identical') 
                or fit separate ones ('distinct'). Defaults to "distinct".
            cell_type_labels (str): Name of column in self.train.obs to use for cell type labels.
            eligible_regulators (List, optional): List of gene names that are allowed to be regulators. Defaults to all genes.
            network_prior (str, optional): How to incorporate user-provided network structure. 
                - "ignore": don't use it. 
                - "restrictive": allow only user-specified regulators for each target.
                - Default is "ignore" if self.network is None else "restrictive"
            pruning_strategy (str, optional) 
                - "prune_and_refit": keep the largest n coefficients across all models (not per model), where n is pruning_parameter.
                - "none": don't prune the model.
                - maybe more options will be implemented. 
            pruning_parameter (numeric, optional): e.g. lasso penalty or total number of nonzero coefficients. See "pruning_strategy" for details.
            matching_method: (str, optional): "closest" (choose the closest control), or
                "user" (do nothing and expect an existing column 'matched_control'), or 
                "random" (choose a random control), or 
                "steady_state" (match each observation with itself).
            regression_method (str): Currently only allows "linear".
            feature_extraction (str): Method of feature extraction. "mrna" or "geneformer". 
            low_dimensional_structure (str) "none" or "dynamics" or "RGQ". See also low_dimensional_training.
                - If "none", dynamics will be modeled using the original data.
                - If "dynamics", dynamics will be modeled in a latent space. 
                - "RGQ" is a deprecated option identical to "dynamics".
            low_dimensional_training (str): "SVD" or "fixed" or "supervised". How to learn the linear subspace. 
                If "SVD", perform an SVD on the data.
                If "fixed", use a user-provided projection matrix.
                If "supervised", learn the projection and its (approximate) inverse via backprop.
            prediction_timescale (int, optional): For time-series models, how many steps separate each observation from its paired control.
            predict_self (bool, optional): Should e.g. POU5F1 activity be used to predict POU5F1 expression? Defaults to False.
            test_set_genes: The genes that are perturbed in the test data. GEARS can use this extra info during training.
            kwargs: Passed to DCDFG or GEARS. See help(dcdfg_wrapper.DCDFGWrapper.train). 
        """
        if self.feature_extraction.lower().startswith("geneformer"):            
            assert predict_self, "GGRN cannot exclude autoregulation when using geneformer for feature extraction."
            assert prediction_timescale == 1, "GGRN cannot do iterative prediction when using geneformer for feature extraction."
        if network_prior is None:
            network_prior = "ignore" if self.network is None else "restrictive"
        if network_prior != "ignore":
            assert self.network is not None, "You must provide a network unless network_prior is 'ignore'."
            assert feature_extraction.lower() in {"tf_mrna", "mrna", "rna", "tf_rna"},  "Only simple feature extraction is compatible with prior network structure."
        self.train = match_controls(self.train, matching_method, matched_control_is_integer=False) 
        if self.features is None:
            self.extract_features(train = self.train, in_place = True)
        self.validate_method(method)
        # Save some training args to later ensure prediction is consistent with training.
        self.training_args["confounders"]                = confounders
        self.training_args["network_prior"]              = network_prior
        self.training_args["cell_type_sharing_strategy"] = cell_type_sharing_strategy
        self.training_args["cell_type_labels"]           = cell_type_labels
        self.training_args["predict_self"]               = predict_self
        self.training_args["feature_extraction"]         = feature_extraction
        self.training_args["method"]                     = method
        self.training_args["matching_method"]            = matching_method
        # Check that the cell types match between the network and the expression data
        if network_prior != "ignore" and self.training_args["cell_type_sharing_strategy"] != 'identical':
            ct_from_network = self.network.get_all_one_field("cell_type")
            ct_from_trainset = self.train.obs[cell_type_labels]
            ct_missing = [ct for ct in ct_from_trainset if ct not in ct_from_network]
            prettyprint = lambda x: '\n'.join(x)
            if len(ct_from_network) > 0:
                if len(ct_missing)>0:
                    raise ValueError(
                        "Some cell types in the training data are not in the networks."
                        "Trainset: \n" f"{prettyprint(ct_from_trainset)}"
                        "Networks: \n" f"{prettyprint(ct_from_network)}"
                        "Missing:  \n" f"{prettyprint(ct_missing)}"
                    )
        if method.startswith("autoregressive"):
            if not HAS_AUTOREGRESSIVE:
                raise Exception("autoregressive backend is not installed, and related models will not be available.")
            np.testing.assert_equal(
                np.array(self.eligible_regulators), 
                np.array(self.train.var_names),     
                err_msg="autoregressive models can only use all genes as regulators."
            )
            assert len(confounders)==0, "autoregressive models cannot currently include confounders."
            assert cell_type_sharing_strategy=="identical", "autoregressive models cannot currently fit each cell type separately."
            assert predict_self, "autoregressive models currently cannot exclude autoregulation. Set predict_self=True to continue."
            # Default to a 20-dimensional latent space learned from data.
            # But if given an informative prior network, reformat it for use as projection to latent space.
            if network_prior=="ignore":
                low_dimensional_value = 20
            else:
                low_dimensional_value = scipy.sparse.coo_matrix(
                    (
                        len(self.network.get_all_regulators()), 
                        self.train.n_vars 
                    )
                )
                for i, target in enumerate(self.network.get_all_one_field("target")):
                    low_dimensional_value[self.train.var_names.index(self.network.get_regulators(target)), i] = 1
            autoregressive_model = autoregressive.GGRNAutoregressiveModel(
                self.train, 
                matching_method = "user", #because it's taken care of above
                low_dimensional_structure = low_dimensional_structure,
                low_dimensional_training = low_dimensional_training,
                low_dimensional_value = low_dimensional_value, 
                S = prediction_timescale
            )
            autoregressive_model.train()      
            self.models = autoregressive_model
        elif method.startswith("SCFormer"):
            raise NotImplementedError("Sorry, the SCFormer backend is still in progress.")
        elif method.startswith("GeneFormerNSP"):
            raise NotImplementedError("Sorry, the GeneFormerNSP backend is still in progress.")
            # FIT:
            #     Tokenize controls and perturb tokenized representation
            #     Tokenize post-perturbation embeddings
            #     Get the GeneFormer network
            #     Feed'em all to a BERT NSP fine tuning task
            #     Train a decoder of some sort to get expression from tokens
            # PREDICT:
            #     run next sentence prediction on test data
            #     decode into expression
        elif method.startswith("docker"):
            # There is no actual training here, just copying data. 
            # Training happens when you call predict.
            # This is because we do not want to try to save trained models inside a container and return control flow to Python while leaving the container intact.
            try:
                shutil.rmtree("from_to_docker")
            except:
                pass
            os.makedirs("from_to_docker", exist_ok=True)
            ggrn_args = {
                "regression_method": method,
                "confounders": list(confounders),
                "cell_type_sharing_strategy": cell_type_sharing_strategy,
                "cell_type_labels": cell_type_labels, 
                "network_prior": network_prior,
                "pruning_strategy": pruning_strategy,
                "pruning_parameter": pruning_parameter,   
                "matching_method": matching_method,
                "low_dimensional_structure":  low_dimensional_structure,
                "low_dimensional_training": low_dimensional_training,
                "prediction_timescale": int(prediction_timescale),
                "predict_self": bool(predict_self), 
            }
            with open("from_to_docker/ggrn_args.json", "x") as f:
                json.dump(ggrn_args, f)
            with open("from_to_docker/kwargs.json", "x") as f:
                json.dump(kwargs, f)
        elif method.startswith("GEARS"):
            if not HAS_GEARS:
                raise ImportError("GEARS is not installed, and related models will not be available.")
            assert len(confounders)==0, "Our interface to GEARS cannot currently include confounders."
            assert network_prior=="ignore", "Our interface to GEARS cannot currently include custom networks."
            assert cell_type_sharing_strategy=="identical", "Our interface to GEARS cannot currently fit each cell type separately."
            assert predict_self, "Our interface to GEARS cannot rule out autoregulation. Set predict_self=True."
            # Data setup according to GEARS data tutorial:
            # https://github.com/snap-stanford/GEARS/blob/master/demo/data_tutorial.ipynb 
            def reformat_perturbation_for_gears(perturbation):
                perturbation = perturbation.split(",")
                perturbation = ["ctrl"] + [p if p in self.train.var.index else "ctrl" for p in perturbation]
                perturbation = list(set(perturbation)) # avoid ctrl+ctrl
                perturbation = "+".join(perturbation)
                return perturbation
            self.train.var['gene_name'] = self.train.var.index
            self.train.obs['condition'] = [reformat_perturbation_for_gears(p) for p in self.train.obs['perturbation']]
            self.train.obs['cell_type'] = "all"
            # GEARS has certain input reqs:
            # - expects sparse matrix for X
            # - needs the base of the log-transform to be explicit
            # - needs to write to hdf5, so no sets in .uns and no Categorical in .obs
            # - cannot handle perturbations with just one sample
            for k in self.train.obs.columns:
                self.train.obs[k] = self.train.obs[k].astype("str") 
            self.train.uns["log1p"] = dict()
            self.train.uns["log1p"]["base"] = np.exp(1)
            self.train.uns["perturbed_and_measured_genes"] = list(self.train.uns["perturbed_and_measured_genes"])
            self.train.uns["perturbed_but_not_measured_genes"] = list(self.train.uns["perturbed_but_not_measured_genes"])
            self.train.X = scipy.sparse.csr_matrix(self.train.X)
            pert_frequencies = self.train.obs['condition'].value_counts()
            singleton_perts = pert_frequencies[pert_frequencies==1].index
            print(f"Marking {len(singleton_perts)} singleton perturbations as controls in the training data.", flush=True)
            self.train.obs.loc[self.train.obs.condition.isin(singleton_perts), "condition"] = "ctrl"
            # Clean up any previous runs
            try:
                shutil.rmtree("./ggrn_gears_input")
            except FileNotFoundError:
                pass
            try: 
                os.remove("data/go_essential_current.csv")
            except FileNotFoundError:
                pass
            
            # Set training params to kwargs or defaults
            defaults = {
                "seed": 1,
                "batch_size": 32,
                "test_batch_size": 128,
                "hidden_size": 64,
                "device": "cuda" if is_gpu_available() else "cpu",
                "epochs": 15,
            }
            if not kwargs:
                kwargs = dict()
            for k in kwargs:
                if k not in defaults:
                    raise KeyError(f"Unexpected keyword arg passed to GEARS wrapper. Expected: {'  '.join(defaults.keys())}")
            for k in defaults:
                if k not in kwargs:
                    kwargs[k] = defaults[k]
            # Follow GEARS data setup tutorial
            pert_data = GEARSPertData("./ggrn_gears_input", default_pert_graph = True)
            pert_data.new_data_process(dataset_name = 'current', adata = self.train)
            pert_data.load(data_path = './ggrn_gears_input/current')
            # For the data split, we use train_gene_set_size = 0.95 instead of the default.
            # This is not documented; however, it has the effect of using more of the input for train and
            # validation instead of the test set. A test set has already been reserved by our 
            # benchmarking framework in a typical use of this code. 
            pert_data.prepare_split(split = 'no_test', seed = kwargs["seed"] )
            pert_data.get_dataloader(batch_size = kwargs["batch_size"], test_batch_size = kwargs["test_batch_size"])
            self.models = GEARS(pert_data, device = kwargs["device"])
            self.models.model_initialize(hidden_size = kwargs["hidden_size"])
            self.models.train(epochs = kwargs["epochs"])
        elif method.startswith("DCDFG"):
            assert HAS_DCDFG, "DCD-FG wrapper is not installed, and related models (DCD-FG, NOTEARS) cannot be used."
            np.testing.assert_equal(
                np.array(self.eligible_regulators), 
                np.array(self.train.var_names),     
                err_msg="DCDFG can only use all genes as regulators."
            )
            assert len(confounders)==0, "DCDFG cannot currently include confounders."
            assert network_prior=="ignore", "DCDFG cannot currently include known network structure."
            assert cell_type_sharing_strategy=="identical", "DCDFG cannot currently fit each cell type separately."
            assert not predict_self, "DCDFG cannot include autoregulation."
            factor_graph_model = dcdfg_wrapper.DCDFGWrapper()
            try:
                _, constraint_mode, model_type, do_use_polynomials = method.split("-")
            except:
                raise ValueError("DCDFG expects hyphen-separated string DCDFG-constraint-model-do_polynomials like 'DCDFG-spectral_radius-mlplr-false'.")
            print(f"""DCDFG args parsed as:
               constraint_mode: {constraint_mode}
                    model_type: {model_type}
            do_use_polynomials: {do_use_polynomials}
            """)
            do_use_polynomials = do_use_polynomials =="True"
            self.models = factor_graph_model.train(
                self.train,
                constraint_mode = constraint_mode,
                model_type = model_type,
                do_use_polynomials = do_use_polynomials,
                **kwargs
            )
        else:     
            assert len(confounders)==0, "sklearn models cannot currently include confounders."
            if method == "mean":
                def FUN(X,y):
                    return sklearn.dummy.DummyRegressor(strategy="mean", **kwargs).fit(X, y)
            elif method == "median":
                def FUN(X,y):
                    return sklearn.dummy.DummyRegressor(strategy="median", **kwargs).fit(X, y)
            elif method == "GradientBoostingRegressor":
                def FUN(X,y):
                    return sklearn.ensemble.GradientBoostingRegressor(**kwargs).fit(X, y)
            elif method == "ExtraTreesRegressor":
                def FUN(X,y):
                    et = sklearn.ensemble.ExtraTreesRegressor(n_jobs=1, **kwargs).fit(X, y)
                    return et
            elif method == "KernelRidge":
                def FUN(X,y):
                    return sklearn.kernel_ridge.KernelRidge(**kwargs).fit(X, y)
            elif method == "ElasticNetCV":
                def FUN(X,y):
                    return sklearn.linear_model.ElasticNetCV(
                        fit_intercept=True, 
                        **kwargs,
                    ).fit(X, y)
            elif method == "LarsCV":
                def FUN(X,y):
                    return sklearn.linear_model.LarsCV(
                        fit_intercept=True,
                        **kwargs,
                    ).fit(X, y)
            elif method == "OrthogonalMatchingPursuitCV":
                def FUN(X,y):
                    try:
                        omp = sklearn.linear_model.OrthogonalMatchingPursuitCV(
                            fit_intercept=True, **kwargs
                        ).fit(X, y)
                    except ValueError: #may fail with ValueError: attempt to get argmin of an empty sequence
                        omp = sklearn.dummy.DummyRegressor(strategy="mean").fit(X, y)
                    return omp
            elif method == "ARDRegression":
                def FUN(X,y):
                    return sklearn.linear_model.ARDRegression(
                        fit_intercept=True, **kwargs,
                    ).fit(X, y)
            elif method == "BayesianRidge":
                def FUN(X,y):
                    br = sklearn.linear_model.BayesianRidge(
                        fit_intercept=True, copy_X = True, **kwargs,
                    ).fit(X, y)
                    del br.sigma_ # Default behavior is to return a full PxP covariance matrix for the coefs -- it's too big.
                    return br
            elif method.lower() == "lasso":
                def FUN(X,y):
                    return sklearn.linear_model.Lasso(
                        fit_intercept=True, **kwargs,
                    ).fit(X, y)
            elif method == "LassoCV":
                def FUN(X,y):
                    return sklearn.linear_model.LassoCV(
                        fit_intercept=True, **kwargs,
                    ).fit(X, y)
            elif method == "LassoLarsIC":
                def FUN(X,y):
                    return sklearn.linear_model.LassoLarsIC(
                        fit_intercept=True, **kwargs,
                    ).fit(X, y)
            elif method == "RidgeCV":
                def FUN(X,y):
                    return sklearn.linear_model.RidgeCV(
                        alphas=(0.01, 0.1, 1.0, 10.0, 100), 
                        fit_intercept=True,
                        alpha_per_target=False, 
                        store_cv_values=True, #this lets us use ._cv_values later for simulating data.
                        **kwargs,
                    ).fit(X, y)
            elif method == "RidgeCVExtraPenalty":
                def FUN(X,y):
                    rcv = sklearn.linear_model.RidgeCV(
                        alphas=(0.01, 0.1, 1.0, 10.0, 100), 
                        fit_intercept=True,
                        alpha_per_target=False, 
                        store_cv_values=True, #this lets us use ._cv_values later for simulating data.
                        **kwargs,
                    ).fit(X, y)
                    if rcv.alpha_ == np.max(rcv.alphas):
                        bigger_alphas = rcv.alpha_ * np.array([0.1, 1, 10, 100, 1000, 10000, 100000])
                        rcv = sklearn.linear_model.RidgeCV(
                            alphas=bigger_alphas, 
                            fit_intercept=True,
                            alpha_per_target=False, 
                            store_cv_values=True,
                             **kwargs,
                        ).fit(X, y)
                    return rcv
            else:
                raise NotImplementedError(f"Method {method} is not supported.")
            self.fit_with_or_without_pruning(
                    FUN, 
                    network_prior=network_prior, 
                    pruning_strategy=pruning_strategy, 
                    pruning_parameter=pruning_parameter,
                    do_parallel=do_parallel,
            )
        return

    def predict(
        self,
        perturbations: list,
        predictions: anndata.AnnData = None,
        control_subtype = None,
        do_parallel: bool = True,
        add_noise: bool = False,
        noise_sd: float = None,
        seed: int = 0,
        prediction_timescale: int = 1,
        feature_extraction_requires_raw_data: bool = False
    ):
        """Predict expression after new perturbations.

        Args:
            perturbations (iterable of tuples): Iterable of tuples with gene and its expression after 
                perturbation, e.g. {("POU5F1", 0.0), ("NANOG", 0.0), ("non_targeting", np.nan)}. Anything with
                expression np.nan will be treated as a control, no matter the name.
            predictions (anndata.AnnData): Expression prior to perturbation, in the same shape as the output predictions. If 
                None, starting state will be set to the mean of the training data control expression values.
            control_subtype (str): only controls with this prefix are considered. For example, 
                in the Nakatake data there are different types of controls labeled "emerald" and "rtTA".
            do_parallel (bool): if True, use joblib parallelization. 
            add_noise (bool): if True, return simulated data Y + e instead of predictions Y (measurement error)
                where e is IID Gaussian with variance equal to the estimated residual variance.
            noise_sd (bool): sd of the variable e described above. Defaults to estimates from the fitted models.
            seed (int): RNG seed.
            prediction_timescale (int): how many time-steps forward to predict.
            feature_extraction_requires_raw_data (bool): We recommend to set to True with GeneFormer and False otherwise.
        """
        # Check inputs
        if type(self.models)==list and not self.models:
            raise RuntimeError("No fitted models found. You may not call GRN.predict() until you have called GRN.fit().")
        assert type(perturbations)==list, "Perturbations must be like [('NANOG', 1.56), ('OCT4', 1.82)]"
        assert all(type(p)==tuple  for p in perturbations), "Perturbations must be like [('NANOG', 1.56), ('OCT4', 1.82)]"
        assert all(len(p)==2       for p in perturbations), "Perturbations must be like [('NANOG', 1.56), ('OCT4', 1.82)]"
        assert all(type(p[0])==str for p in perturbations), "Perturbations must be like [('NANOG', 1.56), ('OCT4', 1.82)]"
        
        # Set up AnnData object in the right shape to contain output
        columns_to_transfer = self.training_args["confounders"].copy() + ["perturbation_type"]
        if self.training_args["cell_type_sharing_strategy"] != "identical":
            columns_to_transfer.append(self.training_args["cell_type_labels"])
        if predictions is None:
            predictions = _prediction_prep(
                nrow = len(perturbations),
                columns_to_transfer = columns_to_transfer,
                var = self.train.var.copy(),
            )
            # Raw data are needed for GeneFormer feature extraction, but otherwise, save RAM by omitting
            if feature_extraction_requires_raw_data:
                predictions.raw = anndata.AnnData(
                    X = scipy.sparse.dok_matrix((predictions.n_obs, self.train.raw.n_vars)),
                    obs = predictions.obs.copy(), 
                    var = self.train.raw.var.copy()
                )
            # We allow users to select a subset of controls using a prefix in .obs["perturbation"], e.g. 
            # GFP overexpression instead of parental cell line in Nakatake. 
            if control_subtype is None or isnan_safe(control_subtype):
                all_controls = self.train.obs["is_control"] 
            else:
                print(f"Using controls prefixed by {control_subtype}")
                all_controls = self.train.obs["is_control"] & self.train.obs["perturbation"].str.contains(control_subtype, regex = True)
                if all_controls.sum() == 0:
                    print(all_controls)
                    print(self.train.obs.head())
                    raise ValueError(f"No controls found for control_subtype {control_subtype}.")
            all_controls = np.where(all_controls)[0]
            # Copy all starting expression and metadata from the aforementioned controls
            starting_metadata_one   = self.train.obs.iloc[[all_controls[0]], :].copy()
            def toArraySafe(X):
                try:
                    return X.toarray()
                except:
                    return X
            predictions_one = toArraySafe(self.train.X[  all_controls,:]).mean(axis=0, keepdims = True)
            if feature_extraction_requires_raw_data:
                predictions_one_raw = toArraySafe(self.train.raw.X[  all_controls,:]).mean(axis=0, keepdims = True)
            for i in range(len(perturbations)):
                idx_str = str(i)
                predictions.X[i, :] = predictions_one.copy()
                if feature_extraction_requires_raw_data:
                    predictions.raw.X[i, :] = predictions_one_raw.copy()
                for col in columns_to_transfer:
                    predictions.obs.loc[idx_str, col] = starting_metadata_one.loc[:,col][0]

        # Enforce that predictions has correct type, shape, metadata fields
        assert type(predictions) == anndata.AnnData, f"predictions must be anndata; got {type(predictions)}"
        assert predictions.obs.shape[0] == len(perturbations), "Starting expression must be None or an AnnData with one obs per perturbation."
        assert all(c in set(predictions.obs.columns) for c in columns_to_transfer), f"predictions must be accompanied by these metadata fields: \n{'  '.join(columns_to_transfer)}"
        predictions.obs["perturbation"] = "NA"
        predictions.obs["expression_level_after_perturbation"] = -999
        predictions.obs.index = [str(x) for x in range(len(predictions.obs.index))]

        # At this point, predictions is in the right shape, and has most of the
        # right metadata.
        # But it doesn't have the right perturbations listed in the metadata.
        # And the perturbations have not been propagated to the features or the expression predictions. 
        
        # Now implement perturbations, including altering metadata and features,
        # but not yet downstream expression.
        predictions.obs["perturbation"] = predictions.obs["perturbation"].astype(str)
        predictions.obs["is_control"] = False
        for i in range(len(perturbations)):
            idx_str = str(i)
            # Expected input: comma-separated strings like ("C6orf226,TIMM50,NANOG", "0,0,0") 
            predictions.obs.loc[idx_str, "perturbation"]                        = perturbations[i][0]
            predictions.obs.loc[idx_str, "expression_level_after_perturbation"] = perturbations[i][1]
            
        # Now predict the expression
        if self.training_args["method"].startswith("autoregressive"):
            predictions = self.models.predict(perturbations = perturbations, predictions = predictions)
        elif self.training_args["method"].startswith("docker"):
            try:
                _, docker_args, image_name = self.training_args["method"].split("__")
            except ValueError as e:
                raise ValueError(f"docker argument parsing failed on input {self.training_args['method']} with error {repr(e)}. Expected format: docker__myargs__mycontainer")
            print(f"image name: {image_name}")
            print(f"args to 'docker run': {docker_args}")
            self.training_args["docker_args"] = docker_args
            self.train.uns["perturbed_and_measured_genes"] = list(self.train.uns["perturbed_and_measured_genes"])
            self.train.uns["perturbed_but_not_measured_genes"] = list(self.train.uns["perturbed_but_not_measured_genes"])
            self.train.write_h5ad("from_to_docker/train.h5ad")
            with open("from_to_docker/perturbations.json", 'w') as f:
                json.dump(perturbations, f)
            assert os.path.isfile("from_to_docker/train.h5ad"), "Expected to find from_to_docker/train.h5ad"
            cmd = f"docker run --rm " + \
                f" --mount type=bind,source={os.getcwd()}/from_to_docker/,destination=/from_to_docker " + \
                f" {self.training_args['docker_args']} " + \
                f" {image_name}"
            print(f"Running command: \n{cmd}")
            os.system(cmd)
            assert os.path.isfile("from_to_docker/predictions.h5ad"), "Expected to find from_to_docker/predictions.h5ad"
            predictions = sc.read_h5ad("from_to_docker/predictions.h5ad")
            predictions.obs["perturbation"] = predictions.obs["perturbation"].astype(str)
            predictions.obs["expression_level_after_perturbation"] = predictions.obs["expression_level_after_perturbation"].astype(str)
            assert all([predictions.obs.loc[idx, "perturbation"] == perturbations[i][0] 
                        for i,idx in enumerate(predictions.obs_names)]), \
                "Method in Docker container violated expectations of GGRN: Output must contain the " + \
                "perturbations in the expected order, labeled in adata.obs['perturbation']"
            assert all([predictions.obs.loc[idx, "expression_level_after_perturbation"] == str(perturbations[i][1])
                        for i,idx in enumerate(predictions.obs_names)]), \
                "Method in Docker container violated expectations of GGRN: Output must contain the " + \
                "expression levels after perturbation in the expected order, labeled in adata.obs['expression_level_after_perturbation']"
            assert all(predictions.var_names == self.train.var_names), \
                "Method in Docker container violated expectations of GGRN:" + \
                "Variable names must be identical between training and test data."
        elif self.training_args["method"].startswith("DCDFG"):
            predictions = self.models.predict(perturbations, baseline_expression = predictions.X)
        elif self.training_args["method"].startswith("GEARS"):
            non_control = [p for p in perturbations if all(g in self.models.pert_list for g in p[0].split(","))]
            # GEARS does not need expression level after perturbation
            non_control = [p[0] for p in non_control]
            # GEARS prediction is slow so it helps a lot to remove the dupes.
            non_control = list(set(non_control))
            # Input is list of lists
            y = self.models.predict([p.split(",") for p in non_control])
            # Result is a dict with keys like "FOXA1_HNF4A"
            for i, p in enumerate(perturbations): 
                if p[0] in self.models.pert_list:
                    predictions.X[i,:] = y[p[0].replace(",", "_")]
        else: 
            if self.training_args["cell_type_sharing_strategy"] == "distinct":
                cell_type_labels = predictions.obs[self.training_args["cell_type_labels"]]
            else:
                cell_type_labels = None
            for i in range(prediction_timescale):
                starting_features = self.extract_features(
                    train = match_controls(predictions, matching_method = "steady_state", matched_control_is_integer=False), # Feature extraction requires matching to be done already. 
                    in_place=False, 
                ).copy()
                if feature_extraction_requires_raw_data:
                    del predictions.raw # save memory: raw data are no longer needed after feature extraction
                gc.collect()
                y = self.predict_parallel(features = starting_features, cell_type_labels = cell_type_labels, do_parallel = do_parallel) 
                for i in range(len(self.train.var_names)):
                    predictions.X[:,i] = y[i]
            # Set perturbed genes equal to user-specified expression, not whatever the endogenous level is predicted to be
            for i, pp in enumerate(perturbations):
                if pp[0] in predictions.var_names:
                    predictions[i, pp[0]].X = pp[1]
            
        # Add noise. This is useful for simulations. 
        if add_noise:
            np.random.seed(seed)
            for i in range(len(self.train.var_names)):
                if noise_sd is None:
                    # From the docs on the RidgeCV attribute ".cv_values_":
                    # 
                    # > only available if store_cv_values=True and cv=None). 
                    # > After fit() has been called, this attribute will contain 
                    # > the mean squared errors if scoring is None otherwise it will 
                    # > contain standardized per point prediction values.
                    # 
                    try:
                        noise_sd = np.sqrt(np.mean(self.models[i].cv_values_))
                    except AttributeError:
                        raise ValueError("Noise standard deviation could not be extracted from trained models.")
                predictions.X[:,i] = predictions.X[:,i] + np.random.standard_normal(len(predictions.X[:,i]))*noise_sd
        return predictions

    def extract_features(self, train: anndata.AnnData = None, in_place: bool = True, geneformer_finetune_labels: str = "louvain", n_cpu = 15):
        """Create a feature matrix where each row matches a row in self.train and
        each column represents a feature in self.eligible_regulators.

        Args:
            train: (anndata.AnnData, optional). Expression data to use in feature extraction. Defaults to self.train.
            in_place (bool, optional): If True (default), put results in self.features. Otherwise, return results as a matrix. 
            geneformer_finetune_labels (str, optional): Labels to use for fine-tuning geneformer. 
        """
        assert self.feature_extraction.lower().startswith("geneformer_model_") or \
            self.feature_extraction.lower() in {'geneformer', 'geneformer_finetune', 'geneformer_hyperparam_finetune'} or \
                self.feature_extraction.lower() in {"tf_mrna", "tf_rna", "mrna"}, \
                    f"Feature extraction must be 'mrna', 'geneformer', 'geneformer_finetune', 'geneformer_hyperparam_finetune', or 'geneformer_model_<path/to/pretrained/model>'; got '{self.feature_extraction}'."
        if train is None:
            train = self.train # shallow copy is best here
        assert "matched_control" in train.obs.columns, "Feature extraction requires that matching (with `match_controls()`) has already been done."

        if self.feature_extraction.lower() in {"tf_mrna", "tf_rna", "mrna"}:
            if self.eligible_regulators is None:
                self.eligible_regulators = train.var_names
            self.eligible_regulators = [g for g in self.eligible_regulators if g in train.var_names]
            features = train[np.array(train.obs["matched_control"]),self.eligible_regulators].X.copy()
            for int_i, i in enumerate(train.obs.index):
                # Expected format: comma-separated strings like ("C6orf226,TIMM50,NANOG", "0,0,0") 
                pert_genes = train.obs.loc[i, "perturbation"].split(",")
                pert_exprs = [float(f) for f in str(train.obs.loc[i, "expression_level_after_perturbation"]).split(",")]
                assert len(pert_genes) == len(pert_exprs), f"Malformed perturbation in sample {i}: {train.obs[i,:]}"
                for pert_idx in range(len(pert_genes)):
                    # If perturbation label is not a gene, leave it unperturbed.
                    # If expression after perturbation is nan, leave it unperturbed.
                    # This behavior is used to handle controls. 
                    if pert_genes[pert_idx] not in self.eligible_regulators or np.isnan(pert_exprs[pert_idx]):
                        pass
                    else:
                        column = self.eligible_regulators.index(pert_genes[pert_idx])
                        features[int_i, column] = pert_exprs[pert_idx]
            # Return or add to GRN object
            if in_place:
                self.features = features # shallow copy is best here too
                return
            else:
                return features

        if self.feature_extraction.lower().startswith("geneformer"):
            assert all(self.train.obs.index==self.train.obs["matched_control"]), "Currently GeneFormer feature extraction can only be used with the steady-state matching scheme."
            if not HAS_GENEFORMER:
                raise ImportError("GeneFormer backend is not installed, and related models will not be available.")
            self.geneformer_finetuned = "ctheodoris/GeneFormer" # Default to pretrained model with no fine-tuning
            self.eligible_regulators = list(range(256))
            file_with_tokens = geneformer_embeddings.tokenize(adata_train = train, geneformer_finetune_labels = geneformer_finetune_labels)
        # Check if user has already fine-tuned a model
        if self.feature_extraction.lower().startswith("geneformer_model_"):
            self.geneformer_finetuned = self.feature_extraction.removeprefix("geneformer_model_")
        # Check if we have labels to fine-tune a model
        if self.feature_extraction.lower() in {"geneformer_finetune", "geneformer_hyperparam_finetune"} and geneformer_finetune_labels not in train.obs.columns:
            print(f"The column {geneformer_finetune_labels} was not found in the training data, so geneformer cannot be fine-tuned. "
                    "Embeddings will be extracted from the pre-trained model. If this message occurs after training, the fine-tuning "
                    "may have taken place already during training, and that model will be re-used.")
            geneformer_finetune_labels = None
        # Fine-tune without hyperparam sweep if 0) desired and 1) possible and 2) not done already
        if self.feature_extraction == "geneformer_finetune" and \
            geneformer_finetune_labels is not None and \
                self.geneformer_finetuned is None:
            self.geneformer_finetuned = geneformer_hyperparameter_optimization.finetune_classify(
                    file_with_tokens = file_with_tokens, 
                    column_with_labels = geneformer_finetune_labels,
                )
        # Fine-tune with hyperparam sweep if 0) desired and 1) possible and 2) not done already
        if self.feature_extraction.lower() in {"geneformer_hyperparam_finetune"} and \
            geneformer_finetune_labels is not None and \
                self.geneformer_finetuned is None:
            try:
                optimal_hyperparameters = geneformer_hyperparameter_optimization.optimize_hyperparameters(file_with_tokens, column_with_labels = geneformer_finetune_labels, n_cpu = n_cpu)
                self.geneformer_finetuned = geneformer_hyperparameter_optimization.finetune_classify(
                    file_with_tokens = file_with_tokens, 
                    column_with_labels = geneformer_finetune_labels,
                    max_input_size = 2 ** 11,  # 2048
                    max_lr                = optimal_hyperparameters[2]["learning_rate"],
                    freeze_layers = 2,
                    geneformer_batch_size = optimal_hyperparameters[2]["per_device_train_batch_size"],
                    lr_schedule_fn        = optimal_hyperparameters[2]["lr_scheduler_type"],
                    warmup_steps          = optimal_hyperparameters[2]["warmup_steps"],
                    epochs                = optimal_hyperparameters[2]["num_train_epochs"],
                    optimizer = "adamw",
                    GPU_NUMBER = [], 
                    seed                  = 42, 
                    weight_decay          = optimal_hyperparameters[2]["weight_decay"],
                )
            except ray.tune.error.TuneError as e:
                print(f"Hyperparameter selection failed with error {repr(e)}. Fine-tuning with default hyperparameters.")
                self.feature_extraction = "geneformer_finetune"
        
        features = geneformer_embeddings.get_geneformer_perturbed_cell_embeddings(
            adata_train=train,
            file_with_tokens = file_with_tokens,
            assume_unrecognized_genes_are_controls = True,
            apply_perturbation_explicitly = True, 
            file_with_finetuned_model = self.geneformer_finetuned
        )
        # Return results or add to GRN object
        if in_place:
            self.features = features # shallow copy is best here too
            return
        else:
            return features
 
    def fit_models_parallel(self, FUN, network = None, do_parallel: bool = True, verbose: bool = False):
        if network is None:
            network = self.network
        if do_parallel:   
                m = Parallel(n_jobs=cpu_count()-1, verbose = verbose, backend="loky")(
                    delayed(apply_supervised_ml_one_gene)(
                        train_obs = self.train.obs,
                        target_expr = self.train.X[:,i],
                        features = self.features,
                        network = network,
                        training_args = self.training_args,
                        eligible_regulators = self.eligible_regulators, 
                        FUN=FUN, 
                        target = self.train.var_names[i],
                    )
                    for i in range(len(self.train.var_names))
                )
                return m
        else:
            return [
                apply_supervised_ml_one_gene(
                    train_obs = self.train.obs,
                    target_expr = self.train.X[:,i],
                    features = self.features,
                    network = network,
                    training_args = self.training_args,
                    eligible_regulators = self.eligible_regulators, 
                    FUN=FUN, 
                    target = self.train.var_names[i],
                )
                for i in range(len(self.train.var_names))
            ]

    def fit_with_or_without_pruning(
        self, 
        FUN, 
        network_prior: str, 
        pruning_strategy: str, 
        pruning_parameter: int,
        do_parallel: bool,
        verbose:int = 1  ,          
    ):
        """Apply a supervised ML method to predict target expression from TF activity.

        Args:
            FUN (function): should accept inputs X,y and return a fitted model, sklearn-style, with a .predict method.
            verbose (int): passes to joblib.Parallel. https://joblib.readthedocs.io/en/latest/parallel.html
            network_prior (str): See args for 'fit' method
            pruning_strategy (str): See args for 'fit' method
            pruning_parameter (int): See args for 'fit' method
            do_parallel (bool): If True, use joblib parallelization to fit models. 
            verbose (int, optional): Passed to joblib.Parallel. Defaults to 1.

        Returns:
            No return value. Instead, this modifies self.models.
        """

        if pruning_strategy == "prune_and_refit":
            assert HAS_NETWORKS, "Pruning network structure is not available without our load_networks module."
            print("Fitting")
            self.models = self.fit_models_parallel(
                FUN = FUN, 
                network = None, 
                do_parallel = do_parallel, 
                verbose = verbose,
            )
            print("Pruning")
            chunks = []
            for i in range(len(self.train.var_names)):
                if self.training_args["cell_type_sharing_strategy"] == "distinct":
                    for cell_type in self.train.obs[self.training_args["cell_type_labels"]].unique():
                        try:
                            coef = self.models[i][cell_type].coef_.squeeze()
                        except Exception as e:
                            raise ValueError(f"Unable to extract coefficients. Pruning may not work with this type of regression model. Error message: {repr(e)}")

                        chunks.append(
                            pd.DataFrame(
                                    {
                                        "regulator": get_regulators(
                                                eligible_regulators = self.eligible_regulators, 
                                                predict_self = self.training_args["predict_self"], 
                                                network_prior = network_prior,
                                                target = self.train.var_names[i], 
                                                network = self.network,
                                                cell_type = cell_type,
                                            ), 
                                        "target": self.train.var_names[i], 
                                        "weight": coef,
                                        "cell_type": cell_type,
                                    } 
                                    
                                )
                            )
                elif self.training_args["cell_type_sharing_strategy"] == "identical":
                    chunks.append(
                        pd.DataFrame(
                                {
                                    "regulator": get_regulators(
                                            eligible_regulators = self.eligible_regulators, 
                                            predict_self = self.training_args["predict_self"], 
                                            network_prior = network_prior,
                                            target = self.train.var_names[i], 
                                            network = self.network,
                                            ), 
                                    "target": self.train.var_names[i], 
                                    "weight": self.models[i].coef_.squeeze(),
                                } 
                            )
                        )
                else:
                    raise NotImplementedError("Invalid value of 'cell_type_sharing_strategy' ")

            pruned_network = pd.concat(chunks)
            pruned_network = pruned_network.query("regulator != 'NO_REGULATORS'")
            pruned_network.loc[:, "abs_weight"] = np.abs(pruned_network["weight"])
            pruned_network = pruned_network.nlargest(int(pruning_parameter), "abs_weight")
            del pruned_network["abs_weight"]
            print("Re-fitting with just features that survived pruning")
            self.network = load_networks.LightNetwork(df=pruned_network)
            self.training_args["network_prior"] = "restrictive"
            self.models = self.fit_models_parallel(
                FUN = FUN, 
                network = None, 
                do_parallel = do_parallel, 
                verbose = verbose,
            )
        elif pruning_strategy.lower() == "none":
            print("Fitting")
            self.models = self.fit_models_parallel(
                FUN = FUN, 
                network = None, 
                do_parallel = do_parallel, 
                verbose = verbose,
            )
        else:
            raise NotImplementedError(f"pruning_strategy should be one of 'none' or 'prune_and_refit'; got {pruning_strategy} ")

    def save_models(self, folder_name:str):
        """Save all the regression models to a folder via joblib.dump (one file per gene).

        Args:
            folder_name (str): Where to save files.
        """
        os.makedirs(folder_name, exist_ok=True)
        if self.training_args["method"].startswith("DCDFG"):
            raise NotImplementedError(f"Parameter saving/loading is not supported for DCDFG")
        elif self.training_args["method"].startswith("docker"):
            raise NotImplementedError(f"Parameter saving/loading is not supported for docker")
        elif self.training_args["method"].startswith("GeneFormer"):
            raise NotImplementedError(f"Parameter saving/loading is not supported for GeneFormer")
        elif self.training_args["method"].startswith("autoregressive"):
            raise NotImplementedError(f"Parameter saving/loading is not supported for autoregressive")
        elif self.training_args["method"].startswith("GEARS"):
            self.models.save_model(os.path.join(folder_name, "gears"))
        else:
            for i,target in enumerate(self.train.var_names):
                dump(self.models[i], os.path.join(folder_name, f'{target}.joblib'))
        return

    def validate_method(self, method):
        assert type(method)==str, f"method arg must be a str; got {type(method)}"
        allowed_methods = [
            "mean",
            "median",
            "GradientBoostingRegressor",
            "ExtraTreesRegressor",
            "KernelRidge",
            "ElasticNetCV",
            "LarsCV",
            "OrthogonalMatchingPursuitCV",
            "ARDRegression",
            "BayesianRidge",
            "LASSO",
            "LassoCV",
            "LassoLarsIC",
            "RidgeCV",
            "RidgeCVExtraPenalty",
        ]
        allowed_prefixes = [
            "autoregressive",
            "GeneFormer",
            "docker",
            "GEARS",
            "DCDFG",
        ]
        for p in allowed_prefixes:
            if method.startswith(p):
                return
        for p in allowed_methods:
            if method == p:
                return
        print("Allowed prefixes:", flush=True)
        print(allowed_prefixes,    flush=True)
        print("Allowed methods:",  flush=True)
        print(allowed_methods,     flush=True)
        raise ValueError("method must start with an allowed prefix or must equal an allowed value (these will be printed to stdout).")  

    def check_perturbation_dataset(self):
        if CAN_VALIDATE:
            return load_perturbations.check_perturbation_dataset(ad=self.train)
        else:
            raise Exception("load_perturbations is not installed, so cannot validate data. Set validate_immediately=False.")

    def predict_parallel(self, features, cell_type_labels, network = None, do_parallel = True): 
        if network is None:
            network = self.network  
        if do_parallel:
            return Parallel(n_jobs=cpu_count()-1, verbose = 1, backend="loky")(
                delayed(predict_one_gene)(
                    target = self.train.var_names[i],
                    model = self.models[i],
                    features = features, 
                    network = network,
                    training_args = self.training_args,
                    eligible_regulators = self.eligible_regulators,
                    cell_type_labels = cell_type_labels,                     
                )
                for i in range(len(self.train.var_names))
            )
        else:
            return [
                predict_one_gene(
                    target = self.train.var_names[i],
                    model = self.models[i],
                    features = features, 
                    network = self.network,
                    training_args = self.training_args,
                    eligible_regulators = self.eligible_regulators,
                    cell_type_labels = cell_type_labels, 
                )
                for i in range(len(self.train.var_names))
            ]

# Having stuff inside Parallel be a stand-alone function (rather than a method of the GRN class) speeds
# up parallel computing by a TON, probably because it avoids copying the 
# entire GRN object to each new thread.  
def predict_one_gene(
    target: str, 
    model,
    features, 
    network,
    training_args,
    eligible_regulators,
    cell_type_labels,
) -> np.array:
    """Predict expression of one gene after perturbation. 

    Args:
        model: self.models[i], but for parallelization, we avoid passing self because it would have to get serialized and sent to another process.
        train_obs: self.train.obs, but we avoid passing self because it would have to get serialized & sent to another process.
        network: self.network (usually), but we avoid passing self because it would have to get serialized & sent to another process.
        training_args: self.training_args, but we avoid passing self because it would have to get serialized & sent to another process.
        eligible_regulators: self.eligible_regulators, but we avoid passing self because it would have to get serialized & sent to another process.

        target (str): Which gene to predict.
        features (matrix-like): inputs to predictive model, with shape num_regulators by num_perturbations. 
        cell_type_labels (Iterable): in case models are cell type-specific, this specifies the cell types. 

    Returns:
        np.array: Expression of gene i across all perturbed samples. 
    """
    predictions = np.zeros(features.shape[0])
    if training_args["cell_type_sharing_strategy"] == "distinct":
        for cell_type in cell_type_labels.unique():
            is_in_cell_type = cell_type_labels==cell_type
            regulators = get_regulators(
                eligible_regulators = eligible_regulators, 
                predict_self = training_args["predict_self"], 
                network_prior = training_args["network_prior"],
                target = target, 
                network = network,
                cell_type=cell_type
            )
            is_in_model = [tf in regulators for tf in eligible_regulators]
            if not any(is_in_model):
                X = 1 + 0*features[:,[0]]
            else:
                X = features[:,is_in_model]
            X = X[is_in_cell_type,:]
            predictions[is_in_cell_type] = model[cell_type].predict(X = X).squeeze()
    elif training_args["cell_type_sharing_strategy"] == "identical":
        regulators = get_regulators(
            eligible_regulators = eligible_regulators, 
            predict_self = training_args["predict_self"], 
            network_prior = training_args["network_prior"],
            target = target, 
            network = network,
            cell_type=None
        )
        is_in_model = [tf in regulators for tf in eligible_regulators]
        if not any(is_in_model):
            X = 1 + 0*features[:,[0]]
        else:
            X = features[:,is_in_model]
        predictions = model.predict(X = X).squeeze()
        
    else:
        raise NotImplementedError("Invalid cell_type_sharing_strategy.")
    return predictions

def apply_supervised_ml_one_gene(
    train_obs,
    features,
    network,
    training_args, 
    eligible_regulators, 
    FUN, 
    target: str,
    target_expr
    ):
    """Apply a supervised ML method to predict target expression from TF activity (one target gene only).

    Args:

        train_obs: self.train.obs, but we avoid passing self because it would have to get serialized & sent to another process.
        features: self.features, but we avoid passing self so that we can treat this as memory-mapped. 
        network: self.network (usually), but we avoid passing self because it would have to get serialized & sent to another process.
        training_args: self.training_args, but we avoid passing self because it would have to get serialized & sent to another process.
        eligible_regulators: self.eligible_regulators, but we avoid passing self because it would have to get serialized & sent to another process.

        FUN (Callable): See GRN.apply_supervised_ml docs.
        target (str): target gene symbol
        target_expr: target gene expression levels

    Returns:
        _type_: dict with cell types as keys and with values containing result of FUN. see GRN.fit docs.
    """
    is_target_perturbed = train_obs["perturbation"]==target
    if training_args["cell_type_sharing_strategy"] == "distinct":
        assert training_args["cell_type_labels"] is not None,               "cell_type_labels must be provided."
        assert training_args["cell_type_labels"] in set(train_obs.columns), "cell_type_labels must name a column in .obs of training data."
        models = {}
        for cell_type in train_obs[training_args["cell_type_labels"]].unique():
            rrrelevant_rrregulators = get_regulators(
                eligible_regulators       = eligible_regulators, 
                predict_self  = training_args["predict_self"], 
                network_prior = training_args["network_prior"],
                target        = target, 
                network       = network,
                cell_type     = cell_type, 
            )
            is_in_model = [tf in rrrelevant_rrregulators for tf in eligible_regulators]
            is_in_cell_type = train_obs[training_args["cell_type_labels"]]==cell_type
            if not any(is_in_model):
                X = np.ones(shape=features[:,[0]].shape)
            else:
                X = features[:,is_in_model]
            models[cell_type] = FUN(
                X = X[is_in_cell_type & ~is_target_perturbed,:], 
                y = target_expr[is_in_cell_type & ~is_target_perturbed]
            )
        return models
    elif training_args["cell_type_sharing_strategy"] == "similar":
        raise NotImplementedError("cell_type_sharing_strategy 'similar' is not implemented yet.")
    elif training_args["cell_type_sharing_strategy"] == "identical":
        rrrelevant_rrregulators = get_regulators(
            eligible_regulators       = eligible_regulators, 
            predict_self  = training_args["predict_self"], 
            network_prior = training_args["network_prior"],
            target        = target, 
            network       = network,
            cell_type = None,
        )
        is_in_model = [tf in rrrelevant_rrregulators for tf in eligible_regulators]
        if not any(is_in_model):
            X = np.ones(shape=features[:,[0]].shape)
        else:
            X = features[:,is_in_model]
        return FUN(X = X[~is_target_perturbed, :], y = target_expr[~is_target_perturbed])
    else:
        raise ValueError("cell_type_sharing_strategy must be 'distinct' or 'identical' or 'similar'.")
    return

def get_regulators(eligible_regulators, predict_self: bool, network_prior: str, target: str, network = None, cell_type = None) -> list:
    """Get candidates for what's directly upstream of a given gene.

    Args:
        eligible_regulators: list of all possible candidate regulators. Something might be missing from this list if it is not a TF and you only allow TF's to be regulators.
        predict_self: Should a candidate regulator be used to predict itself? If False, removed no matter what. if True, added if in eligible_regulators. 
            This completely overrides the network data; presence or absence of self-loops in the network is irrelevant.
        network_prior (str): see GRN.fit docs
        target (str): target gene
        network (load_networks.LightNetwork): see GRN.fit docs
        cell_type: if network structure is cell-type-specific, which cell type to use when getting regulators.

    Returns:
        (list of str): candidate regulators currently included in the model for predicting 'target'.
        Given the same inputs, these will always be returned in the same order -- furthermore, in the 
        order that they appear in eligible_regulators.
    """
    if network_prior == "ignore":
        selected_features = eligible_regulators.copy()
        assert network is None or network.get_all().shape[0]==0, f"Network_prior was set to 'ignore', but an informative network was passed in: \n{network}"
    elif network_prior == "restrictive":
        assert network is not None, "For restrictive network priors, you must provide the network as a LightNetwork object."
        regulators = network.get_regulators(target=target)
        if cell_type is not None and "cell_type" in regulators.keys():
            regulators = regulators.loc[regulators["cell_type"]==cell_type,:]
        selected_features = [tf for tf in eligible_regulators if tf in set(regulators["regulator"])]
    else:
        raise ValueError("network_prior must be one of 'ignore' and 'restrictive'. ")

    if not predict_self:
        try:
            selected_features = [tf for tf in selected_features if tf != target]
        except (KeyError, ValueError) as e:
            pass
    else:
        if target in eligible_regulators:
            selected_features.append(target)
            selected_features = list(set(selected_features))
    if len(selected_features)==0:
        return ["NO_REGULATORS"]
    else:
        return selected_features

def _prediction_prep(nrow: int, columns_to_transfer: list, var: pd.DataFrame) -> anndata.AnnData:
    """Prior to generating predictions, set up an empty container in the right shape.

    Args:
        nrow (int): Number of samples 
        columns_to_transfer (list): List of additional columns to include in .obs.
        var (pd.DataFrame): per-gene metadata

    Returns:
        anndata.AnnData: empty container with 0 expression
    """
    predictions = anndata.AnnData(
        X = np.zeros((nrow, len(var.index))),
        dtype=np.float32,
        var = var,
        obs = pd.DataFrame(
            {
                "perturbation":"NA", 
                "expression_level_after_perturbation": -999, 
            }, 
            index = [str(i) for i in range(nrow)],
            columns = ["perturbation", "expression_level_after_perturbation"] + columns_to_transfer,
        )
    )
    return predictions

def isnan_safe(x):
    """
    A safe way to check if an arbitrary object is np.nan. Useful because when we save metadata to csv, it will
    be read as NaN even if it was written as null. 
    """
    try:
        return np.isnan(x)
    except:
        return False


def match_controls(train_data: anndata.AnnData, matching_method: str, matched_control_is_integer: bool):
    """
    Add info to the training data about which observations are paired with which.

    train_data (AnnData):
    matching_method (str): "steady_state" (match each obs to itself), "closest" (choose the closest control), or
      "user" (do nothing and expect an existing column 'matched_control'), or 
      "random" (choose a random control).
    matched_control_is_integer (bool): If True (default), the "matched_control" column in the obs of the returned anndata contains integers.
        Otherwise, it contains elements of adata.obs_names.
    """

    assert "is_control" in train_data.obs.columns, "A boolean column 'is_control' is required in train_data.obs."
    control_indices_integer = np.where(train_data.obs["is_control"])[0]
    if matching_method.lower() == "closest":
        assert any(train_data.obs["is_control"]), "matching_method=='closest' requires some True entries in train_data.obs['is_control']."
        # Index the control expression with a K-D tree
        kdt = KDTree(train_data.X[train_data.obs["is_control"],:], leaf_size=30, metric='euclidean')
        # Query the index to get 1 nearest neighbor
        nn = [nn[0] for nn in kdt.query(train_data.X, k=1, return_distance=False)]
        # Convert from index among controls to index among all obs
        train_data.obs["matched_control"] = control_indices_integer[nn]
        # Controls should be self-matched. K-D tree can violate this when input data have exact dupes.
        for j,i in enumerate(train_data.obs.index):
            if train_data.obs.loc[i, "is_control"]:
                train_data.obs.loc[i, "matched_control"] = j
    elif matching_method.lower() == "steady_state":
        train_data.obs["matched_control"] = range(len(train_data.obs.index))
    elif matching_method.lower() == "optimal_transport":
        raise NotImplementedError("Sorry, cannot yet match samples to controls by optimal transport.")
    elif matching_method.lower() == "random":
        assert any(train_data.obs["is_control"]), "matching_method=='random' requires some True entries in train_data.obs['is_control']."
        train_data.obs["matched_control"] = np.random.choice(
            control_indices_integer,
            train_data.obs.shape[0],
            replace = True,
        )
    elif matching_method.lower() == "user":
        assert "matched_control" in train_data.obs.columns, "You must provide obs['matched_control']."
        if matched_control_is_integer:
            assert all( train_data.obs.loc[train_data.obs["matched_control"].notnull(), "matched_control"].isin(range(train_data.n_obs)) ), f"Matched controls must be in range(train_data.n_obs)."
        else:
            assert all( train_data.obs.loc[train_data.obs["matched_control"].notnull(), "matched_control"].isin(train_data.obs.index) ), f"Matched control index must be in train_data.obs.index."
    else: 
        raise ValueError("matching method must be 'closest', 'user', or 'random'.")
    # This index_among_eligible_observations column is useful downstream for autoregressive modeling.
    train_data.obs["index_among_eligible_observations"] = train_data.obs["matched_control"].notnull().cumsum()-1
    train_data.obs.loc[train_data.obs["matched_control"].isnull(), "index_among_eligible_observations"] = np.nan

    if not matched_control_is_integer:
        has_matched_control = train_data.obs["matched_control"].notnull()
        train_data.obs.loc[has_matched_control, "matched_control"] = train_data.obs.index[train_data.obs.loc[has_matched_control, "matched_control"]]
    return train_data