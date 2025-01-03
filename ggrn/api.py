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
import subprocess
import tempfile

try:
    from torch.cuda import is_available as is_gpu_available
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
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
    import ray.tune.error
    HAS_GENEFORMER = True
except ImportError:
    HAS_GENEFORMER = False
try:
    import wot
    HAS_WOT = True
except ImportError:
    HAS_WOT = False


# These govern expectations about data format for regulatory networks and perturbation data.
try:
    import pereggrn_networks
    HAS_NETWORKS = True
except:
    HAS_NETWORKS = False
try:
    import pereggrn_perturbations
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
        memoization_folder = tempfile.mkdtemp(), 
        default_geneformer_model = "../Geneformer",
        species = "human",
    ):
        """Create a GRN object.

        Args:
            train (anndata.AnnData): Training data. Should conform to the requirements in pereggrn_perturbations.check_perturbation_dataset().
            network (pereggrn_networks.LightNetwork, optional): LightNetwork object containing prior knowledge about regulators and targets.
            eligible_regulators (List, optional): List of gene names that are allowed to be regulators. Defaults to all genes.
            validate_immediately (bool, optional): check validity of the input anndata?
            feature_extraction (str, optional): How to extract features. Defaults to "mrna", which uses the mRNA level of each regulator. You 
                can also specify: "geneformer_hyperparam_finetune", "geneformer_finetune", "geneformer", or "geneformer_model_<path/to/finetuned/model>"
                These all use GeneFormer to extract cell embeddings. 
                    - "geneformer_hyperparam_finetune" is preferred (fine-tuning with hyperparameter optimization), 
                    - "geneformer_finetune" is the next best option (fine-tuning without hyperparameter optimization). 
                    - "geneformer" does not do any fine-tuning and is not recommended.
                    - If you already have a finetuned GeneFormer model, you can provide the path to it by passing in "geneformer_model_<path/to/finetuned/model>".
                GGRN will cache the fine-tuned model in the memoization folder, so you can dig that up and reuse it.
            memoization_folder (str, optional): Where to store memoized GeneFormer models. Defaults to a temporary directory, so if you don't specify this,
                you'll lose the fine-tuned model when the program ends.
            default_geneformer_model (str, optional): Path to the pretrained GeneFormer model. Defaults to "../Geneformer".
            species (str, optional): Species of the training data. Defaults to "human".
        """
        self.species = species
        self.train = train
        try:
            self.train.X = self.train.X.toarray()
        except AttributeError:
            pass
        assert network is None or type(network)==pereggrn_networks.LightNetwork, "Network must be of type LightNetwork (see our pereggrn_networks package)."
        self.network = network 
        self.models = [None for _ in self.train.var_names]
        self.training_args = {}
        self.features = None
        self.geneformer_finetuned = None
        self.default_geneformer_model = default_geneformer_model
        self.feature_extraction = "mrna" if (feature_extraction is None or isnan_safe(feature_extraction)) else feature_extraction
        self.eligible_regulators = eligible_regulators
        self.memoization_folder = memoization_folder
        if validate_immediately:
            print("Checking the training data.")
            assert self.check_perturbation_dataset(is_timeseries = False, is_perturbation = False) # Less stringent version of checks
        if self.feature_extraction.lower().startswith("geneformer"):
            if not self.eligible_regulators is None:
                print("GeneFormer feature extraction requires all genes to be eligible regulators. eligible_regulators will be ignored.")
        print("Done.")
        return

    def fit(
        self,
        method: str, 
        confounders: list = [], 
        cell_type_sharing_strategy: str = "identical",   
        cell_type_labels: str = "cell_type",
        network_prior: str = None,    
        pruning_strategy: str = "none", 
        pruning_parameter: str = None,
        matching_method: str = "steady_state",
        feature_extraction: str = "mrna",
        low_dimensional_structure: str = None,
        low_dimensional_training: str = None,
        low_dimensional_value: str = None,
        prediction_timescale: list = [1], 
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
            cell_type_labels (str): Name of column in self.train.obs to use for cell type labels. This is deprecated and now it must be "cell_type".
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
            matching_method: (str, optional): see help(ggrn.api.match_controls).
            feature_extraction (str): Method of feature extraction. "mrna" or "geneformer". 
            low_dimensional_structure (str) "none" or "dynamics" or "RGQ". See also low_dimensional_training.
                - If "none", dynamics will be modeled using the original data.
                - If "dynamics", dynamics will be modeled in a latent space. 
                - "RGQ" is a deprecated option identical to "dynamics".
            low_dimensional_training (str): "SVD" or "fixed" or "supervised". How to learn the linear subspace. 
                If "SVD", perform an SVD on the data.
                If "fixed", use a user-provided projection matrix.
                If "supervised", learn the projection and its (approximate) inverse via backprop.
            low_dimensional_value (int): Number of dimensions to use in the low-dimensional space. 
            prediction_timescale (list, optional): For time-series models, the sequence of timepoints to make predictions at.
            predict_self (bool, optional): Should e.g. POU5F1 activity be used to predict POU5F1 expression? Defaults to False.
            test_set_genes: The genes that are perturbed in the test data. GEARS can use this extra info during training.
            kwargs: These are passed to the backend you chose.
        """
        assert cell_type_labels == "cell_type", "The column with cell type labels must be called 'cell_type'."
        if not isinstance(prediction_timescale, list):
            prediction_timescale = [prediction_timescale]
        if self.feature_extraction.lower().startswith("geneformer"):
            assert predict_self, "GGRN cannot exclude autoregulation when using geneformer for feature extraction."
            assert len(prediction_timescale)==1 and prediction_timescale[0]==1, f"GGRN cannot do iterative prediction when using geneformer for feature extraction. Please set prediction_timescale to [1]. Received {prediction_timescale}"
        if network_prior is None:
            network_prior = "ignore" if self.network is None else "restrictive"
        if network_prior != "ignore":
            assert self.network is not None, "You must provide a network unless network_prior is 'ignore'."
            assert feature_extraction.lower() in {"tf_mrna", "mrna", "rna", "tf_rna"},  "Only simple feature extraction is compatible with prior network structure."
        if ("timepoint" in self.train.obs.columns) and (self.train.obs["timepoint"].unique().shape[0] > 1):
            self.train = match_timeseries(self.train, matching_method, matched_control_is_integer=False) 
            assert "matched_control" in self.train.obs.columns, "Matching with `match_timeseries()` was not successful. Please report this issue."
        else:
            self.train = match_controls(self.train, matching_method, matched_control_is_integer=False) 
            assert "matched_control" in self.train.obs.columns, "Matching with `match_controls()` was not successful. Please report this issue."
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
                raise Exception("autoregressive backend is not installed.")
            if not HAS_TORCH: 
                raise ImportError("PyTorch is not installed, so autoregressive models will not be available.")
            np.testing.assert_equal(
                np.array(self.eligible_regulators), 
                np.array(self.train.var_names),     
                err_msg="autoregressive models can only use all genes as regulators."
            )
            assert len(confounders)==0, "autoregressive models cannot currently include confounders."
            assert cell_type_sharing_strategy=="identical", "autoregressive models cannot currently fit each cell type separately."
            assert predict_self, "autoregressive models currently cannot exclude autoregulation. Set predict_self=True to continue."
            assert len(prediction_timescale)==1, "autoregressive models currently can only use prediction_timescale=[1]."
            assert prediction_timescale[0]==1, "autoregressive models currently can only use prediction_timescale=[1]."
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
                S = 1
            )
            autoregressive_model.train()      
            self.models = autoregressive_model
        elif method.startswith("regulon"):
            # There's no training for this; it's a simple function of the user-provided network structure. 
            pass
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
        elif method.startswith("docker") or method.startswith("singularity"):
            # There is no actual training here, just copying data. 
            # Training happens when you call predict.
            # This is because we do not want to try to save trained models inside a container and return control flow to Python while leaving the container intact.
            try:
                subprocess.call(['sudo', 'rm', '-rf', 'from_to_docker'])
            except Exception as e:
                print(f"Could not remove previous docker data; message was {e}.")
                raise e
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
                "prediction_timescale": prediction_timescale,
                "predict_self": bool(predict_self), 
            }
            kwargs["species"] = self.species
            with open("from_to_docker/ggrn_args.json", "w") as f:
                json.dump(ggrn_args, f, default = float) # default=float allows numpy numbers to be converted instead of choking json.
            with open("from_to_docker/kwargs.json", "w") as f:
                json.dump(kwargs, f, default = float) 
        elif method.startswith("GEARS"):
            if not HAS_GEARS:
                raise ImportError("GEARS is not installed.")
            if not HAS_TORCH: 
                raise ImportError("PyTorch is not installed, and GEARS will not be available.")
            assert len(confounders)==0, "Our interface to GEARS cannot currently include confounders."
            assert len(prediction_timescale)==1 and prediction_timescale[0]==1, f"GEARS cannot do iterative prediction. Please set prediction_timescale=[1]. Received {prediction_timescale}."
            assert network_prior=="ignore", "Our interface to GEARS cannot currently include custom networks."
            assert cell_type_sharing_strategy=="identical", "Our interface to GEARS cannot currently fit each cell type separately."
            assert predict_self, "Our interface to GEARS cannot rule out autoregulation. Set predict_self=True."
            assert self.check_perturbation_dataset(is_timeseries = False, is_perturbation = True) # Run a more stringent version of the data format checks
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
            try:
                self.train.uns["perturbed_and_measured_genes"] = list(self.train.uns["perturbed_and_measured_genes"])
                self.train.uns["perturbed_but_not_measured_genes"] = list(self.train.uns["perturbed_but_not_measured_genes"])
            except KeyError:
                # timeseries training data might not have these keys, which is fine. 
                pass
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
            with tempfile.TemporaryDirectory() as gears_input_folder:
                pert_data = GEARSPertData(gears_input_folder, default_pert_graph = True)
                self.train.obs_names = ["cell_" + c for c in self.train.obs_names] #Empty string as obs names causes problems, so concat all with a prefix
                pert_data.new_data_process(dataset_name = 'current', adata = self.train)
                pert_data.load(data_path = os.path.join(gears_input_folder, 'current'))
                try:
                    pert_data.prepare_split(split = 'no_test', seed = kwargs["seed"] )
                    pert_data.get_dataloader(batch_size = kwargs["batch_size"], test_batch_size = kwargs["test_batch_size"])
                    self.models = GEARS(pert_data, device = kwargs["device"])
                    self.models.model_initialize(hidden_size = kwargs["hidden_size"])
                except Exception as e:
                    print(f"GEARS failed with error {repr(e)}. Falling back on a slightly suboptimal training strategy.")
                    pert_data.prepare_split(train_gene_set_size = 0.95, seed = kwargs["seed"] )
                    pert_data.get_dataloader(batch_size = kwargs["batch_size"], test_batch_size = kwargs["test_batch_size"])
                    self.models = GEARS(pert_data, device = kwargs["device"])
                    self.models.model_initialize(hidden_size = kwargs["hidden_size"])
                self.models.train(epochs = kwargs["epochs"])
        elif method.startswith("DCDFG"):
            assert HAS_DCDFG, "DCD-FG wrapper is not installed, and related models (DCD-FG, NOTEARS) cannot be used."
            assert HAS_TORCH, "PyTorch is not installed, so DCD-FG and related models will not be available."
            np.testing.assert_equal(
                np.array(self.eligible_regulators), 
                np.array(self.train.var_names),     
                err_msg="DCDFG can only use all genes as regulators."
            )
            assert len(confounders)==0, "DCDFG cannot currently include confounders."
            assert network_prior=="ignore", "DCDFG cannot currently include known network structure."
            assert cell_type_sharing_strategy=="identical", "DCDFG cannot currently fit each cell type separately."
            assert not predict_self, "DCDFG cannot include autoregulation."
            assert len(prediction_timescale)==1 and prediction_timescale[0]==1, f"DCD-FG cannot do iterative prediction. Please set prediction_timescale=[1]. Received {prediction_timescale}."
            factor_graph_model = dcdfg_wrapper.DCDFGCV()
            try:
                _, constraint_mode, model_type, do_use_polynomials = method.split("-")
            except:
                raise ValueError("DCDFG expects 'method' to be a hyphen-separated string DCDFG-constraint-model-do_polynomials like 'DCDFG-spectral_radius-mlplr-false'.")
            print(f"""DCDFG args parsed as:
               constraint_mode: {constraint_mode}
                    model_type: {model_type}
            do_use_polynomials: {do_use_polynomials}
            """)
            do_use_polynomials = do_use_polynomials =="True"

            try:
                regularization_parameters = kwargs["regularization_parameters"].copy()
                del kwargs["regularization_parameters"]
            except:
                regularization_parameters = None
            try: 
                latent_dimensions = kwargs["latent_dimensions"].copy()
                del kwargs["latent_dimensions"]
            except:
                latent_dimensions = None
            self.models = factor_graph_model.train(
                self.train,
                constraint_mode = constraint_mode,
                model_type = model_type,
                do_use_polynomials = do_use_polynomials,
                latent_dimensions = latent_dimensions,
                regularization_parameters = regularization_parameters,
                memoization_folder = self.memoization_folder,
                do_clean_memoization = True, 
                **kwargs,
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
        predictions: anndata.AnnData = None,
        predictions_metadata: pd.DataFrame = None,
        control_subtype = None,
        do_parallel: bool = True,
        add_noise: bool = False,
        noise_sd: float = None,
        seed: int = 0,
        prediction_timescale: list = [1],
        feature_extraction_requires_raw_data: bool = False
    ):
        """Predict expression after new perturbations.

        Args:
            predictions (anndata.AnnData): Expression prior to perturbation, in the same shape as the output predictions. If 
                None, starting state will be set to the mean of the training data control expression values.
            predictions_metadata (pd.DataFrame):  Dataframe with columns `perturbation_type`, `perturbation`, `expression_level_after_perturbation`, 
            `cell_type` (optional), `timepoint` (optional). It will default to `predictions.obs` if predictions is provided. The meaning is "predict 
            expression in `cell_type` at `time_point` if `perturbation` were set to `expression_level_after_perturbation`". Some details and defaults:
                - `perturbation` and `expression_level_after_perturbation` can contain comma-separated strings for multi-gene perturbations, for 
                    example "NANOG,POU5F1" for `perturbation` and "5.43,0.0" for `expression_level_after_perturbation`. 
                - Anything with expression_level_after_perturbation equal to np.nan will be treated as a control, no matter the name.
                - If timepoint or celltype or perturbation_type are missing, the default is to copy them from the top row of self.train.obs.
                - If the training data do not have a `cell_type` or `timepoint` column, then those *must* be omitted. Sorry; this is for backwards compatibility.
            control_subtype (str): only controls with this prefix are considered. For example, 
                in the Nakatake data there are different types of controls labeled "emerald" (GFP overexpression) and "rtTA" (empty overexpression vector).
            do_parallel (bool): if True, use joblib parallelization. 
            add_noise (bool): if True, return simulated data Y + e instead of predictions Y (measurement error)
                where e is IID Gaussian with variance equal to the estimated residual variance.
            noise_sd (bool): sd of the variable e described above. Defaults to estimates from the fitted models.
            seed (int): RNG seed.
            prediction_timescale (list): how many time-steps forward to predict. If the list has multiple elements, we predict all those points -- we return an expression trajectory).
            feature_extraction_requires_raw_data (bool): We recommend to set to True with GeneFormer and False otherwise.

        Returns: AnnData with predicted expression after perturbations. The shape and .obs metadata will be the same as the input "predictions" or "predictions_metadata".
        """

        if not isinstance(prediction_timescale, list):
            prediction_timescale = [prediction_timescale]
        if type(self.models)==list and not self.models:
            raise RuntimeError("No fitted models found. You may not call GRN.predict() until you have called GRN.fit().")
        assert predictions is None or predictions_metadata is None, "Provide either predictions or predictions_metadata, not both."
        assert predictions is not None or predictions_metadata is not None, "Provide either predictions or predictions_metadata."

        # Some extra info to be transferred from the training data.
        columns_to_transfer = self.training_args["confounders"].copy()
        

        if predictions is None:
            predictions = anndata.AnnData(
                X = np.zeros((predictions_metadata.shape[0], len(self.train.var_names))),
                dtype=np.float32,
                var = self.train.var.copy(),
                obs = predictions_metadata
            )
            predictions.obs_names_make_unique()

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
            # Copy mean expression and extra metadata from the selected controls
            starting_metadata_one   = self.train.obs.iloc[[all_controls[0]], :].copy()
            def toArraySafe(X):
                try:
                    return X.toarray()
                except:
                    return X
            predictions_one = toArraySafe(self.train.X[  all_controls,:]).mean(axis=0, keepdims = True)
            if feature_extraction_requires_raw_data:
                predictions_one_raw = toArraySafe(self.train.raw.X[  all_controls,:]).mean(axis=0, keepdims = True)
            for inti, i in enumerate(predictions.obs_names):
                predictions[i, :].X = predictions_one.copy()
                if feature_extraction_requires_raw_data:
                    predictions.raw.X[inti, :] = predictions_one_raw.copy()
                for col in columns_to_transfer:
                    predictions.obs.loc[i, col] = starting_metadata_one.loc[:,col][0]

        del predictions_metadata 

        # We don't want every single timeseries backend to have to resize the predictions object by a factor of len(prediction_timescale).
        # Instead we do it here, before calling backends. 
        def update_timescale(ad, t):
            ad = ad.copy()
            ad.obs["prediction_timescale"] = t
            return ad
        predictions = anndata.concat([
            update_timescale(predictions, t) for t in prediction_timescale
        ])

        # At this point, predictions is in the right shape, and has most of the
        # right metadata, and the expression is (mostly) set to the mean of the control samples.
        # Perturbed genes are set to the specified values in .obs, but not .X. 
        # But, the perturbations have not been propagated to the features or the expression predictions. 
        assert type(predictions) == anndata.AnnData, f"predictions must be anndata; got {type(predictions)}"
        predictions.obs["perturbation"] = predictions.obs["perturbation"].astype(str)
        if 'perturbation_type' not in predictions.obs.columns:
            if "perturbation_type" not in self.train.obs.columns:
                raise KeyError("A column perturbation_type must be present in predictions_metadata, or in predictions.obs, or in self.train.obs.")
            predictions.obs["perturbation_type"] = self.train.obs["perturbation_type"][0]
        if 'timepoint' not in predictions.obs.columns:
            if "timepoint" not in self.train.obs.columns:
                self.train.obs["timepoint"] = 1
            predictions.obs["timepoint"] = self.train.obs["timepoint"][0]
        if 'cell_type' not in predictions.obs.columns:
            if "cell_type" not in self.train.obs.columns:
                self.train.obs["cell_type"] = "unknown"
            predictions.obs["cell_type"] = self.train.obs["cell_type"][0]
        predictions.obs.index = [str(x) for x in range(len(predictions.obs.index))]
        desired_cols = columns_to_transfer + ['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation", "is_control"]
        for c in desired_cols:
            assert c in set(predictions.obs.columns), f"predictions must be accompanied by a metadata field {c}"
        
        # Now predict the expression
        if self.training_args["method"].startswith("autoregressive"):
            predictions = self.models.predict(predictions = predictions, prediction_timescale = prediction_timescale)
        elif self.training_args["method"].startswith("regulon"):
            # Assume targets go up if regulator goes up, and down if down
            assert HAS_NETWORKS, "To use the 'regulon' backend you must have access to our pereggrn_networks module."
            assert type(self.network) is pereggrn_networks.LightNetwork, "To use the 'regulon' backend you must provide a network structure."
            for i,row in predictions.obs.iterrows():
                assert len(row["perturbation"].split(","))==1, "the 'regulon' backend currently only handles single perturbations."
                logfc = float(row["expression_level_after_perturbation"]) - predictions[i, row["perturbation"]].X
                targets = self.network.get_targets(regulator = row["perturbation"])["target"]
                targets = list(set(predictions.var_names).intersection(targets))
                if len(targets) > 0:
                    predictions[i, targets].X = predictions[i, targets].X + logfc
        elif self.training_args["method"].startswith("docker") or self.training_args["method"].startswith("singularity"):
            try:
                containerizer, containerizer_args, image_name = self.training_args["method"].split("__")
            except ValueError as e:
                raise ValueError(f"docker/singularity argument parsing failed on input {self.training_args['method']} with error {repr(e)}. Expected format: docker__myargs__mycontainer")
            print(f"containerizer: {containerizer}")
            print(f"image name: {image_name}")
            print(f"args to containerizer: {containerizer_args}")
            self.training_args["containerizer_args"] = [s for s in containerizer_args.split(" ") if s!=""]
            try:
                self.train.uns["perturbed_and_measured_genes"] = list(self.train.uns["perturbed_and_measured_genes"])
                self.train.uns["perturbed_but_not_measured_genes"] = list(self.train.uns["perturbed_but_not_measured_genes"])
            except KeyError:
                # Timeseries training data are not required to have .uns["perturbed_but_not_measured_genes"] or .uns["perturbed_and_measured_genes"]
                pass
            self.train.write_h5ad("from_to_docker/train.h5ad")
            if self.network is not None: 
                self.network.get_all().to_parquet("from_to_docker/network.parquet")
            predictions.obs.to_csv("from_to_docker/predictions_metadata.csv")
            assert os.path.isfile("from_to_docker/train.h5ad"), "Expected to find from_to_docker/train.h5ad"
            if containerizer=='docker':
                pull_cmd = ["docker", "pull", f"{image_name}"]
                cmd = [
                    "docker", "run", 
                    "--rm",
                    "--mount", f"type=bind,source={os.getcwd()}/from_to_docker/,destination=/from_to_docker",
                ] + self.training_args['containerizer_args'] + [
                    f"{image_name}"
                ]
            elif containerizer=='singularity':
                pull_cmd = ["singularity", "pull", f"{image_name}"]
                cmd = [
                    "singularity", "run", 
                    "--no-home", # singularity default is to make a huge mess of portals between container and host and start in a working dir different from Docker. This helps avoid some of that. 
                    "--bind", f"{os.getcwd()}/from_to_docker/:/from_to_docker",
                ] + self.training_args['containerizer_args'] + [
                    f"docker://{image_name}"
                ]                
            else:
                raise ValueError(f"Unknown containerizer {containerizer}. Expected 'docker' or 'singularity'.")
            print(f"Pulling latest with command:\n\n{' '.join(pull_cmd)}\n\n", flush = True)
            return_code = subprocess.call(pull_cmd)
            print(f"Running container with command:\n\n{' '.join(cmd)}\n\nwith logs from_to_docker/stdout.txt, from_to_docker/err.txt", flush = True)
            with open('from_to_docker/stdout.txt', 'w') as out:
                with open('from_to_docker/err.txt', 'w') as err:
                    return_code = subprocess.call(cmd, stdout=out, stderr=err)
            print("\n\n\n===== DOCKER STDOUT =====\n\n\n")
            with open('from_to_docker/stdout.txt', 'r') as file:
                print(file.read())
            print("\n\n\n===== DOCKER STDERR =====\n\n\n")
            with open('from_to_docker/err.txt', 'r') as file:
                print(file.read())
            assert os.path.isfile("from_to_docker/predictions.h5ad"), "Expected to find from_to_docker/predictions.h5ad . You may find more info in the logs generated within the container: from_to_docker/err.txt , which should also get copied to stdout."
            predictions = sc.read_h5ad("from_to_docker/predictions.h5ad")
            assert predictions.n_obs > 0, "Method in Docker container predicted zero observations. Was there a warning about timepoints?"
            predictions.obs["perturbation"] = predictions.obs["perturbation"].astype(str)
            predictions.obs["expression_level_after_perturbation"] = predictions.obs["expression_level_after_perturbation"].astype(str)
            assert all(predictions.var_names == self.train.var_names), \
                "The backend running inside the Docker container violated expectations of GGRN:" + \
                "Variable names must be identical between training and test data."
        elif self.training_args["method"].startswith("DCDFG"):
            predictions = self.models.predict(
                predictions.obs[["perturbation", 'timepoint', 'cell_type', "expression_level_after_perturbation", 'prediction_timescale']], 
                baseline_expression = predictions.X
            )
        elif self.training_args["method"].startswith("GEARS"):
            # GEARS does not need expression level after perturbation, just gene names. 
            # The gene names must be in GEARS' GO graph.
            non_control = [p for p in predictions.obs["perturbation"] if all(g in self.models.pert_list for g in p.split(","))]
            # GEARS prediction is slow enough that we avoid duplicating work.
            non_control = list(set(non_control))
            # Input to GEARS is list of lists
            y = self.models.predict([p.split(",") for p in non_control])
            # Result is a dict with keys like "FOXA1_HNF4A"
            for i, row in predictions.obs.iterrows(): 
                if row["perturbation"] in self.models.pert_list:
                    predictions[i,:].X = y[row["perturbation"].replace(",", "_")]
        else: # backend 1, regression-based
            if self.training_args["cell_type_sharing_strategy"] == "distinct":
                cell_type_labels = predictions.obs[self.training_args["cell_type_labels"]]
            else:
                cell_type_labels = None
            for current_time in range(max(prediction_timescale)):
                print(f"Predicting time-point {current_time}")
                # We are computing f(f(...f(X)...)) one f() at a time and saving whichever intermediate timepoints are in prediction_timescale. 
                # predictions is already in the right shape etc to contain all timepoints.
                # 
                # Why is_last? Because we're going to use a view of the final timepoint as working memory, because it can be continuously 
                # overwritten until the last iteration. One the final iteration, it will reach what we want.
                is_last = predictions.obs["prediction_timescale"]==max(prediction_timescale)
                starting_features = self.extract_features(
                    # Feature extraction requires matching to be done already. 
                    # We can just tell ggrn to extract features for each predicted instance from that same instance,
                    # because above, we initialized the expression to the right starting state. 
                    train = match_controls(
                        predictions[is_last, :].copy(), 
                        matching_method = "steady_state", 
                        matched_control_is_integer=False
                    ),
                    in_place=False, 
                ).copy()
                cell_type_labels = None if cell_type_labels is None else cell_type_labels[is_last]
                if feature_extraction_requires_raw_data:
                    del predictions.raw # save memory: raw data are no longer needed after feature extraction
                gc.collect()
                y = self.predict_parallel(features = starting_features, cell_type_labels = cell_type_labels, do_parallel = do_parallel) 
                for gene in range(len(self.train.var_names)):
                    predictions[is_last,gene].X = y[gene]
                # This will be output to the user or calling function                   
                if current_time in prediction_timescale and current_time in predictions.obs["timepoint"]:
                    for gene in range(len(self.train.var_names)):
                        predictions.X[predictions.obs["timepoint"]==current_time,gene] = y[gene]
            # Set perturbed genes equal to user-specified expression, not whatever the endogenous level is predicted to be
            for i in predictions.obs_names:
                for j,gene_name in enumerate(predictions.obs.loc[i, "perturbation"].split(",")):
                    if gene_name in predictions.var:
                        elap = str(predictions.obs.loc[i, "expression_level_after_perturbation"]).split(",")[j]
                        if pd.notnull(elap):
                            predictions[i, gene_name].X = float(elap)

        # Add noise. This is useful for simulations. 
        if add_noise:
            np.random.seed(seed)
            for i in range(len(self.train.var_names)):
                if noise_sd is None:
                    # From the docs on the RidgeCV attribute ".cv_values_":
                    # 
                    # > only available if store_cv_results=True and cv=None). 
                    # > After fit() has been called, this attribute will contain 
                    # > the mean squared errors if scoring is None otherwise it will 
                    # > contain standardized per point prediction values.
                    # 
                    try:
                        noise_sd = np.sqrt(np.mean(self.models[i].cv_values_))
                    except AttributeError:
                        raise ValueError("Noise standard deviation could not be extracted from trained models.")
                predictions.X[:,i] = predictions.X[:,i] + np.random.standard_normal(len(predictions.X[:,i]))*noise_sd
        
        print(f"{predictions.obs['perturbation'].unique().shape[0]} unique perturbation(s)", flush=True)        
        return predictions


    def get_finetuned_geneformer_model_memoized(self, file_with_tokens):
        """Follow slightly complicated GeneFormer finetuning logic

        - Does the user want fine-tuning? 
            - If no, don't. 
            - If yes, see next.
        - Do we have the right labels to do fine-tuning? If yes, see next. 
        - Have we finetuned a model already? If yes, don't fine-tune. If no, see next. 
        - Does the user want to optimize hyperparams? 
            - If no, `model = finetune_with_fixed_hyperparameters()`.
            - If yes, `model = finetune_with_hyperparameter_search()`. Did it work?
                - If no, `model = finetune_with_fixed_hyperparameters()`. 
        """
        if self.feature_extraction.lower().startswith("geneformer"):
            return self.default_geneformer_model
        if self.feature_extraction.lower().startswith("geneformer_model_"):
            return self.feature_extraction.removeprefix("geneformer_model_")
        assert self.feature_extraction.lower() in {"geneformer_finetune", "geneformer_hyperparam_finetune"}, f"Unknown feature extraction method {self.feature_extraction}."
        
        # Label-finding behavior: 
            # Default to "louvain" for backwards compatibility with results in the 2024 Oct 01 preprint.
            # If "louvain" is not there, default to "cell_type".
            # Use the user input, not the default, if it's there.
        if geneformer_finetune_labels is None: 
            if "louvain" in self.train.obs.columns:
                geneformer_finetune_labels = "louvain"
            elif "cell_type" in self.train.obs.columns:
                geneformer_finetune_labels = "cell_type"
            else:
                geneformer_finetune_labels = None
        if geneformer_finetune_labels not in self.train.obs.columns:
            print(f"The column {geneformer_finetune_labels} was not found in the training data, so geneformer cannot be fine-tuned. "
                "Embeddings will be extracted from the pre-trained model. If this message occurs after training (e.g. during prediction), the fine-tuning "
                "may have taken place already during training, and you can ignore this message.")
            return self.default_geneformer_model
        
        # Model will be cached during initial training. No need to redo fine tuning at prediction time. 
        if self.geneformer_finetuned is not None:
            return self.geneformer_finetuned
        
        # - Does the user want to optimize hyperparams? 
        #     - If yes, `model = finetune_with_hyperparameter_search()`. Did it work?
        #         - If no, `model = finetune_with_fixed_hyperparameters()`. 
        #     - If no, `model = finetune_with_fixed_hyperparameters()`.

        
        # Fine-tune with hyperparam sweep if 0) desired and 1) possible and 2) not done already
        if self.feature_extraction.lower() == "geneformer_hyperparam_finetune" and geneformer_finetune_labels is not None:
            try:
                optimal_hyperparameters = geneformer_hyperparameter_optimization.optimize_hyperparameters(file_with_tokens, column_with_labels = geneformer_finetune_labels, n_cpu = n_cpu)
                return geneformer_hyperparameter_optimization.finetune_classify(
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
        
        if self.feature_extraction == "geneformer_finetune" and geneformer_finetune_labels is not None:
            return geneformer_hyperparameter_optimization.finetune_classify(
                    file_with_tokens = file_with_tokens, 
                    column_with_labels = geneformer_finetune_labels,
                    base_model = self.default_geneformer_model,
                )
            

    def extract_features(self, train: anndata.AnnData = None, in_place: bool = True, geneformer_finetune_labels: str = None, n_cpu = 15):
        """Create a feature matrix to be used when predicting each observation from its matched control. 

        Args:
            train: (anndata.AnnData, optional). Expression data to use in feature extraction. Defaults to self.train.
            in_place (bool, optional): If True (default), put results in self.features. Otherwise, return results as a matrix. 
            geneformer_finetune_labels (str, optional): Name of column in self.train.obs containing labels to use for fine-tuning geneformer. 
        """
        assert self.feature_extraction.lower().startswith("geneformer_model_") or \
            self.feature_extraction.lower() in {'geneformer', 'geneformer_finetune', 'geneformer_hyperparam_finetune'} or \
                self.feature_extraction.lower() in {"tf_mrna", "tf_rna", "mrna"}, \
                    f"Feature extraction must be 'mrna', 'geneformer', 'geneformer_finetune', 'geneformer_hyperparam_finetune', or 'geneformer_model_<path/to/finetuned/model>'; got '{self.feature_extraction}'."
        if train is None:
            train = self.train # shallow copy is best here
        assert "matched_control" in train.obs.columns, "Feature extraction requires that matching (with `match_controls()`) has already been done."

        if self.feature_extraction.lower() in {"tf_mrna", "tf_rna", "mrna"}:
            if self.eligible_regulators is None:
                self.eligible_regulators = train.var_names
            self.eligible_regulators = [g for g in self.eligible_regulators if g in train.var_names]
            features = train[np.array(train.obs["matched_control"][train.obs["matched_control"].notnull()]),self.eligible_regulators].X.copy()
            for int_i, i in enumerate(train.obs.index[train.obs["matched_control"].notnull()]):
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

        # GeneFormer feature extraction
        assert self.feature_extraction.lower().startswith("geneformer"), f"unrecognized feature extraction method {self.feature_extraction}"
        if not all(self.train.obs.index==self.train.obs["matched_control"]):
            print("Currently, GeneFormer feature extraction can only be used with the steady-state matching scheme. Matching will be ignored.")
            self.train.obs["matched_control"] = self.train.obs.index
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed, so GeneFormer cannot be used.")
        if not HAS_GENEFORMER:
            raise ImportError("GeneFormer backend is not installed -- are you missing GeneFormer, geneformer_embeddings, or ray[tune]?")
        with tempfile.TemporaryDirectory() as folder_with_tokens:
            # Default to pretrained model with no fine-tuning
            self.eligible_regulators = list(range(256))
            file_with_tokens = geneformer_embeddings.tokenize(adata_train = train, geneformer_finetune_labels = geneformer_finetune_labels, folder_with_tokens = folder_with_tokens)
            # Either finetune a geneformer model, or retrieve an existing one.
            self.geneformer_finetuned = self.get_finetuned_geneformer_model_memoized(file_with_tokens)
            features = geneformer_embeddings.get_geneformer_perturbed_cell_embeddings(
                adata_train=train,
                file_with_tokens = file_with_tokens,
                assume_unrecognized_genes_are_controls = True,
                apply_perturbation_explicitly = True, 
                file_with_finetuned_model = self.geneformer_finetuned, 
                file_with_default_model = self.default_geneformer_model,
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
                m = Parallel(n_jobs=cpu_count(), verbose = verbose, backend="loky")(
                    delayed(apply_supervised_ml_one_gene)(
                        train_obs = self.train.obs.loc[self.train.obs["matched_control"].notnull(), :],
                        target_expr = self.train.X[self.train.obs["matched_control"].notnull(),i],
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
                    train_obs = self.train.obs.loc[self.train.obs["matched_control"].notnull(), :],
                    target_expr = self.train.X[self.train.obs["matched_control"].notnull(),i],
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

        Returns:
            No return value. Instead, this modifies self.models.
        """

        if pruning_strategy == "prune_and_refit":
            assert HAS_NETWORKS, "Pruning network structure is not available without our pereggrn_networks module."
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
            self.network = pereggrn_networks.LightNetwork(df=pruned_network)
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

    def list_regression_methods(self):
        return [
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
    
    
    def validate_method(self, method):
        assert type(method)==str, f"method arg must be a str; got {type(method)}"
        allowed_methods = self.list_regression_methods()
        allowed_prefixes = [
            "autoregressive",
            "GeneFormer",
            "docker",
            "singularity",
            "GEARS",
            "DCDFG",
            "regulon",
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

    def check_perturbation_dataset(self, **kwargs):
        if CAN_VALIDATE:
            return pereggrn_perturbations.check_perturbation_dataset(ad=self.train, **kwargs)
        else:
            raise Exception("pereggrn_perturbations is not installed, so cannot validate data. Set validate_immediately=False or install pereggrn_perturbations.")

    def predict_parallel(self, features, cell_type_labels, network = None, do_parallel = True): 
        if network is None:
            network = self.network  
        if do_parallel:
            return Parallel(n_jobs=cpu_count(), verbose = 1, backend="loky")(
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
            try:
                predictions[is_in_cell_type] = model[cell_type].predict(X = X).squeeze() 
            except KeyError:
                pass # not every cell type gets a model, due to the non-steady-state matching schemes
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
    # We train on input-output pairs, and there may be fewer pairs than samples when we don't assume steady state.
    # This vector needs to answer "was the target gene perturbed" for each **output** member of a pair.
    # That's why we subset to those with a non-null matched_control.
    is_target_perturbed = is_target_perturbed[train_obs["matched_control"].notnull()]
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
        network (pereggrn_networks.LightNetwork): see GRN.fit docs
        cell_type: if network structure is cell-type-specific, which cell type to use when getting regulators.

    Returns:
        (list of str): candidate regulators currently included in the model for predicting 'target'.
        Given the same inputs, these will always be returned in the same order -- furthermore, in the 
        order that they appear in eligible_regulators.
    """
    if network_prior == "ignore":
        selected_features = eligible_regulators.copy()
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

def isnan_safe(x):
    """
    A safe way to check if an arbitrary object is np.nan. Useful because when we save metadata to csv, it will
    be read as NaN even if it was written as null. 
    """
    try:
        return np.isnan(x)
    except:
        return False

def match_timeseries(train_data: anndata.AnnData, matching_method: str, matched_control_is_integer: bool, do_look_backwards: bool = True) -> anndata.AnnData:
    """For a timeseries dataset, match observations at each timepoint to counterparts, usually at the previous timepoint. 
    This function matches each data point to a counterpart, usually at the previous timepoint.

    Args:
        - train_data (AnnData): Expression data to use in feature extraction. Defaults to self.train.
        - matching_method (str): See match_controls.
        - matched_control_is_integer (bool): See match_controls.
        - do_look_backwards (bool): If True (default), match each timepoint to the previous one. If False, match each timepoint to the next one. 
            Ignored if "matching_method" is "user" or "steady_state".
    """
    train_data.obs_names = train_data.obs_names.astype(str)
    if matching_method == "user":
        assert "matched_control" in train_data.obs.columns, "If matching_method=='user', a column 'matched_control' must be present in train_data.obs."
        return train_data
    if matching_method == "steady_state":
        train_data.obs["matched_control"] = train_data.obs.index
        return train_data
    # Otherwise, proceed to match each timepoint to the next.
    timepoints = np.sort(train_data.obs["timepoint"].unique())
    assert len(timepoints) > 1, "match_timeseries requires at least two timepoints."
    train_data.obs["matched_control"] = np.nan
    if do_look_backwards:
        for i in range(len(timepoints)-1, 0, -1):
            current = train_data[train_data.obs["timepoint"].isin(timepoints[[i-1, i]]), :].copy()
            current.obs["is_control"] = current.obs["timepoint"] == timepoints[i-1]
            del current.obs["matched_control"]
            current = match_controls(current, matching_method=matching_method, matched_control_is_integer=False)
            current.obs.loc[current.obs["is_control"], "matched_control"] = np.nan # usually we would match controls to themselves. We don't want that here.
            train_data.obs.loc[current.obs.index, "matched_control"] = current.obs.loc[:, "matched_control"]
    else:
        for i in range(len(timepoints)-1):
            current = train_data[train_data.obs["timepoint"].isin(timepoints[[i+1, i]]), :].copy()
            current.obs["is_control"] = current.obs["timepoint"] == timepoints[i+1]
            del current.obs["matched_control"]
            current = match_controls(current, matching_method=matching_method, matched_control_is_integer=False)
            current.obs.loc[current.obs["is_control"], "matched_control"] = np.nan # usually we would match controls to themselves. We don't want that here.
            train_data.obs.loc[current.obs.index, "matched_control"] = current.obs.loc[:, "matched_control"]        
    if matched_control_is_integer:
        train_data.obs["integer_index"] = range(train_data.n_obs)
        train_data.obs["matched_control"] = [train_data.obs.loc[m, "integer_index"] if pd.notnull(m) else np.nan for m in train_data.obs.loc[:, "matched_control"]]
    return train_data

def match_controls(train_data: anndata.AnnData, matching_method: str, matched_control_is_integer: bool) -> anndata.AnnData:
    """
    Add info to the training data about which observations are paired with which.

    train_data (AnnData):
    matching_method (str): one of
        - "steady_state" (match each obs to itself)
        - "closest" (choose the closest control and match each control to itself)
        - "closest_with_unmatched_controls" (choose the closest control for each perturbed sample, but leave each control sample unmatched)
        - "user" (do nothing and expect an existing column 'matched_control')
        - "random" (choose a random control)
        - "optimal_transport" (match samples to controls by optimal transport, and match controls to themselves)
    matched_control_is_integer (bool): If True (default), the "matched_control" column in the obs of the returned anndata contains integers.
        Otherwise, it contains elements of adata.obs_names.
    """
    assert "is_control" in train_data.obs.columns, "A boolean column 'is_control' is required in train_data.obs."
    control_indices_integer = np.where(train_data.obs["is_control"])[0]
    if matching_method.lower() in {"closest", "closest_with_unmatched_controls"}:
        assert any(train_data.obs["is_control"]), "matching_method=='closest' requires some True entries in train_data.obs['is_control']."
        # Index the control expression with a K-D tree
        try: 
            X = train_data.X.toarray()
        except AttributeError:
            X = train_data.X
        kdt = KDTree(X[train_data.obs["is_control"],:], leaf_size=30, metric='euclidean')
        # Query the index to get 1 nearest neighbor
        nn = [nn[0] for nn in kdt.query(X, k=1, return_distance=False)]
        # Convert from index among controls to index among all obs
        train_data.obs["matched_control"] = control_indices_integer[nn]
        # Controls should be self-matched. K-D tree can violate this when input data have exact dupes.
        for j,i in enumerate(train_data.obs.index):
            if train_data.obs.loc[i, "is_control"]:
                train_data.obs.loc[i, "matched_control"] = np.nan if matching_method.lower() == "closest_with_unmatched_controls" else j
    elif matching_method.lower() == "optimal_transport":
        assert HAS_WOT, "wot is not installed, so optimal transport matching is unavailable. Please install it with 'pip install wot'."
        with tempfile.TemporaryDirectory() as temp_dir:
            VAR_GENE_DS_PATH = f"{temp_dir}/train.h5ad"
            CELL_DAYS_PATH = f"{temp_dir}/timepoints.txt"
            pd.DataFrame({"id": train_data.obs_names, "day": [int(b) for b in train_data.obs["is_control"]]}).to_csv(CELL_DAYS_PATH, sep = "\t")
            train_data.write_h5ad(VAR_GENE_DS_PATH)
            wot_adata = wot.io.read_dataset(VAR_GENE_DS_PATH, obs=[CELL_DAYS_PATH])
        wot_adata.obs["day"] = [int(b) for b in train_data.obs["is_control"]] # This seems redundant, but sometimes wot.io returns NaNs for day.
        ot_model  = wot.ot.OTModel(wot_adata,epsilon = 0.05, lambda1 = 1,lambda2 = 50)
        train_data.obs["matched_control"] = np.nan
        train_data.obs.loc[~train_data.obs["is_control"], "matched_control"] = control_indices_integer[ot_model.compute_transport_map(0,1).X.argmax(axis = 1)]
        # Controls should be self-matched. 
        for j,i in enumerate(train_data.obs.index):
            if train_data.obs.loc[i, "is_control"]:
                train_data.obs.loc[i, "matched_control"] = j
        train_data.obs["matched_control"] = train_data.obs["matched_control"].astype(int)
    elif matching_method.lower() == "steady_state":
        train_data.obs["matched_control"] = range(len(train_data.obs.index))
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
            mc = train_data.obs["matched_control"]
            mc = mc[train_data.obs["matched_control"].notnull()]
            assert all( mc.isin(train_data.obs.index) ), f"Matched control index must be in train_data.obs.index."
    else: 
        raise ValueError("matching method must be 'closest', 'user', 'optimal_transport', or 'random'.")

    # This index_among_eligible_observations column is useful downstream for autoregressive modeling.
    train_data.obs["index_among_eligible_observations"] = train_data.obs["matched_control"].notnull().cumsum()-1
    train_data.obs.loc[train_data.obs["matched_control"].isnull(), "index_among_eligible_observations"] = np.nan
    # The above code provides an integer index for each matched control. This code is for if you want e.g. a cell barcode instead.
    if not matched_control_is_integer and matching_method.lower() != "user":
        has_matched_control = train_data.obs["matched_control"].notnull()
        train_data.obs["matched_control_int"] =train_data.obs["matched_control"]
        train_data.obs["matched_control"] = train_data.obs["matched_control"].astype(str)
        train_data.obs.loc[has_matched_control, "matched_control"] = train_data.obs.index[[int(i) for i in train_data.obs.loc[has_matched_control, "matched_control_int"]]]
        train_data.obs.loc[~has_matched_control, "matched_control"] = np.nan
    return train_data