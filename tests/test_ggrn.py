import os
import shutil
import unittest
import pandas as pd
import numpy as np
import scanpy as sc 
import scipy
import anndata
import sys
import importlib
import ggrn.api as ggrn
import pereggrn_networks
import pereggrn_perturbations
import io
from pandas.api.types import is_numeric_dtype

pereggrn_perturbations.set_data_path("../../perturbation_data/perturbations")
pereggrn_networks.set_grn_location("../../network_collection/networks")
train   = pereggrn_perturbations.load_perturbation("nakatake")
train.X = scipy.sparse.csr_matrix(train.X) 
network = pereggrn_networks.LightNetwork("celloracle_human", ["human_promoters.parquet"])
example_perturbations = pd.read_csv(io.StringIO(
"""perturbation expression_level_after_perturbation
"Control,KLF8" "0,0"
"GATA1" 0
"GATA1,GFP" "0,NAN"
"empty_vector" NAN"""
    ), 
    sep = " "
)
print(example_perturbations)
class TestModelRuns(unittest.TestCase):
    def test_make_GRN(self):
        self.assertIsInstance(
            ggrn.GRN(train, network = network), ggrn.GRN
        )
        self.assertIsInstance(
            ggrn.GRN(train), ggrn.GRN
        )
        grn = ggrn.GRN(train, network = network)

    def test_validate_input(self):
        self.assertTrue(
            ggrn.GRN(train, network = network).check_perturbation_dataset() 
        )    
        self.assertTrue(
            ggrn.GRN(train).check_perturbation_dataset() 
        )    

    def test_match_timeseries(self):
        grn = ggrn.GRN(train, network = network)
        train.obs["timepoint"] = np.random.randint(0, 10, len(train.obs))
        grn.train.obs_names = [str(i) for i in grn.train.obs_names]
        for is_int in [True, False]:
            for mm in ["steady_state", "user",  "closest", "user", "random", "user", "optimal_transport", "user"]:
                grn.train = ggrn.match_timeseries(grn.train, matching_method=mm, matched_control_is_integer=is_int) 
                has_control = grn.train.obs["matched_control"].notnull()
                # Everything that has a match should have a match at the same or an earlier timepoint.
                self.assertTrue(
                     all(grn.train.obs[grn.train.obs["matched_control"][has_control], "timepoint"] <= grn.train.obs["timepoint"][has_control])
                )

    def test_match_controls(self):
        grn = ggrn.GRN(train, network = network)
        grn.train.obs_names = [str(i) for i in grn.train.obs_names]
        for is_int in [True, False]:
            # We can include "user" in this list as long as it's not last. It will check a couple things and leave what's already there.
            # In fact those checks could be useful so we'll run it in user-specified matching mode once after every other mode.
            for mm in ["steady_state", "user",  "closest", "user", "random", "user", "optimal_transport", "user"]:
                grn.train = ggrn.match_controls(grn.train, matching_method = mm, matched_control_is_integer=is_int) 
                # make sure ouput is ints or strings as specified
                if is_int: 
                    self.assertTrue(
                        is_numeric_dtype(grn.train.obs["matched_control"])
                    )
                else:
                    self.assertTrue(
                        grn.train.obs["matched_control"].dtype == grn.train.obs_names.dtype
                    )
                # make sure the controls are matched
                if mm not in ["closest_with_unmatched_controls", "user"]:
                    self.assertTrue(
                        all(pd.notnull(grn.train.obs.loc[grn.train.obs["is_control"], "matched_control"]))
                    )
                # Make sure steady state is correct
                if mm == "steady_state" and not is_int:
                    self.assertTrue(
                        all(grn.train.obs["matched_control"]==grn.train.obs.index)
                    )
                if mm == "steady_state" and is_int:
                    self.assertTrue(
                        all(grn.train.obs["matched_control"]==range(len(grn.train.obs.index)))
                    )                
                # Make sure random and closest are matching to controls only
                if mm in ["random", "closest"] and not is_int:
                    self.assertTrue(
                        all(grn.train.obs.loc[grn.train.obs["matched_control"], "is_control"])
                    )
                if mm in ["random", "closest"] and is_int:
                    self.assertTrue(
                        all(grn.train.obs.iloc[grn.train.obs["matched_control"], :]["is_control"])
                    )

    def test_extract_features(self):
        grn    = ggrn.GRN(train, network = network)
        grn.train = ggrn.match_controls(grn.train, matching_method = "steady_state", matched_control_is_integer=False) 
        self.assertIsNone(
            grn.extract_features()
        )    
        grn    = ggrn.GRN(train)
        self.assertIsNone(
            grn.extract_features()
        )    

    def test_fit_various_methods(self):
        grn    = ggrn.GRN(train[0:100, 0:10].copy(), validate_immediately=False)
        for regression_method in [
                "mean", 
                "median", 
                "GradientBoostingRegressor",
                "ExtraTreesRegressor",
                "KernelRidge",
                "RidgeCV", 
                "RidgeCVExtraPenalty",
                "LassoCV",
                "ElasticNetCV",
                "LarsCV",
                "LassoLarsIC",
                "OrthogonalMatchingPursuitCV",
                "ARDRegression",
                "BayesianRidge",
                # "BernoulliRBM",
                # "MLPRegressor",
            ]:
            grn.fit(
                method = regression_method, 
                cell_type_sharing_strategy = "identical", 
                network_prior = "ignore",
                pruning_strategy = "none", 
                pruning_parameter = None, 
                do_parallel=False, #catch bugs easier
            )
            x1 = grn.predict(predictions_metadata = example_perturbations)  
            initial_state = train[0:len(example_perturbations), 0:10].copy()
            initial_state.obs = example_perturbations
            x2 = grn.predict(predictions = initial_state)
    
    def test_simple_fit_and_predict(self):
        grn    = ggrn.GRN(train[0:100, 0:100].copy(), validate_immediately=False)
        grn.fit(
            method = "RidgeCVExtraPenalty", 
            cell_type_sharing_strategy = "identical", 
            network_prior = "ignore",
            pruning_strategy = "none", 
            pruning_parameter = None, 
        )
        grn.fit(
            method = "RidgeCVExtraPenalty", 
            cell_type_sharing_strategy = "identical", 
            network_prior = "ignore",
            pruning_strategy = "none", 
            pruning_parameter = None, 
            do_parallel=False,
        )
        shutil.rmtree("tempfolder23587623", ignore_errors=True)
        self.assertTrue(grn.save_models("tempfolder23587623") is None)
        shutil.rmtree("tempfolder23587623")
        p = grn.predict(predictions_metadata = example_perturbations)
        self.assertIsInstance(
            p, anndata.AnnData
        )

    def test_network_fit_and_predict(self):
        grn    = ggrn.GRN(train[0:100, 0:100].copy(), network = network, validate_immediately=False)
        grn.fit(
            method = "RidgeCVExtraPenalty", 
            cell_type_sharing_strategy = "identical", 
            network_prior = "restrictive", 
            pruning_strategy = "none", 
            pruning_parameter = None, 
        )
        p = grn.predict(predictions_metadata = example_perturbations)
        self.assertIsInstance(
            p, anndata.AnnData
        )

    def test_pruned_fit_and_predict(self):
        grn    = ggrn.GRN(train[0:100, 0:100].copy(), validate_immediately=False)
        grn.fit(
            method = "RidgeCVExtraPenalty", 
            cell_type_sharing_strategy = "identical", 
            network_prior = "ignore", 
            pruning_strategy = "prune_and_refit",
            pruning_parameter = 5000, 
        )
        p = grn.predict(predictions_metadata = example_perturbations, do_parallel=False)
        self.assertIsInstance(
            p, anndata.AnnData
        )



# Make some simulated data that ridge regressions should be able to get exactly right
easy_simulated = anndata.AnnData(
    dtype=np.float32,
    obs = pd.DataFrame(
        {"is_control": True,
        "perturbation": "control",
        "perturbation_type": "overexpression",
        "expression_level_after_perturbation": 1,
        "cell_type": np.resize([1,2], 1000)},
        index = [str(i) for i in range(1000)],
    ),
    var = pd.DataFrame(
        index = ["g" + str(i) for i in range(5)],
    ),
    X = np.random.rand(1000, 5) - 0.5
)
# One gene is a linear function
easy_simulated.X[:,0] = easy_simulated.X[:,1]*2 
# Another is a cell-type-specific linear function
easy_simulated.X[easy_simulated.obs["cell_type"]==1,4] = easy_simulated.X[easy_simulated.obs["cell_type"]==1,2]*2
easy_simulated.X[easy_simulated.obs["cell_type"]==2,4] = easy_simulated.X[easy_simulated.obs["cell_type"]==2,3]*2
# empty network should fit worse than correct network
# correct network should fit roughly same as dense network
empty_network   = pereggrn_networks.LightNetwork(df = pd.DataFrame(index=[], columns=["regulator", "target", "weight"]))
correct_network = pereggrn_networks.LightNetwork(df = pd.DataFrame(
    {
        "regulator":["g" + str(i) for i in [1, 2, 3]], 
        "target":   ["g" + str(i) for i in [0, 4, 4]], 
        "weight":1,
    }
    ))
dense_network = pereggrn_networks.LightNetwork(df = pd.DataFrame({
    "regulator": ["g" + str(i) for _ in range(5) for i in range(5)],
    "target":    ["g" + str(i) for i in range(5) for _ in range(5)],
    "weight": 1,
    }))

class TestModelExactlyRightOnEasySimulation(unittest.TestCase):
    def test_simple_dense(self):
        grn = ggrn.GRN(easy_simulated, validate_immediately=False, eligible_regulators=easy_simulated.var_names)
        grn.fit(
            method = "RidgeCVExtraPenalty", 
            cell_type_sharing_strategy = "identical", 
            network_prior = "ignore", 
            pruning_strategy = "none",
        )
        p = grn.predict(predictions_metadata = pd.DataFrame([("g1", 0), ("g1", 1), ("g1", 2)], columns = ["perturbation", "expression_level_after_perturbation"]), do_parallel=False)
        np.testing.assert_almost_equal(p[:, "g0"].X.squeeze(), [0,2,4], decimal=1)

    def test_simple_dense_parallel_predict(self):
        grn = ggrn.GRN(easy_simulated, validate_immediately=False, eligible_regulators=easy_simulated.var_names)
        grn.fit(
            method = "RidgeCVExtraPenalty", 
            cell_type_sharing_strategy = "identical", 
            network_prior = "ignore", 
            pruning_strategy = "none",
        )
        p = grn.predict(predictions_metadata = pd.DataFrame([("g1", 0), ("g1", 1), ("g1", 2)], columns = ["perturbation", "expression_level_after_perturbation"]), do_parallel=True)
        np.testing.assert_almost_equal(p[:, "g0"].X.squeeze(), [0,2,4], decimal=1)

    def test_network_dense(self):
        grn = ggrn.GRN(easy_simulated, network = dense_network, validate_immediately=False, eligible_regulators=easy_simulated.var_names)
        grn.fit(
            method = "RidgeCVExtraPenalty", 
            cell_type_sharing_strategy = "identical", 
            network_prior = "restrictive",
            pruning_strategy = "none"
        )
        p = grn.predict(predictions_metadata = pd.DataFrame([("g1", 0), ("g1", 1), ("g1", 2)], columns = ["perturbation", "expression_level_after_perturbation"]), do_parallel=True)
        np.testing.assert_almost_equal(p[:, "g0"].X.squeeze(), [0,2,4], decimal=1)

    def test_network_correct(self):
        grn = ggrn.GRN(easy_simulated, network = correct_network, validate_immediately=False, eligible_regulators=easy_simulated.var_names)
        grn.fit(
            method = "RidgeCVExtraPenalty", 
            cell_type_sharing_strategy = "identical", 
            network_prior = "restrictive",
            pruning_strategy = "none",
        )
        p = grn.predict(predictions_metadata = pd.DataFrame([("g1", 0), ("g1", 1), ("g1", 2)], columns = ["perturbation", "expression_level_after_perturbation"]), do_parallel=True)
        np.testing.assert_almost_equal(p[:, "g0"].X.squeeze(), [0,2,4], decimal=1)

    def test_network_empty(self):
        grn = ggrn.GRN(easy_simulated, network = empty_network, validate_immediately=False, eligible_regulators=easy_simulated.var_names)
        grn.fit(
            method = "RidgeCVExtraPenalty", 
            cell_type_sharing_strategy = "identical", 
            network_prior = "restrictive",
            pruning_strategy = "none",
        )
        p = grn.predict(predictions_metadata = pd.DataFrame([("g1", 0), ("g1", 1), ("g1", 2)], columns = ["perturbation", "expression_level_after_perturbation"]), do_parallel=True)
        np.testing.assert_almost_equal(p[:, "g0"].X.squeeze(), [0,0,0], decimal=1)

    def test_ct_specific(self):
        grn = ggrn.GRN(easy_simulated, validate_immediately=False, eligible_regulators=easy_simulated.var_names)
        grn.fit(
            method = "RidgeCVExtraPenalty", 
            cell_type_sharing_strategy = "distinct", 
            cell_type_labels='cell_type',
            network_prior = "ignore",
            pruning_strategy = "none",
        )
        
        for dp in [False, True]:
            # Cell type 1: only gene 2 affects gene 4
            dummy_control = easy_simulated[0:3,:].copy()
            dummy_control.X = dummy_control.X*0
            dummy_control.obs["cell_type"] = 1
            initial_state = dummy_control.copy()
            initial_state.obs["perturbation"] = "g2"
            initial_state.obs["expression_level_after_perturbation"] = [0,1,2]
            p = grn.predict(
                predictions=initial_state, 
                do_parallel=dp
            )
            np.testing.assert_almost_equal(p[:,"g4"].X.squeeze(), [0,2,4], decimal=1)
            initial_state = dummy_control.copy()
            initial_state.obs.merge(pd.DataFrame([("g3", 0), ("g3", 1), ("g2", 2)], columns = ["perturbation", "expression_level_after_perturbation"]))
            p = grn.predict(
                predictions=initial_state, 
                do_parallel=dp
            )

            np.testing.assert_almost_equal(p[:,"g4"].X.squeeze(), [0,0,0], decimal=1)
            # Cell type 2: only gene 3 affects gene 4
            dummy_control = easy_simulated[0:3,:].copy()
            dummy_control.X = dummy_control.X*0
            dummy_control.obs["cell_type"] = 2
            initial_state = dummy_control.copy()
            initial_state.obs["perturbation"] = "g2"
            initial_state.obs["expression_level_after_perturbation"] = [0,1,2]
            p = grn.predict(
                predictions=initial_state, 
                do_parallel=dp
            )
            np.testing.assert_almost_equal(p[:,"g4"].X.squeeze(), [0,0,0], decimal=1)
            initial_state = dummy_control.copy()
            initial_state.obs["perturbation"] = "g3"
            initial_state.obs["expression_level_after_perturbation"] = [0,1,2]
            p = grn.predict(
                predictions=initial_state, 
                do_parallel=dp
            )
            np.testing.assert_almost_equal(p[:,"g4"].X.squeeze(), [0,2,4], decimal=1)


if __name__ == '__main__':
    unittest.main()
