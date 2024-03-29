
import os
import numpy as np
import pandas as pd
import anndata
import torch
import unittest
import pereggrn_perturbations
import ggrn.api as ggrn
import ggrn_backend2.api as dcdfg_wrapper

pereggrn_perturbations.set_data_path(
    '../../perturbation_data/perturbations'
)

test_data = pereggrn_perturbations.load_perturbation("dcdfg_test")
test_data.X = test_data.X.toarray()

class TestDCDFG(unittest.TestCase):
    
    os.environ['WANDB_MODE'] = 'offline'
    os.environ["WANDB_SILENT"] = "true"

    def test_CV_run(self):
        factor_graph_cv = dcdfg_wrapper.DCDFGCV()
        factor_graph_cv.train(adata=test_data, num_train_epochs = 50, num_fine_epochs = 50)
        assert type(factor_graph_cv.final_model) == dcdfg_wrapper.DCDFGWrapper

    def test_NOTEARS_run(self):
        grn = ggrn.GRN(train=test_data, eligible_regulators=test_data.var_names, validate_immediately=False)
        grn.fit(
            method = "DCDFG-exp-linear-False",
            cell_type_sharing_strategy = "identical",
            network_prior = "ignore",
            kwargs = { 
                "num_train_epochs": 2, 
                "num_fine_epochs": 1,
                "num_gpus": 1 if torch.cuda.is_available() else 0,
                "train_batch_size": 64,
                "verbose": False,
                }
        )
        
    def test_NOTEARSLR_run(self):
        grn = ggrn.GRN(train=test_data, eligible_regulators=test_data.var_names, validate_immediately=False)
        grn.fit(
            method = "DCDFG-spectral_radius-linearlr-False",
            cell_type_sharing_strategy = "identical",
            network_prior = "ignore",
            kwargs = { 
                "num_train_epochs": 20, # This fails with 2 epochs 
                "num_fine_epochs": 1,
                "num_gpus": 1 if torch.cuda.is_available() else 0,
                "train_batch_size": 64,
                "verbose": False,
                }
        )
        
    def test_NOBEARS_run(self):
        grn = ggrn.GRN(train=test_data, eligible_regulators=test_data.var_names, validate_immediately=False)
        grn.fit(
            method = "DCDFG-exp-linear-True",
            cell_type_sharing_strategy = "identical",
            network_prior = "ignore",
            kwargs = { 
                "num_train_epochs": 2, 
                "num_fine_epochs": 1,
                "num_gpus": 1 if torch.cuda.is_available() else 0,
                "train_batch_size": 64,
                "verbose": False,
                }
        )  
        
    def test_DCDFG_run(self):
        grn = ggrn.GRN(train=test_data, eligible_regulators=test_data.var_names, validate_immediately=False)
        grn.fit(
            method = "DCDFG-spectral_radius-mlplr-False",
            cell_type_sharing_strategy = "identical",
            network_prior = "ignore",
            kwargs = { 
                "num_train_epochs": 2, 
                "num_fine_epochs": 1,
                "num_gpus": 1 if torch.cuda.is_available() else 0,
                "train_batch_size": 64,
                "verbose": False,
                "num_modules": 4,
            }
        )
    

    def test_NOTEARS_prediction(self):
        grn = ggrn.GRN(train=test_data, eligible_regulators=test_data.var_names, validate_immediately=False)
        grn.fit(
            method = "DCDFG-spectral_radius-linear-False", 
            kwargs = { 
                "num_train_epochs": 2, 
                "num_fine_epochs": 1,
                "num_gpus": 1 if torch.cuda.is_available() else 0,
                "train_batch_size": 64,
                "verbose": False,
                }
        )

        _ = grn.models.model.module.load_state_dict(
            {
                "weights": torch.nn.Parameter(torch.tensor(np.array(
                [[0.0,   1, 0, 0, 0],
                [0,     0, 1, 0, 0],
                [0,     0, 0, 1, 0],
                [0,     0, 0, 0, 1],
                [0,     0, 0, 0, 0]])), requires_grad=False)
            },  
            strict=False)

        grn.models.model.module.biases = torch.nn.Parameter(torch.tensor(np.array(
            [5.0,    0, 0, 0, 0])), requires_grad=False)
        grn.models.model.module.weight_mask = torch.nn.Parameter(torch.tensor(np.ones((5,5))))
        grn.models.model.module.weights = torch.nn.Parameter(torch.tensor(np.array(
            [[0.0,   1, 0, 0, 0],
             [0,     0, 1, 0, 0],
             [0,     0, 0, 1, 0],
             [0,     0, 0, 0, 1],
             [0,     0, 0, 0, 0]])), requires_grad=False)
        control = test_data.X[-1,:]
        koAnswer2 = grn.predict(
            predictions = anndata.AnnData(
                X=np.tile(control.copy(), (7,1)), 
                var = pd.DataFrame(index=[f'Gene{i}' for i in range(5)]),
            ),
            perturbations = [(f'Gene{i}',1000) for i in range(5)] + [("control", 0)] + [("Gene0", np.nan)]
        )
        np.testing.assert_almost_equal(
            koAnswer2.X / np.array([
                [1000, 1000, 5, 5, 5],
                [5, 1000, 1000, 5, 5],
                [5, 5, 1000, 1000, 5],
                [5, 5, 5, 1000, 1000],
                [5, 5, 5,    5, 1000],
                [5, 5, 5,    5,    5],
                [5, 5, 5,    5,    5],
            ]),
            np.ones((7,5)),
            decimal=2
        )

        koAnswer3 = grn.predict(
            perturbations = [(f'Gene{i}',1000) for i in range(5)] + [("control", 0)] + [("Gene0", np.nan)]
        )
        np.testing.assert_almost_equal(
            koAnswer3.X / np.array([
                [1000, 1000, 5, 5, 5],
                [5, 1000, 1000, 5, 5],
                [5, 5, 1000, 1000, 5],
                [5, 5, 5, 1000, 1000],
                [5, 5, 5,    5, 1000],
                [5, 5, 5,    5,    5],
                [5, 5, 5,    5,    5],
            ]),
            np.ones((7,5)),
            decimal=2
        )


    def test_NOTEARS_inference(self):
        grn = ggrn.GRN(train=test_data, eligible_regulators=test_data.var_names, validate_immediately=False)
        grn.fit(
            method = "DCDFG-exp-linear-False",
            # method = "DCDFG-spectral_radius-mlplr-False",
            cell_type_sharing_strategy = "identical",
            network_prior = "ignore",
            kwargs = { 
                "num_train_epochs": 600, 
                "num_fine_epochs": 100,
                "num_gpus": [1] if torch.cuda.is_available() else 0,
                "train_batch_size": 64,
                "verbose": False,
            }
        )
        posWeight = (grn.models.model.module.weights * grn.models.model.module.weight_mask).detach().cpu().numpy()
        posWeightAnswer = np.array(
            [[0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0]])
        bias = grn.models.model.module.biases.detach().cpu().numpy()
        biasAnswer = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_almost_equal(posWeight, posWeightAnswer, decimal=2)
        np.testing.assert_almost_equal(bias, biasAnswer, decimal=2)


if __name__ == '__main__':
    unittest.main()
