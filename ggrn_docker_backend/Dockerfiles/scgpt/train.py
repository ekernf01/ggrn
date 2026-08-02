"""
GGRN docker backend for scGPT (https://github.com/bowang-lab/scgpt).

This wires the standard GGRN docker I/O contract to scGPT's GEARS-style perturbation-prediction
task. Everything here was written against the real scGPT source, in particular:

  * tutorials/Tutorial_Perturbation.ipynb  -- the end-to-end fine-tune + predict workflow.
        - model construction: scgpt.model.TransformerGenerator(...)
        - inference:          model.pred_perturb(batch, include_zero_gene, gene_ids, amp)  and the
                              tutorial `predict(model, pert_list, pool_size)` helper, which calls
                              gears.utils.create_cell_graph_dataset_for_prediction(...).
  * scgpt/model/generation_model.py  -- TransformerGenerator.pred_perturb returns a [N, n_genes]
        tensor of predicted expression over the model's full gene set.
  * scgpt/utils/util.py  -- map_raw_id_to_vocab_id, load_pretrained (handles the flash-attn -> vanilla
        MultiheadAttention key rename so the pretrained weights load with use_fast_transformer=False).
  * scgpt/tokenizer/gene_tokenizer.py  -- GeneVocab.from_file(vocab.json).
  * The pretrained "whole-human" checkpoint folder (best_model.pt / args.json / vocab.json) is a real
        public artifact on Google Drive (folder id 1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y).

IMPORTANT SEMANTIC NOTES (see README.md):
  * scGPT perturbation is a knockdown/overexpression *flag* model. The perturbation is encoded only
    as a per-gene flag; the GGRN `expression_level_after_perturbation` *magnitude* is NOT used by the
    model. We still read it so we can identify controls and label the output rows.
  * Perturbed genes must exist in the training/vocab gene set. Perturbations naming genes that are not
    measured are reported via `error_message` and predicted as the control mean.
  * scGPT does not use a base GRN, timeseries structure, or most GGRN network/regression args. See README.
"""

import os
import json
import time
import copy
import traceback
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import torch

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------------------
# 0. Config / defaults
# --------------------------------------------------------------------------------------------------
OUT_DIR = "from_to_docker"
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
PAD_VALUE = 0
PERT_PAD_ID = 0
INCLUDE_ZERO_GENE = "all"
MAX_SEQ_LEN = 1536

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"scGPT backend running on device: {device}")
if device.type != "cuda":
    print("WARNING: no CUDA device detected. scGPT is a transformer and will be extremely slow on CPU. "
          "Run the container with `--gpus all`.")

# --------------------------------------------------------------------------------------------------
# 1. Read the GGRN inputs
# --------------------------------------------------------------------------------------------------
train = sc.read_h5ad(os.path.join(OUT_DIR, "train.h5ad"))
predictions_metadata = pd.read_csv(os.path.join(OUT_DIR, "predictions_metadata.csv"), index_col=0)
assert predictions_metadata.shape[0] > 0, "predictions_metadata is empty."
for c in ["perturbation", "expression_level_after_perturbation", "is_control"]:
    assert c in predictions_metadata.columns, f"For the ggrn scgpt backend, predictions_metadata must contain column {c}."

with open(os.path.join(OUT_DIR, "kwargs.json")) as f:
    kwargs = json.load(f)
with open(os.path.join(OUT_DIR, "ggrn_args.json")) as f:
    ggrn_args = json.load(f)

# scGPT does not use a base network, but read it if present so behaviour matches the template/celloracle
# convention of gracefully handling from_to_docker/network.parquet.
try:
    from pereggrn_networks import LightNetwork
    _ = LightNetwork(files=[os.path.join(OUT_DIR, "network.parquet")])
    print("A base network was provided but scGPT does not use one; ignoring it.")
except Exception:
    pass

# Method-specific kwargs (all optional).
defaults = {
    "epochs": 15,                # fine-tuning epochs on the user's data. Set to 0 for zero-shot.
    "batch_size": 64,
    "lr": 1e-4,
    "pool_size": 300,            # # of control cells sampled per perturbation at prediction time
    "load_model": os.environ.get("SCGPT_MODEL_DIR", "/scgpt_weights/scGPT_human"),
    "gdrive_folder_id": "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",  # "whole-human" checkpoint folder
    "early_stop": 5,
}
for k, v in defaults.items():
    if k not in kwargs or kwargs[k] is None:
        kwargs[k] = v

EPOCHS = int(kwargs["epochs"])
BATCH_SIZE = int(kwargs["batch_size"])
POOL_SIZE = int(kwargs["pool_size"])

# --------------------------------------------------------------------------------------------------
# 2. Obtain the pretrained checkpoint (best_model.pt / args.json / vocab.json)
# --------------------------------------------------------------------------------------------------
model_dir = kwargs["load_model"]
needed = ["best_model.pt", "args.json", "vocab.json"]
have_ckpt = model_dir and all(os.path.exists(os.path.join(model_dir, f)) for f in needed)
if not have_ckpt:
    print(f"Pretrained scGPT checkpoint not found at {model_dir}; attempting to download the "
          f"'whole-human' folder from Google Drive with gdown.")
    try:
        import gdown
        os.makedirs("/scgpt_weights", exist_ok=True)
        model_dir = "/scgpt_weights/scGPT_human"
        gdown.download_folder(id=kwargs["gdrive_folder_id"], output=model_dir, quiet=False, use_cookies=False)
        have_ckpt = all(os.path.exists(os.path.join(model_dir, f)) for f in needed)
    except Exception as e:
        print(f"Could not download pretrained weights automatically: {repr(e)}")
        have_ckpt = False
if not have_ckpt:
    raise FileNotFoundError(
        "scGPT requires the pretrained 'whole-human' checkpoint (best_model.pt, args.json, vocab.json). "
        "Either bake it into the image (see the commented gdown block in the Dockerfile), mount it, or "
        "set the SCGPT_MODEL_DIR env var / kwargs['load_model'] to a folder containing it. Download from "
        "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
    )

# --------------------------------------------------------------------------------------------------
# 3. scGPT / GEARS imports (kept after arg-parsing so config errors surface fast)
# --------------------------------------------------------------------------------------------------
from torch import nn
from torch_geometric.loader import DataLoader
from gears import PertData
from gears.utils import create_cell_graph_dataset_for_prediction

from scgpt.model import TransformerGenerator
from scgpt.loss import masked_mse_loss
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

set_seed(42)

# --------------------------------------------------------------------------------------------------
# 4. Build vocab + gene id mapping from the checkpoint, restricted to the training genes
# --------------------------------------------------------------------------------------------------
vocab = GeneVocab.from_file(os.path.join(model_dir, "vocab.json"))
for s in SPECIAL_TOKENS:
    if s not in vocab:
        vocab.append_token(s)
vocab.set_default_index(vocab[PAD_TOKEN])

genes = list(train.var_names)                      # gene symbols in the training AnnData
n_genes = len(genes)
gene_ids = np.array([vocab[g] if g in vocab else vocab[PAD_TOKEN] for g in genes], dtype=int)
in_vocab = np.array([g in vocab for g in genes])
print(f"{in_vocab.sum()}/{n_genes} training genes are in the scGPT vocabulary.")

with open(os.path.join(model_dir, "args.json")) as f:
    model_configs = json.load(f)
embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]

# --------------------------------------------------------------------------------------------------
# 5. Construct the model and load pretrained weights (vanilla attention, no flash-attn)
# --------------------------------------------------------------------------------------------------
ntokens = len(vocab)
model = TransformerGenerator(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=1,
    vocab=vocab,
    dropout=0.0,
    pad_token=PAD_TOKEN,
    pad_value=PAD_VALUE,
    pert_pad_id=PERT_PAD_ID,
    use_fast_transformer=False,   # avoid the flash-attn dependency; load_pretrained renames keys
)
# Load only the parameter groups that are shape-compatible, mirroring the tutorial's fallback path.
model_dict = model.state_dict()
pretrained_dict = torch.load(os.path.join(model_dir, "best_model.pt"), map_location="cpu")
try:
    from scgpt.utils import load_pretrained
    model = load_pretrained(model, pretrained_dict, verbose=False)
except Exception as e:
    print(f"load_pretrained fell back to manual shape-matched loading: {repr(e)}")
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
model.to(device)

# --------------------------------------------------------------------------------------------------
# 6. Build a GEARS PertData object from the GGRN training data and (optionally) fine-tune
# --------------------------------------------------------------------------------------------------
def ggrn_condition(row_pert, row_is_control):
    """Map GGRN (perturbation, is_control) to a GEARS condition string, e.g. 'GENE+ctrl' or 'A+B'."""
    if bool(row_is_control):
        return "ctrl"
    parts = [g for g in str(row_pert).split(",") if g in set(genes)]
    if len(parts) == 0:
        return "ctrl"       # perturbed gene(s) not measured -> treat as control for training
    return "+".join(parts + (["ctrl"] if len(parts) == 1 else []))

pert_data = None
finetuned = False
if EPOCHS > 0:
    try:
        ft = train.copy()
        # GEARS requires raw-ish counts in .X, plus 'condition', 'cell_type' obs and 'gene_name' var.
        if "cell_type" not in ft.obs.columns:
            ft.obs["cell_type"] = "unknown"
        ft.obs["cell_type"] = ft.obs["cell_type"].astype(str)
        ic = ft.obs["is_control"] if "is_control" in ft.obs.columns else np.zeros(ft.n_obs, dtype=bool)
        ft.obs["condition"] = [ggrn_condition(p, c) for p, c in zip(ft.obs.get("perturbation", ""), ic)]
        ft.var["gene_name"] = list(ft.var_names)

        assert (ft.obs["condition"] == "ctrl").sum() > 0, "GEARS fine-tuning needs control cells."
        n_pert_conditions = ft.obs["condition"].nunique() - 1
        assert n_pert_conditions >= 1, "GEARS fine-tuning needs at least one non-control condition."

        pert_data = PertData(os.path.join(OUT_DIR, "gears_data"))
        pert_data.new_data_process(dataset_name="ggrn_scgpt", adata=ft)
        # 'no_test' keeps everything for train/val (we do not need a held-out test split here).
        pert_data.prepare_split(split="no_test", seed=1)
        pert_data.get_dataloader(batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE)

        criterion = masked_mse_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=float(kwargs["lr"]))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        amp_enabled = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        def train_one_epoch(loader):
            model.train()
            for batch_data in loader:
                batch_data.to(device)
                bs = len(batch_data.y)
                x = batch_data.x
                ori_gene_values = x[:, 0].view(bs, n_genes)
                pert_flags = x[:, 1].long().view(bs, n_genes)
                target_gene_values = batch_data.y
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
                if len(input_gene_ids) > MAX_SEQ_LEN:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[:MAX_SEQ_LEN]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]
                mapped = map_raw_id_to_vocab_id(input_gene_ids, gene_ids).repeat(bs, 1)
                mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    out = model(mapped, input_values, input_pert_flags,
                                src_key_padding_mask=mask, CLS=False, CCE=False, MVC=False, ECS=False)
                    loss = criterion(out["mlm_output"], target_values,
                                     torch.ones_like(input_values, dtype=torch.bool))
                model.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()

        print(f"Fine-tuning scGPT for {EPOCHS} epochs on {n_pert_conditions} perturbation conditions.")
        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_one_epoch(pert_data.dataloader["train_loader"])
            scheduler.step()
            print(f"  epoch {epoch}/{EPOCHS} done in {time.time()-t0:.1f}s")
        finetuned = True
    except Exception as e:
        print("Fine-tuning failed; falling back to zero-shot pretrained inference. Reason:")
        traceback.print_exc()
        pert_data = None
        finetuned = False

if pert_data is None:
    # Zero-shot: we still need a control AnnData and gene_names for the predictor. Build a minimal one.
    class _MiniPertData:
        pass
    pert_data = _MiniPertData()
    ctrl_mask = (train.obs["is_control"].values.astype(bool)
                 if "is_control" in train.obs.columns else np.ones(train.n_obs, dtype=bool))
    ad = train.copy()
    ad.var["gene_name"] = list(ad.var_names)
    pert_data.adata = ad
    pert_data.gene_names = pd.Series(list(ad.var_names), index=list(ad.var_names))
    # Mark controls with GEARS 'condition' so the predictor can select them.
    cond = np.where(ctrl_mask, "ctrl", "perturbed")
    pert_data.adata.obs["condition"] = cond

# --------------------------------------------------------------------------------------------------
# 7. Prediction helper (tutorial `predict()` path) -> {joined_pert_name: mean_expr_vector}
# --------------------------------------------------------------------------------------------------
gene_list = list(pert_data.gene_names.values) if hasattr(pert_data.gene_names, "values") else list(pert_data.gene_names)
ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
if ctrl_adata.n_obs == 0:
    ctrl_adata = pert_data.adata  # fall back to using all cells as the control pool
control_mean = np.asarray(ctrl_adata.X.mean(axis=0)).ravel()

def predict_perturbation(pert_genes):
    """pert_genes: list of gene symbols. Returns a length-n_genes predicted expression vector."""
    model.eval()
    with torch.no_grad():
        cell_graphs = create_cell_graph_dataset_for_prediction(
            pert_genes, ctrl_adata, gene_list, device, num_samples=min(POOL_SIZE, ctrl_adata.n_obs)
        )
        loader = DataLoader(cell_graphs, batch_size=BATCH_SIZE, shuffle=False)
        preds = []
        for batch_data in loader:
            preds.append(model.pred_perturb(batch_data, INCLUDE_ZERO_GENE, gene_ids=gene_ids, amp=(device.type == "cuda")))
        preds = torch.cat(preds, dim=0)
    return np.mean(preds.detach().cpu().numpy(), axis=0)

# --------------------------------------------------------------------------------------------------
# 8. Loop over requested predictions, filling the output AnnData
# --------------------------------------------------------------------------------------------------
predictions = anndata.AnnData(
    X=np.zeros((predictions_metadata.shape[0], n_genes), dtype=np.float32),
    obs=predictions_metadata.copy(),
    var=train.var.copy(),
)
predictions.obs["error_message"] = "no errors"
gene_set = set(gene_list)

group_cols = [c for c in ["perturbation", "expression_level_after_perturbation", "is_control"]
              if c in predictions.obs.columns]
for _, row in predictions.obs[group_cols].drop_duplicates().iterrows():
    sel = np.ones(predictions.n_obs, dtype=bool)
    for c in group_cols:
        v = row[c]
        sel &= (predictions.obs[c].isna().values if pd.isna(v) else (predictions.obs[c] == v).values)
    idx = np.where(sel)[0]
    if len(idx) == 0:
        continue
    try:
        if bool(row.get("is_control", False)):
            pred_vec = control_mean
        else:
            pert_genes = [g for g in str(row["perturbation"]).split(",")]
            missing = [g for g in pert_genes if g not in gene_set]
            if missing:
                raise ValueError(f"Perturbed gene(s) {missing} not in the measured/vocab gene set; "
                                 f"scGPT cannot predict this perturbation.")
            pred_vec = predict_perturbation(pert_genes)
        for i in idx:
            predictions.X[i, :] = pred_vec
    except Exception as e:
        for i in idx:
            predictions.X[i, :] = np.nan
        predictions.obs.iloc[idx, predictions.obs.columns.get_loc("error_message")] = repr(e)

print("Fine-tuned:" , finetuned)
print("Summary of error messages encountered during prediction:")
print(predictions.obs["error_message"].value_counts())
print("Saving results.")
predictions.write_h5ad(os.path.join(OUT_DIR, "predictions.h5ad"))
