# scGPT GGRN backend

A Docker container to let GGRN interface to [scGPT](https://github.com/bowang-lab/scgpt)'s
perturbation-prediction task.

scGPT is a large single-cell transformer. Its perturbation workflow (from
`tutorials/Tutorial_Perturbation.ipynb`) is GEARS-style: starting from the pretrained "whole-human"
checkpoint, it (optionally) fine-tunes `scgpt.model.TransformerGenerator` on the user's perturbation
data, then predicts the post-perturbation single-cell expression profile of control cells. The
perturbation is encoded as a per-gene *flag* (perturbed / not) rather than a target expression level.

## What this backend actually does

1. Reads `train.h5ad`, `predictions_metadata.csv`, `kwargs.json`, `ggrn_args.json` (and ignores
   `network.parquet` if present).
2. Loads the pretrained scGPT checkpoint (`best_model.pt` / `args.json` / `vocab.json`), restricting
   the gene set to the training genes and mapping them to the scGPT vocabulary.
3. Builds a GEARS `PertData` object from the GGRN training cells (mapping GGRN's `perturbation` /
   `is_control` to GEARS `condition` strings like `GENE+ctrl`) and fine-tunes for `epochs` epochs.
   If fine-tuning is not possible (e.g. no controls, no perturbation conditions, or a runtime error),
   it falls back to zero-shot inference with the pretrained weights.
4. For each requested perturbation, samples control cells and calls `model.pred_perturb(...)`
   (via `gears.utils.create_cell_graph_dataset_for_prediction`), averaging over the control pool.
5. Writes `predictions.h5ad` with `obs = predictions_metadata`, `var = train.var`, `X` = predictions.

## Requirements / constraints

- **GPU**: scGPT is a PyTorch transformer. CUDA is effectively required at runtime; run the container
  with `--gpus all`. It will run on CPU but is impractically slow.
- **Pretrained weights**: the ~1.5GB "whole-human" checkpoint folder
  (<https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y>) must be available. Options:
  (a) uncomment the `gdown` block in the `Dockerfile` to bake it in at build time; (b) mount a persistent
  host dir and set `SCGPT_MODEL_DIR`; or (c) let `train.py` download it on first run with `gdown` (needs
  network access). See **Caching the weights** below for how to make (c) download only once.

### Caching the weights

`train.py` downloads the checkpoint **into `kwargs['load_model']` / `SCGPT_MODEL_DIR`** (default
`/scgpt_weights/scGPT_human`) and **only if it is not already present**. So caching just means pointing
that directory at a location that survives between container runs.

GGRN itself only bind-mounts `from_to_docker/` — and it deletes and recreates that directory on every
`fit()` — so the weights *cannot* be cached there. Instead, pass an extra bind mount to `docker run`
through the method string, whose format is `docker__<args>__<image>`:

```
docker__--gpus all -v $HOME/.cache/scgpt_weights:/scgpt_weights__ekernf01/ggrn_docker_backend_scgpt
```

Because the container's default checkpoint dir is `/scgpt_weights/scGPT_human`, mounting a host cache at
`/scgpt_weights` means the first run downloads into `$HOME/.cache/scgpt_weights/scGPT_human` on the host
and every later run finds it already there and skips the download. (`--gpus all` is included because
scGPT needs a GPU anyway.) See `test.py` for a runnable example and `build.sh` for the interactive form.
For a zero-runtime-download alternative, bake the weights into the image via the commented `gdown` block
in the `Dockerfile`.
- **Gene vocabulary**: only genes present in both `train.var_names` and the scGPT vocab are usable.
  Perturbations naming a gene that is not measured are reported in `obs["error_message"]` and predicted
  as the control mean.
- **No flash-attn**: we run with `use_fast_transformer=False`. `scgpt.utils.load_pretrained` renames the
  flash-attn checkpoint keys to vanilla `nn.MultiheadAttention` keys, so the pretrained weights load fine.

## How the standard GGRN inputs are used

- `matching_method`: ignored (scGPT predicts from control cells, GEARS-style).
- `do_perturbations_persist`: ignored.
- `prediction_timescale`: ignored (scGPT produces a single post-perturbation profile; there is no
  propagation/time-stepping notion as in CellOracle).
- `regression_method`: ignored (scGPT is a fixed transformer architecture).
- `eligible_regulators`: ignored (any gene in the vocab can be a perturbation target).
- `predict_self`: ignored.
- `feature_extraction`: ignored (fixed to scGPT's gene-token + value-binning representation).
- `low_dimensional_structure` / `low_dimensional_training`: ignored (scGPT has its own latent).
- `pruning_strategy` / `pruning_parameter`: ignored (no explicit GRN pruning step).
- `network`: **ignored** — scGPT does not use a base GRN. A provided `network.parquet` is read but
  discarded (so that data collections carrying a network still run).
- `network_prior`: ignored.
- `cell_type_sharing_strategy`: ignored (a single shared model is fine-tuned across all cells).
- `expression_level_after_perturbation`: **not used as a magnitude**. scGPT only takes a perturbation
  flag, so the requested post-perturbation level cannot be honored. It is still read to identify
  controls and to label output rows.

## Method-specific `kwargs`

- `epochs` (default 15): fine-tuning epochs on the user's data. Set to `0` for pure zero-shot prediction.
- `batch_size` (default 64)
- `lr` (default 1e-4)
- `pool_size` (default 300): number of control cells sampled per perturbation at prediction time.
- `load_model` (default `/scgpt_weights/scGPT_human`, or `SCGPT_MODEL_DIR`): checkpoint folder.
- `gdrive_folder_id` (default the whole-human folder id): used if the checkpoint must be downloaded.
- `early_stop` (default 5): reserved for a validation-based early stop (currently we train a fixed
  number of epochs; see TODO).

## Files

- `train.py`: runs inside the container (the real implementation described above).
- `Dockerfile`: CUDA + conda base, installs scGPT + gears + torch-geometric + pereggrn_networks.
- `test.py`: makes GGRN use the container to make predictions (nakatake dataset).
- `build.sh`: builds, pushes, pulls, and tests the image.
- `environment.yml`: the conda base env (`scgpt_env`); heavy deps are pip-installed in the Dockerfile.

## TODOs / documented limitations

- Multi-gene perturbations are passed through to scGPT as GEARS `A+B` combos; this exercises scGPT's
  combinatorial path but has only been *wired*, not validated end-to-end here.
- `expression_level_after_perturbation` magnitude and `prediction_timescale` are structurally
  unsupported by scGPT and are ignored (documented above).
- `early_stop` is accepted but not yet used (no held-out split is created; fine-tuning runs a fixed
  number of epochs). Add a validation split + early stopping if desired.
- End-to-end execution requires the pretrained weights and a GPU, which were not available in the
  authoring environment, so the fine-tune/predict path has been verified against the scGPT source and
  perturbation tutorial rather than by a live run.
