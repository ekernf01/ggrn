GGRN can interface with any GRN software that can run in a Docker or Singularity container. You need to make a Docker image and a program that will run in it. The folder `from_to_docker` will be "mounted" to the container, allowing files to pass in and out. `from_to_docker` will appear on the host machine in the working directory where GGRN is run, and it will appear in the Docker container at the root of the filesystem. Once the container finishes running, it will be deleted, but outputs in `from_to_docker` will persist; GGRN will read these and return them, allowing the user to operate entirely within Python.

### Prediction program

This runs inside the container. It must:

- read training data from `from_to_docker/train.h5ad`. You can expect this training data to pass the checks in `ggrn.validate_training_data(train)`.
- read perturbations to predict in `from_to_docker/predictions_metadata.csv`. You can expect a table containing the following columns: `['timepoint', 'cell_type', 'perturbation', "expression_level_after_perturbation", 'prediction_timescale']`. **Please return one prediction per row in this table.** For multi-gene perturbations, you'll find quoted comma-separated lists as strings, and yes, I'm very sorry about this. An example: "NANOG,POU5F1" for `perturbation` and "5.43,0.0" for `expression_level_after_perturbation`. 
- read all the GGRN args from `from_to_docker/ggrn_args.json`. (To learn more about these, consult the Grammar of Gene Regulatory Networks documentation.) Two exceptions:
    - For the `prediction_timescale` parameter, look for a column in `predictions_metadata.csv`
    - For the `network` parameter, you can read a base network from `from_to_docker/network.parquet`.
- read custom keyword args from `from_to_docker/kwargs.json`
- train your method and make those predictions, preserving the order.
- log stdout to `from_to_docker/stdout.txt`
- log stderr to `from_to_docker/err.txt`
- save the predictions in `h5ad` format as `from_to_docker/predictions.h5ad`. To be ultra-safe about not scrambling them, the order is expected to match what you find in `predictions_metadata.csv`, and the `.obs` is expected to have all those columns. 

More about `kwargs`. When using GGRN directly (not through the benchmarking code), you can pass a dict `kwargs` to `GRN.fit` in your python code. This is what gets saved to `from_to_docker/kwargs.json`. Be aware that Python may not translate perfectly to json; for instance, json lacks Python's `None` value. When using GGRN via our benchmarking framework, you can specify kwargs by adding a key `kwargs` to the `metadata.json` for your experiment. The value should be a dict containing anything you want. For more info, you can read the [pereggrn reference](https://github.com/ekernf01/perturbation_benchmarking/blob/main/docs/reference.md) on `kwargs` and `kwargs_to_expand`, and for an example, you can [consult our how-to](https://github.com/ekernf01/perturbation_benchmarking/blob/main/docs/how_to.md).

### Docker image and interface to GGRN

You must create a Docker image with an enviroment where your program can run. As a starting point, you can use the examples in `Dockerfiles`; copy and rename into an adjacent folder. Although you can name the image whatever you prefer, we use the convention `ggrn_docker_backend_{f}` where f is the folder you just created. 

The only requirement is that when a container runs, it should run your program described above. This is taken care of in our examples by the `ENTRYPOINT` directive. If you remove that for interactive development, you will probably need to restore it eventually.

### Quick debugging

- Look for `build.sh` to see how we build and push an image. 
- Look for `test.py` next to our example Dockerfiles to easily test your newly built image.
- To run your container interactively, you can remove the `ENTRYPOINT` directive. Once you remove it, build the image and interact like this.

```bash
docker run --rm -it  --mount type=bind,source=/absolute/path/to/from_to_docker/,destination=/from_to_docker    your_docker_image
```

### Example

To try out a docker backend, install our [benchmarking project](http://github.com/ekernf01/perturbation_benchmarking) and run the following python snippet.

```python
import ggrn.api as ggrn
import load_perturbations
# Obtain example data
load_perturbations.set_data_path(
    '../perturbation_data/perturbations' # Change this to where you put the perturbation data collection.
)
train = load_perturbations.load_perturbation("nakatake")
grn = ggrn.GRN(train) 
# This line saves some inputs to `from_to_docker`, but doesn't actually run the container, because we don't currently save trained models inside the container.
grn.fit(
    method ="docker__--cpus='.5'__ekernf01/ggrn_docker_backend_template", 
    kwargs = {"example_kwarg":"george"}                    
)
# This line runs the container, doing both training and prediction, then removes the container.
predictions = grn.predict([("POU5F1", 0), ("NANOG", 0)])
# You should be left with an AnnData. 
predictions
```

To run the same backend via Singularity, use `method ="singularity__--cpus='.5'__ekernf01/ggrn_docker_backend_template"`.

### Advanced: passing args to docker or singularity

You will specify `method="docker__myargs__myimage"` when you call `GRN.fit`. We split on double-underscore, and the stuff in the middle, `"myargs"`, is provided to Docker as arguments, as in `docker run myargs <other relevant stuff> myimage`. Same with `singularity__myargs__myimage`. Some relevant technicalities:

- We split `myargs` on spaces and provide it as a list, like `subprocess.call([docker] + myargs.split(' ') + ...)`. For example, to limit the cpu usage, you can use `method="docker__--cpus='.5'__myimage"`, or to use a gpu, you can use `method="docker__--gpus all ubuntu nvidia-smi__myimage"`. 
- Certain options are already used. With Docker, we always pass in `--rm` to remove the container when it finishes, and we always use `--mount` to share files as described above. With singularity, we use `--bind` to share files and we use `--no-home` for docker compatibility. Do not use these args. We have not tested whether there are things you could provide that would interfere with this, so we recommend using no Docker args, like `method="docker____myimage"`. Otherwise you may need some knowledge of Docker or singularity to troubleshoot. 


