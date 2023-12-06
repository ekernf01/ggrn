GGRN can interface with any GRN software that can run in a Docker container. 

- **Basics:** Create a Docker image. As a starting point, use the Dockerfile and the python script given in this repo in `Dockerfiles/template`; copy and rename into an adjacent folder. Although you can name the image whatever you prefer, we use the convention `ggrn_docker_backend_{f}` where f is the folder you just created. Upon running, any container from this image should run your program, which should:
    - read training data from `to_from_docker/train.h5ad`. You can expect this training data to pass the checks in `ggrn.validate_training_data(train)`.
    - read perturbations to predict in `to_from_docker/perturbations.json`. You can expect a list of lists like `[["NANOG", 5.43], ["KLF4", 6.78]]`, meaning you should predict one observation where NANOG expression is set to 5.34 and another where KLF4 expression is set to 6.78. For multi-gene perturbations, you'll find comma-separated lists as strings, and yes, I'm very sorry about this. An example: `[["NANOG,POU5F1", "5.43,0.0"], ["KLF4,SOX2", "6.78,9.12"]]`. 
    - train your method and make those predictions, preserving the order.
    - save the predictions in `h5ad` format as `to_from_docker/predictions.h5ad`. To be ultra-safe about not scrambling them, the order is expected to match what you find in `perturbations.json`, and the `.obs` is expected to have a column `perturbation` and a column `expression_level_after_perturbation`. You can reuse the boilerplate from our example in `Dockerfiles/template`.
- **Passing arguments to your method**:
    - When using GGRN with a Docker backend, you can still pass all the usual GGRN args to `GRN.fit()`. These will be saved to `to_from_docker/ggrn_args.json` and mounted to the container, so your code can look for keyword args in `to_from_docker/ggrn_args.json`.
    - If your method does not fit cleanly into the grammar defining GGRN or has other keyword args, you can also pass in custom keyword args. They will be saved to `to_from_docker/kwargs.json` and mounted to the container, so your code can look for keyword args in `to_from_docker/kwargs.json`.
        - When using GGRN via our benchmarking framework, you can specify kwargs by adding a key `kwargs` to the `metadata.json` for your experiment. The value should be a dict containing anything you want. For more info, you can read the [reference](https://github.com/ekernf01/perturbation_benchmarking/blob/main/docs/reference.md) on `kwargs` and `kwargs_to_expand`, and for an example, you can [consult our how-to](https://github.com/ekernf01/perturbation_benchmarking/blob/main/docs/how_to.md).
        - When using GGRN directly, you can pass a dict `kwargs` to `GRN.fit` in your python code.  Be aware that Python may not translate perfectly to json; for instance, json lacks Python's `None` value. 

### Example

To try it out, install GGRN and obtain some test data, then run the following. One way to set up your environment is documented in our [benchmarking project](http://github.com/ekernf01/perturbation_benchmarking).

```python
import ggrn.api as ggrn
import load_perturbations
# Obtain example data
load_perturbations.set_data_path(
    '../perturbation_data/perturbations' # Change this to where you put the perturbation data collection.
)
train = load_perturbations.load_perturbation("nakatake")
grn = ggrn.GRN(train) 
# This line saves some inputs to `to_from_docker`, but doesn't actually run the container, because we don't currently save trained models inside the container.
grn.fit(
    method ="docker__--cpus='.5'__ekernf01/ggrn_docker_backend_template", 
    kwargs = {"example_kwarg":"george"}                    
)
# This line runs the container, doing both training and prediction, then removes the container.
predictions = grn.predict([("POU5F1", 0), ("NANOG", 0)])
# You should be left with an AnnData. 
predictions
```

### Advanced 

- **Passing args to docker**: When you call GGRN, you will specify `method="docker__myargs__myimage"` when you call `GRN.fit`. The second element of this double-underscore-separated list is provided directly to docker, as in `docker run myargs <other relevant stuff> myimage`. For example, to limit the cpu usage, you can use `method="docker__--cpus='.5'__myimage"`, or to use a gpu, you can use `method="docker__--gpus all ubuntu nvidia-smi__myimage"`. Certain options are already used. We always pass in `--rm` to remove the container when it finishes, and we always use `--mount` to share files as described above. We have not tested whether there are things you could provide that would interfere with this, so you will need some knowledge of Docker to avoid problems. Initially, we recommend using no Docker args, like `method="docker____myimage"`.


