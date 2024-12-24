An example Docker container for methods developers to modify.

- `train.jl`: runs inside the container.
- `Dockerfile`: defines the environment/image.
- `test.py`: makes ggrn try to use the container to make predictions.
- `build.sh`: builds the image, pushes it, pulls it, bops it, smashes it, and tests it.
