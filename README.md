# HappyPose

### Example with uv

```
cd happypose
uv sync
```

## Create data directory

Set a directory where model weights and datasets will be stored (can be several GB):
```
export HAPPYPOSE_DATA_DIR=/somewhere/convenient
```

Download all model weights and example data with a single command:
```
uv run python examples/download_data.py --all
```

Individual downloads are also available if needed:
```
uv run python examples/download_data.py --cosypose-models   # CosyPose HOPE weights
uv run python examples/download_data.py --megapose-models   # MegaPose weights
uv run python examples/download_data.py --example-data      # mustard0 example dataset
```

### Only if GPU (cuda) is available
Tests related to `evaluation` and `training` will be run if a GPU is available. Hence, a few more downloads are needed :

```
#ycbv models
uv run python -m happypose.toolbox.utils.download --cosypose_models \
            coarse-bop-ycbv-pbr--724183 \
            refiner-bop-ycbv-pbr--604090
```

```
uv run python -m happypose.toolbox.utils.download --bop_dataset ycbv
```
