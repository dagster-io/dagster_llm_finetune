from dagster import load_assets_from_modules

from . import datasets, models, predictions

get_module_name = lambda module: module.__name__.rsplit(".", 1)[-1]

dataset_assets = load_assets_from_modules(
    [datasets],
    group_name=get_module_name(datasets),
)
model_assets = load_assets_from_modules([models], group_name=get_module_name(models))
prediction_assets = load_assets_from_modules(
    [predictions], group_name=get_module_name(predictions)
)
