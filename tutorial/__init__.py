import warnings

from dagster import (
    AssetSelection,
    DefaultScheduleStatus,
    Definitions,
    ExperimentalWarning,
    ScheduleDefinition,
    define_asset_job,
)

warnings.filterwarnings("ignore", category=ExperimentalWarning)

from .assets import dataset_assets, model_assets, prediction_assets
from .resources import RESOURCES

all_assets = [*dataset_assets, *model_assets, *prediction_assets]

refresh_all_assets = define_asset_job(
    "refresh_all_assets", AssetSelection.all()
)
raw_dataset_refresh_schedule = ScheduleDefinition(
    job=refresh_all_assets,
    cron_schedule="0 * * * *",
    default_status=DefaultScheduleStatus.RUNNING,
)

defs = Definitions(
    assets=all_assets, resources=RESOURCES, schedules=[raw_dataset_refresh_schedule]
)
