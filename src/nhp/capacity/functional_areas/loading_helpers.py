import json
import os
from typing import List

import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from packaging.version import Version
from pyspark.sql import DataFrame

from nhp.capacity.functional_areas.utils import (
    earlier_minor,
    is_version_folder,
    latest,
    same_minor,
)

spark = DatabricksSession.builder.getOrCreate()
w = WorkspaceClient()
dbutils = w.dbutils


def load_env_vars() -> dict:
    """Loads environment variables. Note TODO: #14

    Raises:
        ValueError: Checks all env vars are present

    Returns:
        dict: Dictionary with loaded environment variables
    """
    load_dotenv()
    env_vars = {}
    env_vars["secret_scope"] = os.getenv("DATABRICKS_SECRET_SCOPE", "")
    env_vars["table_key"] = os.getenv("DATABRICKS_SECRET_TABLE_KEY", "")
    env_vars["storage_key"] = os.getenv("DATABRICKS_SECRET_STORAGE_KEY", "")
    env_vars["account_name"] = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
    env_vars["table_name"] = os.getenv("TABLE_NAME", "")
    for k, v in env_vars.items():
        try:
            assert len(v) > 0
        except AssertionError:
            raise ValueError(f"Env var missing! please check: {k}")
    return env_vars


def extract_model_run_details(
    path_to_full_model_results: str,
) -> tuple[str, int, str, str, str]:
    """Extract metadata about the model run

    Args:
        path_to_full_model_results (str): Path to full model results folder for specific model run

    Returns:
        tuple[str, int, str, str, str]: Tuple with demand_model_version, fyear, provider,
        scenario_name, and scenario_runtime
    """
    split_path = path_to_full_model_results.split("/")
    demand_model_version = split_path[1]
    fyear = int(get_params_json(path_to_full_model_results)["start_year"])
    provider = split_path[2]
    scenario_name = split_path[3]
    scenario_runtime = split_path[4]
    return demand_model_version, fyear, provider, scenario_name, scenario_runtime


def get_params_json(db_path_to_full_model_results: str) -> dict:
    """Load params JSON from given full model results path in Azure Storage

    Args:
        db_path_to_full_model_results (str): Path to full model results in Azure Storage

    Returns:
        dict: Loaded scenario parameters
    """
    path_to_aggregated_folder = db_path_to_full_model_results.replace(
        "full-", "aggregated-"
    )
    with open(
        "/Volumes/nhp/results/files/" + path_to_aggregated_folder + "params.json"
    ) as f:
        params = json.load(f)
    return params


def find_latest_data_version_db(volume_path: str, model_version: str) -> str:
    """Databricks equvalent of nhpy.az.find_latest_version (uses Databricks Volume instead of container client)

    Args:
        volume_path (str): Path to the volume on Databricks with model data
        model_version (str): App version in format v{major}.{minor}

    Returns:
        str: Latest data version
    """
    all_versions = [
        f.name.rstrip("/")
        for f in dbutils.fs.ls(volume_path)
        if is_version_folder(f.name.rstrip("/"))
    ]

    parsed = Version(model_version.lstrip("v"))
    major, minor = parsed.major, parsed.minor

    same = [v for v in all_versions if same_minor(v, major, minor)]
    if same:
        return latest(same)  # ty: ignore[invalid-return-type]

    earlier = [v for v in all_versions if earlier_minor(v, major, minor)]
    return latest(earlier) or "N/A"


def load_model_data(
    demand_model_version: str,
    activity_type: str,
    fyear: int,
    provider: str,
    sites: List[str],
) -> DataFrame:
    """Loads original model data. Adds model_run column, setting the value to 0
    for the baseline.

    Args:
        demand_model_version (str): Which version of demand model data to use
        activity_type (str): Which activity type data to load: op or aae
        fyear (int): Which fyear data to load
        provider (str): Which provider data to load

    Returns:
        DataFrame: Pyspark dataframe with original data
    """
    volume_path = "/Volumes/nhp/model_data/files/"
    data_version = find_latest_data_version_db(
        volume_path=volume_path, model_version=demand_model_version
    )
    data_folder = (
        volume_path
        + f"{data_version}/{activity_type}/fyear={fyear}/dataset={provider}/"
    )
    df = (
        spark.read.parquet(data_folder)
        .withColumn("model_run", F.lit(0))
        .withColumnRenamed("index", "rn")
        .fillna({"sitetret": "unknown"})
    )
    if "ALL" not in sites:
        df = df.where(F.col("sitetret").isin(sites))
    return df


def validate_result_path(path_to_full_model_results: str) -> str | None:
    """Checks if result path is provided, and exists

    Args:
        path_to_full_model_results (str): Path to full model results, in the format
        full-model-results/MODEL_VERSION/DATASET/SCENARIO_NAME/DATETIME/

    Returns:
        str | None: None if path to full model results not valid, Databricks path to folder if valid.
    """
    # Check path provided
    try:
        assert len(path_to_full_model_results) > 0
    except AssertionError:
        print("Error! Please supply path to full model results")
        raise
    # If path provided, check it exists
    db_path_to_full_model_results = (
        "/Volumes/nhp/results/files/" + path_to_full_model_results
    )
    try:
        dbutils.fs.ls(db_path_to_full_model_results)
        return db_path_to_full_model_results
    except Exception as e:
        if "java.io.FileNotFoundException" in str(e):
            raise


def load_default_results(
    db_path_to_full_model_results: str, activity_type: str, sites: List[str]
) -> DataFrame:
    """Load default results for QA purposes

    Args:
        db_path_to_full_model_results (str): Path to full model results folder
        activity_type (str): Activity type (aae, op, or ip)
        sites (List[str]): Which sites to filter results to

    Returns:
        DataFrame: Default results from model run, filtered to activity type and sites of interest
    """
    default = spark.read.parquet(db_path_to_full_model_results + "default.parquet")
    default_filtered = default.filter(
        (F.col("pod").like(f"{activity_type}%")) & (F.col("model_run") != 0)
    )
    if "ALL" not in sites:
        default_filtered = default_filtered.where(F.col("sitetret").isin(sites))
    return default_filtered
