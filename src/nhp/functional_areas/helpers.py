import os
import json
from dotenv import load_dotenv
from pyspark.sql import DataFrame
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
import pyspark.sql.functions as F

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
    env_vars["secret_scope"] = os.getenv("DATABRICKS_SECRET_SCOPE")
    env_vars["table_key"] = os.getenv("DATABRICKS_SECRET_TABLE_KEY")
    env_vars["storage_key"] = os.getenv("DATABRICKS_SECRET_STORAGE_KEY")
    env_vars["account_name"] = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    env_vars["table_name"] = os.getenv("TABLE_NAME")
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


def find_latest_data_patch_version(minor_version: str) -> str:
    """Data is stored in format vX.X.X whilst demand model version records only vX.X in params.
    We need to find out what the latest patch version of the data is and use that.
    For example, for v4.4 of the model we might have v4.4.0 and v4.4.1 data folders.

    Args:
        minor_version (str): Demand model version, in format vX.X

    Returns:
        str: Latest patch version matching the minor version.
    """
    path_to_data = "/Volumes/nhp/model_data/files/"
    list_of_patch_versions = [
        name for name in os.listdir(path_to_data) if name.startswith(minor_version)
    ]
    return sorted(list_of_patch_versions)[-1]


def load_op_aae_data(
    demand_model_version: str, activity_type: str, fyear: int, provider: str
) -> DataFrame:
    """Loads OP and AAE original model data. Adds model_run column, setting the value to 0
    for the baseline.

    Args:
        demand_model_version (str): Which version of demand model data to use
        activity_type (str): Which activity type data to load: op or aae
        fyear (int): Which fyear data to load
        provider (str): Which provider data to load

    Returns:
        DataFrame: Pyspark dataframe with original data
    """
    data_version = find_latest_data_patch_version(demand_model_version)
    data_folder = f"/Volumes/nhp/model_data/files/{data_version}/{activity_type}/fyear={fyear}/dataset={provider}/"
    return (
        spark.read.parquet(data_folder)
        .withColumn("model_run", F.lit(0))
        .withColumnRenamed("index", "rn")
    )


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
