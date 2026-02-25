import os
import json
from dotenv import load_dotenv


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
