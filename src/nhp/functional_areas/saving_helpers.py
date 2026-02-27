from azure.storage.blob import ContainerClient
from databricks.sdk import WorkspaceClient
from pyspark.sql import DataFrame
from azure.data.tables import TableClient
from azure.core.credentials import AzureSasCredential


w = WorkspaceClient()
dbutils = w.dbutils


def upload_data(env_vars: dict, metadata: dict, df: DataFrame, df_name: str):
    """Uploads functional area aggregations to Azure Storage, with metadata

    Args:
        env_vars (dict): Dictionary of required environment variables. Required values listed in nhp.loading_helpers.load_env_vars
        metadata (dict): Dictionary with metadata for data upload
        df (DataFrame): Functional area aggregations dataframe
        df_name (str): Name of functional area aggregations dataframe
    """
    storage_token = dbutils.secrets.get(
        env_vars["secret_scope"], env_vars["storage_key"]
    )
    storage_url = dbutils.secrets.get(env_vars["secret_scope"], "url")
    container_client = ContainerClient.from_container_url(
        f"{storage_url}?{storage_token}"
    )
    container_client.upload_blob(
        f"functional-aggregations/{dbutils.widgets.get('capacity_model_version')}/{metadata['RowKey']}/f{df_name}.parquet",
        df.toPandas().set_index("model_run").sort_index().to_parquet(),
        metadata=metadata,
        overwrite=True,
    )


def add_metadata_to_ats(env_vars: dict, metadata: dict):
    """Creates new entity in Azure Table Storage with relevant metadata from functional area aggregation

    Args:
        env_vars (dict): Dictionary of required environment variables. Required values listed in nhp.loading_helpers.load_env_vars
        metadata (dict): Dictionary with metadata for data upload
    """
    table_token = dbutils.secrets.get(
        env_vars["secret_scope"], env_vars["table_key"]
    ).strip()
    table_endpoint = f"https://{env_vars['account_name']}.table.core.windows.net"
    table_client = TableClient(
        endpoint=table_endpoint,
        table_name=env_vars["table_name"],
        credential=AzureSasCredential(table_token),
    )
    table_client.create_entity(entity=metadata)
