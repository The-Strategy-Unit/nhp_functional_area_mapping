import pandas as pd
import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame

spark = DatabricksSession.builder.getOrCreate()


def add_missing_groupings(
    final_groupings_per_run_with_baseline: DataFrame,
    required_groupings: list[str],
    grouping_col_name: str = "grouping",
) -> DataFrame:
    """Add missing functional area groupings into DataFrame, with value 0

    Args:
        final_groupings_per_run_with_baseline (DataFrame): Fully processed dataframe
        required_groupings (list): Full list of all groupings
        col_name (str): Name of grouping column

    Returns:
        DataFrame: DataFrame with required functional area groupings for every run and site
    """
    model_runs_sites = final_groupings_per_run_with_baseline.select(
        "model_run", "sitetret"
    ).distinct()
    required_df = spark.createDataFrame(
        [(g,) for g in required_groupings], [grouping_col_name]
    )
    expected = model_runs_sites.crossJoin(required_df)
    completed_required = expected.join(
        final_groupings_per_run_with_baseline,
        on=["model_run", "sitetret", grouping_col_name],
        how="left",
    )
    # Fill all non-key columns with 0
    value_columns = [
        c
        for c in final_groupings_per_run_with_baseline.columns
        if c not in {"model_run", "sitetret", grouping_col_name}
    ]
    for col in value_columns:
        completed_required = completed_required.withColumn(
            col,
            F.coalesce(F.col(col), F.lit(0)),
        )
    non_required = final_groupings_per_run_with_baseline.filter(
        ~F.col(grouping_col_name).isin(required_groupings)
    )
    final_df = non_required.unionByName(completed_required)
    return final_df


def get_tretspef_lookup(
    excel_path: str,
) -> DataFrame:
    """Load tretspef lookup file from given location. Currently using custom edit of
    https://digital.nhs.uk/binaries/content/assets/website-assets/isce/dcb0028/0028452019codelistspecificationv1.2.xlsx

    Args:
        excel_path (str): Path to Excel file

    Returns:
        DataFrame: DataFrame with two columns, tretspef (treatment function code) and treatment specialty type (medical/surgical)
    """
    tretspef_lookup = pd.read_excel(excel_path, engine="openpyxl")
    tretspef_lookup["tretspef"] = tretspef_lookup["tretspef"].astype(int)
    tretspef_lookup_df = spark.createDataFrame(
        tretspef_lookup.rename(columns={"NHP categorisation": "tretspef_type"})
    )
    return tretspef_lookup_df


def add_tretspef_type(
    ip_original: DataFrame, tretspef_lookup_df: DataFrame
) -> DataFrame:
    """Adds tretspef_type column to inpatients data to help with mapping daycase activity to medical/surgical

    Args:
        ip_original (DataFrame): Inpatients model data
        tretspef_lookup_df (DataFrame): DataFrame mapping treatment specialty code to treatment specialty type (medical/surgical)

    Returns:
        DataFrame: Inpatients model data with additional tretspef_type column
    """
    return ip_original.join(
        tretspef_lookup_df,
        on=ip_original["tretspef"] == tretspef_lookup_df["tretspef"],
        how="left",
    ).drop(tretspef_lookup_df["tretspef"])
