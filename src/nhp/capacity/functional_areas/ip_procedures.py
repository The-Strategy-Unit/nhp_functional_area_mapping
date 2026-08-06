import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame

from nhp.capacity.functional_areas.classifications import (
    class_cardiac_cath,
    class_cardiology,
    class_has_procedure,
    class_int_radiology,
)
from nhp.capacity.functional_areas.processing_helpers import add_missing_groupings

spark = DatabricksSession.builder.getOrCreate()


def is_cardiology():
    return class_cardiology() & class_has_procedure()


def is_catheter_procedure():
    return class_has_procedure() & class_cardiac_cath()


def is_cardiac_catheter_procedure():
    return is_cardiology() | is_catheter_procedure()


def is_int_radiology_proc():
    return class_int_radiology() & class_has_procedure()


def create_ip_procedure_groupings(ip_data: DataFrame) -> DataFrame:
    """Adds "grouping" column to the IP data with the functional areas for IP procedures

    Args:
        ip_data (DataFrame): IP data

    Returns:
        DataFrame: IP data, aggregated by procedures functional areas
    """
    df = (
        ip_data.withColumn(
            "grouping",
            F.when(is_int_radiology_proc(), "interventional_radiology_procedure").when(
                is_cardiac_catheter_procedure(), "cardiac_catheter_procedure"
            ),
        )
        .groupby("grouping", "sitetret", "model_run")
        .agg(F.count("rn").alias("total"))
    )
    return df


def process_ip_procedures(
    ip_original_mapped: DataFrame, ip_model_results: DataFrame
) -> DataFrame:
    """Processes and aggregates the IP baseline and model results to produce functional area outputs for IP procedures,
    aggregated by model run and procedure grouping.

    Args:
        ip_original_mapped (DataFrame): Baseline IP data
        ip_model_results (DataFrame): IP full model results

    Returns:
        DataFrame: IP data for each of the model runs aggregated into functional areas for procedures
    """
    baseline_grouped = create_ip_procedure_groupings(ip_original_mapped)
    groupings_per_run = create_ip_procedure_groupings(
        ip_original_mapped.drop("speldur", "classpat", "model_run").join(
            ip_model_results, on="rn", how="left"
        )
    )
    final_groupings_per_run_with_baseline = groupings_per_run.unionByName(
        baseline_grouped
    )
    # Add missing groupings - we need all groupings to be present in all model runs even if value is 0
    required_procedure_groupings = [
        "interventional_radiology_procedure",
        "cardiac_catheter_procedure",
    ]
    final_df = add_missing_groupings(
        final_groupings_per_run_with_baseline, required_procedure_groupings
    )
    return final_df
