from typing import List

import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame
from pyspark.sql.column import Column

from nhp.capacity.functional_areas.classifications import (
    class_daycase,
    class_elective,
    class_haem_onc,
    class_regular_day_night,
    class_renal,
    class_zero_los,
)
from nhp.capacity.functional_areas.processing_helpers import add_missing_groupings

spark = DatabricksSession.builder.getOrCreate()


def class_has_procedure() -> Column:
    return F.col("has_procedure")


def is_renal_elective():
    return class_renal() & class_elective() & class_zero_los()


def is_renal_regular_day_night():
    return class_renal() & class_regular_day_night()


def is_daycase_haem_onc():
    return class_daycase() & class_haem_onc() & class_has_procedure()


def create_ip_daycase_groupings(ip_data: DataFrame) -> DataFrame:
    """Adds "grouping" column to the IP data with the functional areas for IP daycase

    Args:
        ip_data (DataFrame): IP data

    Returns:
        DataFrame: IP data, aggregated by daycase functional areas
    """
    df = ip_data.withColumn(
        "grouping",
        F.when(
            is_renal_elective() | is_renal_regular_day_night(), "daycase_renal_spells"
        ).when(is_daycase_haem_onc(), "daycase_haem_onc_spells"),
    )
    return df


def process_ip_daycase(
    ip_original_mapped: DataFrame, ip_model_results: DataFrame
) -> DataFrame:
    """Processes and aggregates the IP baseline and model results to produce functional area outputs for IP daycase,
    aggregated by model run and grouping.

    Args:
        ip_original_mapped (DataFrame): Baseline IP data
        ip_model_results (DataFrame): IP full model results

    Returns:
        DataFrame: IP data for each of the model runs aggregated into functional areas for daycase
    """
    baseline_grouped = create_ip_daycase_groupings(ip_original_mapped)
    groupings_per_run = create_ip_daycase_groupings(
        ip_original_mapped.drop("speldur", "classpat", "model_run").join(
            ip_model_results, on="rn", how="left"
        )
    )
    final_groupings_per_run_with_baseline = groupings_per_run.unionByName(
        baseline_grouped
    )
    # Add missing groupings - we need all groupings to be present in all model runs even if value is 0
    required_ip_groupings = [
        "adult_surgical_daycase",
        "adult_medical_daycase",
        "paediatric_surgical_daycase",
        "paediatric_medical_daycase",
        "adult_unknown_daycase",
        "paediatric_unknown_daycase",
    ]
    final_df = add_missing_groupings(
        final_groupings_per_run_with_baseline, required_ip_groupings
    )
    return final_df


def qa_ip_daycase_results(
    default_results: DataFrame, final_ip_daycase_df: DataFrame, sites: List[str]
):
    """Quality Assurance step: checks that values in a given column produce the same mean in new functional area
    aggregations as with default model results.

    Args:
        default_results (DataFrame): Default model results
        final_ip_daycase_df (DataFrame): DataFrame of functional area pipeline outputs
        sites (List[str]):
    """
    if "ALL" not in sites:
        default_results = default_results.where((F.col("sitetret").isin(sites)))
    default_results = default_results.where(
        (F.col("pod").like("%daycase%")) & (F.col("measure") == "admissions")
    )
    default_results_value = (
        default_results.groupBy("model_run")
        .agg(F.sum("value").alias("value"))
        .agg(F.mean("value").alias("mean"))
        .collect()[0][0]
    )
    grouped_results_value = (
        final_ip_daycase_df.filter(F.col("model_run") != 0)  # model_run 0 is baseline
        .groupBy("model_run")
        .agg(F.sum("total").alias("total"))
        .agg(F.mean("total").alias("mean_total"))
        .collect()[0][0]
    )
    try:
        assert float(default_results_value) == float(grouped_results_value)
    except AssertionError:
        print(
            "Aggregated results are not aligned with default model results. Check daycase calculations"
        )
        raise
