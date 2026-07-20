import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame

from nhp.capacity.functional_areas.classifications import (
    class_birth_assisted,
    class_birth_elective_csection,
    class_birth_event,
    class_birth_nonelective_c_section,
    class_birth_normal,
    class_maternity,
    class_no_birth_event,
    class_non_zero_los,
    class_zero_los,
)
from nhp.capacity.functional_areas.processing_helpers import add_missing_groupings

spark = DatabricksSession.builder.getOrCreate()


def is_normal_delivery():
    return class_maternity() & class_birth_event() & class_birth_normal()


def is_assisted_delivery():
    return class_maternity() & class_birth_event() & class_birth_assisted()


def is_maternity_assessment():
    return class_maternity() & class_zero_los() & class_no_birth_event()


def is_nonelective_csection():
    return class_maternity() & class_birth_event() & class_birth_nonelective_c_section()


def is_elective_csection():
    return class_maternity() & class_birth_event() & class_birth_elective_csection()


def is_overnight_no_birth_event():
    return class_maternity() & class_non_zero_los() & class_no_birth_event()


def create_ip_maternity_groupings(ip_data: DataFrame) -> DataFrame:
    """Adds "grouping" column to the IP data with the functional areas for IP daycase

    Args:
        ip_data (DataFrame): IP data

    Returns:
        DataFrame: IP data, aggregated by daycase functional areas
    """
    df = (
        ip_data.withColumn(
            "grouping",
            F.when(is_normal_delivery(), "maternity_normal_delivery")
            .when(is_assisted_delivery(), "maternity_assissted_delivery")
            .when(is_maternity_assessment(), "maternity_assessment")
            .when(is_nonelective_csection(), "maternity_nonelective_csection")
            .when(is_elective_csection(), "maternity_elective_csection")
            .when(is_overnight_no_birth_event(), "maternity_overnight_no_birth"),
        )
        .groupby("grouping", "model_run")
        .agg(F.count("rn").alias("total"))
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
    baseline_grouped = create_ip_maternity_groupings(ip_original_mapped)
    groupings_per_run = create_ip_maternity_groupings(
        ip_original_mapped.drop("speldur", "classpat", "model_run").join(
            ip_model_results, on="rn", how="left"
        )
    )
    final_groupings_per_run_with_baseline = groupings_per_run.unionByName(
        baseline_grouped
    )
    # Add missing groupings - we need all groupings to be present in all model runs even if value is 0
    required_ip_groupings = [
        "maternity_normal_delivery",
        "maternity_assisted_delivery",
        "maternity_assessment",
        "maternity_nonelective_csection",
        "maternity_elective_csection",
        "maternity_overnight_no_birth",
    ]
    final_df = add_missing_groupings(
        final_groupings_per_run_with_baseline, required_ip_groupings
    )
    return final_df
