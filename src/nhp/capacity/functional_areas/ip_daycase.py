import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame
from pyspark.sql.column import Column

from nhp.capacity.functional_areas.classifications import (
    class_age_adult,
    class_age_child,
    class_daycase,
    class_elective,
    class_endoscopy,
    class_haem_onc,
    class_medical,
    class_regular_day_night,
    class_renal,
    class_surgical,
    class_zero_los,
)
from nhp.capacity.functional_areas.processing_helpers import add_missing_groupings

spark = DatabricksSession.builder.getOrCreate()


def class_has_procedure() -> Column:
    # defined separately because col name differs between OP and IP
    return F.col("has_procedure")


def is_renal_elective():
    return class_renal() & class_elective() & class_zero_los()


def is_renal_regular_day_night():
    return class_renal() & class_regular_day_night()


def is_daycase_haem_onc():
    return class_daycase() & class_haem_onc() & class_has_procedure()


def is_daycase_endoscopy():
    return class_daycase() & class_endoscopy()


def is_specialty_daycase():
    return (
        is_renal_elective()
        | is_renal_regular_day_night()
        | is_daycase_haem_onc()
        | is_daycase_endoscopy()
    )


def is_daycase_adult_medical():
    return (
        class_daycase()
        & class_age_adult()
        & class_medical()
        & ~F.coalesce(is_specialty_daycase(), F.lit(False))
    )


def is_daycase_adult_surgical():
    return (
        class_daycase()
        & class_age_adult()
        & class_surgical()
        & ~F.coalesce(is_specialty_daycase(), F.lit(False))
    )


def is_daycase_child_medical():
    return (
        class_daycase()
        & class_age_child()
        & class_medical()
        & ~F.coalesce(is_specialty_daycase(), F.lit(False))
    )


def is_daycase_child_surgical():
    return (
        class_daycase()
        & class_age_child()
        & class_surgical()
        & ~F.coalesce(is_specialty_daycase(), F.lit(False))
    )


def create_ip_daycase_groupings(ip_data: DataFrame) -> DataFrame:
    """Adds "grouping" column to the IP data with the functional areas for IP daycase

    Args:
        ip_data (DataFrame): IP data

    Returns:
        DataFrame: IP data, aggregated by daycase functional areas
    """
    df = (
        ip_data.withColumn(
            "grouping",
            F.when(
                is_renal_elective() | is_renal_regular_day_night(),
                "daycase_renal_spells",
            )
            .when(is_daycase_haem_onc(), "daycase_haem_onc_spells")
            .when(is_daycase_endoscopy(), "daycase_endoscopy_spells")
            .when(is_daycase_adult_medical(), "daycase_adult_medical_spells")
            .when(is_daycase_adult_surgical(), "daycase_adult_surgical_spells")
            .when(is_daycase_child_medical(), "daycase_child_medical_spells")
            .when(is_daycase_child_surgical(), "daycase_child_surgical_spells"),
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
        "daycase_haem_onc_spells",
        "daycase_endoscopy_spells",
        "daycase_adult_medical_spells",
        "daycase_adult_surgical_spells",
        "daycase_child_medical_spells",
        "daycase_child_surgical_spells",
        "daycase_renal_spells",
    ]
    final_df = add_missing_groupings(
        final_groupings_per_run_with_baseline, required_ip_groupings
    )
    return final_df
