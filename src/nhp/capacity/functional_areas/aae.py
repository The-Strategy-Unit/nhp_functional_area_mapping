import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame

from nhp.capacity.functional_areas.classifications import (
    class_ae,
    class_ae_major,
    class_ae_minor,
    class_ae_resus,
    class_age_adult,
    class_age_child,
    class_sdec,
)
from nhp.capacity.functional_areas.processing_helpers import add_missing_groupings

spark = DatabricksSession.builder.getOrCreate()

## Groupings


def is_adult_minor():
    return class_ae() & class_age_adult() & class_ae_minor()


def is_adult_major():
    return class_ae() & class_age_adult() & class_ae_major()


def is_child_minor():
    return class_ae() & class_age_child() & class_ae_minor()


def is_child_major():
    return class_ae() & class_age_child() & class_ae_major()


def create_aae_groupings(df: DataFrame) -> DataFrame:
    """Adds "grouping" column to the A&E data with the functional areas

    Args:
        df (DataFrame): Raw A&E data

    Returns:
        DataFrame: A&E data, with additional "grouping" column with functional areas created as detailed
        in the specification
    """
    df = df.withColumn(
        "grouping",
        F.when(class_sdec(), "sdec_attendances")
        .when(
            class_ae_resus(),
            "resus_attendances",
        )
        .when(
            is_adult_minor(),
            "adult_minor_attendances",
        )
        .when(
            is_adult_major(),
            "adult_major_attendances",
        )
        .when(
            is_child_minor(),
            "child_minor_attendances",
        )
        .when(
            is_child_major(),
            "child_major_attendances",
        ),
    )
    return df


def process_sdec_converted(db_path_to_full_model_results: str) -> DataFrame:
    """Processes the activity converted from IP to SDEC, adding functional area grouping column and
    aggregating by model run and grouping

    Args:
        db_path_to_full_model_results (str): Path to location of full model results on Databricks

    Returns:
        DataFrame: Activity converted from IP to SDEC, with functional area grouping column
    """
    sdec_converted = spark.read.parquet(
        db_path_to_full_model_results + "sdec_conversion"
    ).withColumn("grouping", F.lit("sdec_attendances"))
    sdec_groupings_per_run = sdec_converted.groupBy("model_run", "grouping").agg(
        F.sum("arrivals").alias("total")
    )
    return sdec_groupings_per_run


def process_aae(
    aae_original: DataFrame,
    aae_model_results: DataFrame,
    sdec_groupings_per_run: DataFrame,
) -> DataFrame:
    """Processes and aggregates the A&E baseline and model results to produce functional area outputs,
    aggregated by model run and grouping.

    Args:
        aae_original (DataFrame): Baseline A&E data
        aae_model_results (DataFrame): A&E full model results
        sdec_groupings_per_run (DataFrame): A&E SDEC activity converted from inpatients in modelling process

    Returns:
        DataFrame: A&E data for each of the model runs aggregated into functional areas
    """
    baseline_grouped = (
        create_aae_groupings(aae_original)
        .groupBy("model_run", "grouping")
        .agg(F.sum("arrivals").alias("total"))
    )
    groupings_per_run = (
        create_aae_groupings(aae_original)
        .drop("model_run", "arrivals")
        .join(aae_model_results, on="rn", how="left")
        .groupBy("model_run", "grouping")
        .agg(F.sum("arrivals").alias("total"))
    )
    groupings_per_run_with_sdec = (
        groupings_per_run.unionByName(sdec_groupings_per_run)
        .groupBy("model_run", "grouping")
        .agg(F.sum("total").alias("total"))
    )
    final_groupings_per_run_with_baseline = groupings_per_run_with_sdec.unionByName(
        baseline_grouped
    )
    # Add missing groupings - we need all groupings to be present in all model runs even if value is 0
    required_aae_groupings = [
        "adult_minor_attendances",
        "adult_major_attendances",
        "child_minor_attendances",
        "child_major_attendances",
        "resus_attendances",
        "sdec_attendances",
    ]
    final_df = add_missing_groupings(
        final_groupings_per_run_with_baseline, required_aae_groupings
    )
    return final_df


def qa_aae_results(
    default_results: DataFrame,
    final_aae_df: DataFrame,
):
    """Quality Assurance step: checks that values in a given column produce the same mean in new functional area
    aggregations as with default model results.

    Args:
        default_results (DataFrame): Default model results
        final_aae_df (DataFrame): DataFrame of functional area pipeline outputs
    """
    default_results = default_results.filter(
        F.col("pod").isin(["aae_type-01", "aae_type-05"])
    )  # functional areas only use type 01 and type 05
    default_results_value = (
        default_results.groupBy("model_run")
        .agg(F.sum("value").alias("value"))
        .agg(F.mean("value").alias("mean_arrivals"))
        .collect()[0][0]
    )
    grouped_results_value = (
        final_aae_df.filter(F.col("model_run") != 0)  # model_run 0 is baseline
        .groupBy("model_run")
        .agg(F.sum("total").alias("arrivals"))
        .agg(F.mean("arrivals").alias("mean_arrivals"))
        .collect()[0][0]
    )
    try:
        assert float(default_results_value) == float(grouped_results_value)
    except AssertionError:
        print(
            "Aggregated results are not aligned with default model results. Check arrivals"
        )
        raise
