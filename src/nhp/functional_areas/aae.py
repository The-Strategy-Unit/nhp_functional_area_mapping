from pyasn1_modules.rfc2315 import Data
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from nhp.functional_areas.processing_helpers import add_missing_groupings
from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.getOrCreate()


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
        F.when(F.col("pod") == "aae_type-05", "sdec_attendances")
        .when(
            (F.col("acuity") == "immediate-resuscitation"),
            "resus_attendances",
        )
        .when(
            (F.col("is_adult"))
            & (F.col("pod") == "aae_type-01")
            & (F.col("acuity").isin("standard", "non-urgent", "urgent")),
            "adult_minor_attendances",
        )
        .when(
            (F.col("is_adult"))
            & (F.col("pod") == "aae_type-01")
            & (F.col("acuity") == "very-urgent"),
            "adult_major_attendances",
        )
        .when(
            (~F.col("is_adult"))
            & (F.col("pod") == "aae_type-01")
            & (F.col("acuity").isin("standard", "non-urgent", "urgent")),
            "child_minor_attendances",
        )
        .when(
            (~F.col("is_adult"))
            & (F.col("pod") == "aae_type-01")
            & (F.col("acuity") == "very-urgent"),
            "child_major_attendances",
        )
        .when(
            (~F.col("is_adult")) & (F.col("pod") == "aae_type-02"),
            "child_type-02",
        )
        .when(
            (F.col("is_adult")) & (F.col("pod") == "aae_type-02"),
            "adult_type-02",
        )
        .when(
            (~F.col("is_adult")),
            "child_unknown",
        )
        .when(
            (F.col("is_adult")),
            "adult_unknown",
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
        F.sum("arrivals").alias("arrivals")
    )
    return sdec_groupings_per_run


def process_aae(
    aae_original: DataFrame,
    aae_model_results: DataFrame,
    sdec_groupings_per_run: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """Processes and aggregates the A&E baseline and model results to produce functional area outputs,
    aggregated by model run and grouping.

    Args:
        aae_original (DataFrame): Baseline A&E data
        aae_model_results (DataFrame): A&E full model results
        sdec_groupings_per_run (DataFrame): A&E SDEC activity converted from inpatients in modelling process

    Returns:
        tuple[DataFrame, DataFrame]: _description_
    """
    baseline_grouped = (
        create_aae_groupings(aae_original)
        .groupBy("model_run", "grouping")
        .agg(F.sum("arrivals").alias("arrivals"))
    )
    groupings_per_run = (
        baseline_grouped.drop("model_run", "arrivals")
        .join(aae_model_results, on="rn", how="left")
        .groupBy("model_run", "grouping")
        .agg(F.sum("arrivals").alias("arrivals"))
    )
    groupings_per_run_with_sdec = (
        groupings_per_run.unionByName(sdec_groupings_per_run)
        .groupBy("model_run", "grouping")
        .agg(F.sum("arrivals").alias("arrivals"))
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
        "adult_unknown",
        "child_unknown",
    ]
    final_df = add_missing_groupings(
        final_groupings_per_run_with_baseline, "arrivals", required_aae_groupings
    )
    summary = final_groupings_per_run_with_baseline.groupBy("grouping").agg(
        F.mean("arrivals").alias("mean"),
        F.expr("percentile_approx(arrivals, 0.10)").alias("p10"),
        F.expr("percentile_approx(arrivals, 0.90)").alias("p90"),
    )
    return final_df, summary
