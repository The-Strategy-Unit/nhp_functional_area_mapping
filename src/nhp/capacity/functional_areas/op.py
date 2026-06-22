from typing import List

import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame

from nhp.capacity.functional_areas.classifications import (
    class_has_procedure,
    class_op_face_to_face,
    class_op_first,
    class_op_follow_up,
    class_op_virtual,
)
from nhp.capacity.functional_areas.processing_helpers import add_missing_groupings

spark = DatabricksSession.builder.getOrCreate()


def is_op_procedures():
    return F.sum(
        F.when(
            class_has_procedure(),
            class_op_face_to_face(),
        )
    )


def is_op_first_attendances():
    return F.sum(
        F.when(
            class_op_first(),
            class_op_face_to_face(),
        )
    )


def is_op_follow_up_attendances():
    return F.sum(
        F.when(
            class_op_follow_up(),
            class_op_face_to_face(),
        )
    )


def process_op_converted(db_path_to_full_model_results: str) -> DataFrame:
    """Processes the activity converted from IP to OP, adding functional area grouping column and
    aggregating by model run and grouping

    Args:
        db_path_to_full_model_results (str): Path to location of full model results on Databricks

    Returns:
        DataFrame: Activity converted from IP to OP, with functional area grouping column
    """
    op_converted = spark.read.parquet(
        db_path_to_full_model_results + "op_conversion"
    ).withColumn("grouping", F.lit("outpatient_procedures"))
    op_converted_groupings_per_run = op_converted.groupBy("model_run", "grouping").agg(
        F.sum("attendances").alias("total")
    )
    return op_converted_groupings_per_run


def create_op_groupings(df: DataFrame) -> DataFrame:
    """Calculates outpatients (OP) groupings and aggregates total for each grouping in each model run

    Args:
        df (DataFrame): OP data

    Returns:
        DataFrame: Aggregated OP data with sum of attendances by grouping and model run
    """
    agg_df = df.groupBy("model_run").agg(
        is_op_procedures().alias("op_procedures"),
        is_op_first_attendances().alias("op_first_attendances"),
        is_op_follow_up_attendances().alias("op_follow_up_attendances"),
        class_op_virtual().alias("op_virtual_attendances"),
    )
    cols = [
        "op_procedures",
        "op_first_attendances",
        "op_follow_up_attendances",
        "op_virtual_attendances",
    ]

    mapping = []
    for c in cols:
        mapping += [F.lit(c), F.col(c)]

    return agg_df.select(
        "model_run", F.explode(F.create_map(*mapping)).alias("grouping", "total")
    )


def process_op(
    op_original: DataFrame, op_model_results: DataFrame, op_converted: DataFrame
) -> DataFrame:
    """Processes and aggregates the OP baseline and model results to produce functional area outputs,
    aggregated by model run and grouping.

    Args:
        op_original (DataFrame): Baseline OP data
        op_model_results (DataFrame): OP full model results
        op_converted (DataFrame): OP activity converted from inpatients in modelling process due to TPMAs

    Returns:
        DataFrame: OP data for each of the model runs aggregated into functional areas
    """
    baseline_grouped = create_op_groupings(op_original)
    groupings_per_run = create_op_groupings(
        op_original.drop("attendances", "tele_attendances", "model_run").join(
            op_model_results, on="rn", how="left"
        )
    )
    groupings_per_run_with_op_converted = (
        groupings_per_run.unionByName(op_converted)
        .groupBy("model_run", "grouping")
        .agg(F.sum("total").alias("total"))
    )
    final_groupings_per_run_with_baseline = (
        groupings_per_run_with_op_converted.unionByName(baseline_grouped)
    )
    # # Add missing groupings - we need all groupings to be present in all model runs even if value is 0
    required_op_groupings = [
        "outpatient_procedures",
        "outpatient_first_attendances",
        "outpatient_followup_attendances",
        "outpatient_virtual_attendances",
    ]
    final_df = add_missing_groupings(
        final_groupings_per_run_with_baseline, required_op_groupings
    )
    return final_df


def qa_op_results(
    default_results: DataFrame,
    final_op_df: DataFrame,
    sites: List[str],
):
    """Quality Assurance step: checks that values in a given column produce the same mean in new functional area
    aggregations as with default model results.

    Args:
        default_results (DataFrame): Default model results
        final_op_df (DataFrame): DataFrame of functional area pipeline outputs
        sites (List[str]):
    """
    if "ALL" not in sites:
        default_results = default_results.where(F.col("sitetret").isin(sites))

    default_grouped = default_results.groupBy("model_run", "pod", "measure").agg(
        F.sum("value").alias("value")
    )

    # Define what we want to check
    checks = {
        "outpatient_virtual_attendances": (
            ["op_first", "op_follow-up"],
            ["tele_attendances"],
        ),
        "outpatient_procedures": (["op_procedure"], ["attendances"]),
        "outpatient_first_attendances": (["op_first"], ["attendances"]),
        "outpatient_followup_attendances": (["op_follow-up"], ["attendances"]),
    }

    default_means = {}

    for key, (pods, measures) in checks.items():
        df = default_grouped
        df = df.filter(F.col("pod").isin(pods))
        df = df.filter(F.col("measure").isin(measures))

        default_means[key] = (
            df.groupBy("model_run")
            .agg(F.sum("value").alias("value"))
            .agg(F.mean("value").alias("default_value"))
            .collect()[0][0]
        )

    aggregation_results_dict = {
        r["grouping"]: r["mean"]
        for r in (
            final_op_df.filter(F.col("model_run") != 0)
            .groupBy("grouping")
            .agg(F.mean("total").alias("mean"))
            .collect()
        )
    }

    try:
        for key in checks:
            assert float(default_means[key]) == float(aggregation_results_dict[key])
    except AssertionError:
        print(
            f"Aggregated results are not aligned with default model results. Check {key}"
        )
