import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from nhp.capacity.functional_areas.processing_helpers import add_missing_groupings
from databricks.connect import DatabricksSession
from typing import List

spark = DatabricksSession.builder.getOrCreate()


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
    """Adds "grouping" column to the OP data with the functional areas

    Args:
        df (DataFrame): Raw OP data

    Returns:
        DataFrame: OP data, with additional "grouping" column with functional areas created as detailed
        in the specification
    """
    agg_df = df.groupBy("model_run").agg(
        F.sum(F.when(F.col("has_procedures"), F.col("attendances"))).alias(
            "outpatient_procedures"
        ),
        F.sum(
            F.when(
                (~F.col("has_procedures")) & (F.col("is_first")), F.col("attendances")
            )
        ).alias("outpatient_first_attendances"),
        F.sum(
            F.when(
                (~F.col("has_procedures")) & (~F.col("is_first")), F.col("attendances")
            )
        ).alias("outpatient_followup_attendances"),
        F.sum(F.when(~F.col("has_procedures"), F.col("tele_attendances"))).alias(
            "outpatient_virtual_attendances"
        ),
    )

    cols = [
        "outpatient_procedures",
        "outpatient_first_attendances",
        "outpatient_followup_attendances",
        "outpatient_virtual_attendances",
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
    pass
