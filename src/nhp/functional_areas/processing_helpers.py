from pyspark.sql.streaming.proto.StateMessage_pb2 import DATA_PROCESSED
from databricks.connect import DatabricksSession
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

spark = DatabricksSession.builder.getOrCreate()


def add_missing_groupings(
    final_groupings_per_run_with_baseline: DataFrame,
    measure_column: str,
    required_groupings: list,
) -> DataFrame:
    """Add missing functional area groupings into DataFrame, with value 0

    Args:
        final_groupings_per_run_with_baseline (DataFrame): Fully processed dataframe
        measure_column (str): Column name in DataFrame with values to be aggregated
        required_groupings (list): Full list of all groupings

    Returns:
        DataFrame: DataFrame with required functional area groupings for every run
    """
    model_runs = final_groupings_per_run_with_baseline.select("model_run").distinct()
    required_df = spark.createDataFrame(
        [(g,) for g in required_groupings], ["grouping"]
    )
    expected = model_runs.crossJoin(required_df)
    completed_required = expected.join(
        final_groupings_per_run_with_baseline, on=["model_run", "grouping"], how="left"
    ).withColumn(measure_column, F.coalesce(F.col(measure_column), F.lit(0)))
    non_required = final_groupings_per_run_with_baseline.filter(
        ~F.col("grouping").isin(required_groupings)
    )
    final_df = non_required.unionByName(completed_required)
    return final_df


def qa_results(
    original_model_results: DataFrame,
    final_groupings_per_run_with_baseline: DataFrame,
    column_of_interest: str,
):
    """Quality Assurance step: checks that values in a given column produce the same mean in new functional area
    aggregations as with default model results.

    Args:
        original_model_results (DataFrame): Default model results
        final_groupings_per_run_with_baseline (DataFrame): _description_
        column_of_interest (str): _description_
    """
    default_results = (
        original_model_results.groupBy("model_run")
        .agg(F.sum(column_of_interest).alias(column_of_interest))
        .agg(F.mean(column_of_interest).alias(f"mean_{column_of_interest}"))
        .collect()[0][0]
    )
    grouped_results = (
        final_groupings_per_run_with_baseline.filter(
            F.col("model_run") != 0
        )  # model_run 0 is baseline
        .groupBy("model_run")
        .agg(F.sum(column_of_interest).alias(column_of_interest))
        .agg(F.mean(column_of_interest).alias(f"mean_{column_of_interest}"))
        .collect()[0][0]
    )
    try:
        assert float(default_results) == float(grouped_results)
    except AssertionError:
        print(
            f"Aggregated results are not aligned with default model results. Check {column_of_interest}"
        )
        raise
