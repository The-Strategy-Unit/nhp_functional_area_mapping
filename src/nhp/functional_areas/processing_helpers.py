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
