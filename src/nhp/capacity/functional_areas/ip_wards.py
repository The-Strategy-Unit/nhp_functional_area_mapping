import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame

from nhp.capacity.functional_areas.classifications import (
    class_age_adult,
    class_age_child,
    class_elective,
    class_medical,
    class_non_elective,
    class_non_zero_los,
    class_surgical,
    class_zero_los,
)

spark = DatabricksSession.builder.getOrCreate()
from functools import reduce
from itertools import product

from nhp.capacity.functional_areas.processing_helpers import add_missing_groupings

# Each dimension: label -> predicate-producing function
AGE_DIMENSION = {
    "adult": class_age_adult,
    "paediatric": class_age_child,
}
ADMISSION_DIMENSION = {
    "elective": class_elective,
    "nonelective": class_non_elective,
}
TREATMENT_DIMENSION = {
    "medical": class_medical,
    "surgical": class_surgical,
}
LOS_DIMENSION = {
    "nonzerolos": class_non_zero_los,
    "zerolos": class_zero_los,
}

DIMENSIONS = [
    AGE_DIMENSION,
    ADMISSION_DIMENSION,
    TREATMENT_DIMENSION,
    LOS_DIMENSION,
]


def build_ward_grouping_column():
    """Builds the F.when(...).when(...)... chain from the 4 classification
    dimensions instead of 16 hand-written predicate functions."""

    combinations = product(*[d.items() for d in DIMENSIONS])

    when_chain = None

    for combo in combinations:
        labels, predicate_fns = zip(*combo)
        label = "_".join(labels)
        condition = reduce(lambda a, b: a & b, (fn() for fn in predicate_fns))

        if when_chain is None:
            when_chain = F.when(condition, label)
        else:
            when_chain = when_chain.when(condition, label)

    return when_chain


def create_ip_ward_groupings(ip_data: DataFrame) -> DataFrame:
    """Adds "grouping" column to the IP data with the functional areas for IP wards

    Args:
        ip_data (DataFrame): IP data

    Returns:
        DataFrame: IP data, aggregated by ward functional areas
    """
    df = (
        ip_data.withColumn("grouping", build_ward_grouping_column())
        .groupby("grouping", "sitetret", "model_run")
        .agg(F.count("rn").alias("spells"), F.sum("speldur").alias("beddays"))
    )
    return df


def process_ip_wards(
    ip_original_mapped: DataFrame, ip_model_results: DataFrame
) -> DataFrame:
    """Processes and aggregates the IP baseline and model results to produce functional area outputs for IP wards,
    aggregated by model run and grouping.

    Args:
        ip_original_mapped (DataFrame): Baseline IP data
        ip_model_results (DataFrame): IP full model results

    Returns:
        DataFrame: IP data for each of the model runs aggregated into functional areas for wards
    """
    baseline_grouped = create_ip_ward_groupings(ip_original_mapped)
    groupings_per_run = create_ip_ward_groupings(
        ip_original_mapped.drop("speldur", "classpat", "model_run").join(
            ip_model_results, on="rn", how="left"
        )
    )
    final_groupings_per_run_with_baseline = groupings_per_run.unionByName(
        baseline_grouped
    )
    # Add missing groupings - we need all groupings to be present in all model runs even if value is 0
    combinations = product(*[d.keys() for d in DIMENSIONS])

    required_ip_groupings = ["_".join(combo) for combo in combinations]

    final_df = add_missing_groupings(
        final_groupings_per_run_with_baseline, required_ip_groupings
    )
    return final_df
