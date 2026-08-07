import pandas as pd
import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame
from pyspark.sql.column import Column

from nhp.capacity.functional_areas.classifications import (
    class_age_adult,
    class_age_child,
    class_daycase,
    class_elective,
    class_has_procedure,
    class_non_elective,
    class_surgical,
)

spark = DatabricksSession.builder.getOrCreate()
from nhp.capacity.functional_areas.processing_helpers import add_missing_groupings


def load_theatre_times(path_to_file: str) -> DataFrame:
    """Loads theatre times dataset

    Args:
        path_to_file (str): Path to theatre times Excel spreadsheet

    Returns:
        DataFrame: Spark dataframe of theatre times, with columns opcs_code,
        opcs_description and theatre_time
    """
    theatre_times_df = pd.read_excel(path_to_file, header=7, na_values="NaN")[
        [
            "Dominant Procedure (OPCS Code)",
            "Dominant Procedure (OPCS Description)",
            "Avg Theatre Time (in minutes, where Theatre time recorded)",
        ]
    ].rename(
        columns={
            "Dominant Procedure (OPCS Code)": "opcs_code",
            "Dominant Procedure (OPCS Description)": "opcs_description",
            "Avg Theatre Time (in minutes, where Theatre time recorded)": "theatre_time",
        }
    )
    theatre_times_df["theatre_time"] = pd.to_numeric(
        theatre_times_df["theatre_time"], errors="coerce"
    )
    theatre_times_df = theatre_times_df.dropna()
    return spark.createDataFrame(theatre_times_df)


def is_unknown_time() -> Column:
    return F.col("theatre_time").isNull()


def is_adult_elective_surgical_procedures() -> Column:
    return (
        class_age_adult() & class_elective() & class_has_procedure() & class_surgical()
    )


def is_adult_nonelective_surgical_procedures() -> Column:
    return (
        class_age_adult()
        & class_non_elective()
        & class_has_procedure()
        & class_surgical()
    )


def is_adult_surgical_daycase_procedures() -> Column:
    return (
        class_age_adult() & class_daycase() & class_has_procedure() & class_surgical()
    )


def is_paediatric_elective_procedures() -> Column:
    return class_age_child() & class_elective() & class_has_procedure()


def is_paediatric_nonelective_procedures() -> Column:
    return class_age_child() & class_non_elective() & class_has_procedure()


def is_paediatric_daycase_procedures() -> Column:
    return class_age_child() & class_daycase() & class_has_procedure()


GROUPINGS = [
    (
        "adult_elective_surgical_procedures",
        is_adult_elective_surgical_procedures(),
    ),
    (
        "adult_nonelective_surgical_procedures",
        is_adult_nonelective_surgical_procedures(),
    ),
    (
        "adult_surgical_daycase_procedures",
        is_adult_surgical_daycase_procedures(),
    ),
    (
        "paediatric_elective_procedures",
        is_paediatric_elective_procedures(),
    ),
    (
        "paediatric_nonelective_procedures",
        is_paediatric_nonelective_procedures(),
    ),
    (
        "paediatric_daycase_procedures",
        is_paediatric_daycase_procedures(),
    ),
]


def build_ip_theatres_grouping_column():
    """Builds the F.when(...).when(...)... chain for IP theatres"""

    when_chain = None

    for label, predicate_fn in GROUPINGS:
        if when_chain is None:
            when_chain = F.when(
                predicate_fn & is_unknown_time(),
                f"{label}_unknown_time",
            ).when(
                predicate_fn,
                label,
            )
        else:
            when_chain = when_chain.when(
                predicate_fn & is_unknown_time(),
                f"{label}_unknown_time",
            ).when(
                predicate_fn,
                label,
            )
    return when_chain


def create_ip_theatres_groupings(ip_data_with_theatre_time: DataFrame) -> DataFrame:
    """Adds "grouping" column to the IP data with the functional areas for IP theatres

    Args:
        ip_data_with_theatre_time (DataFrame): IP data with theatre time column added

    Returns:
        DataFrame: IP data, aggregated by theatres functional areas
    """
    df = (
        ip_data_with_theatre_time.withColumn(
            "grouping", build_ip_theatres_grouping_column()
        )
        .groupby("grouping", "sitetret", "model_run")
        .agg(
            F.count("rn").alias("spells"),
            F.coalesce(
                F.sum("theatre_time"),
                F.lit(0),
            ).alias("total_theatre_time"),
        )
    )
    return df


def process_ip_theatres(
    ip_original_mapped: DataFrame, ip_model_results: DataFrame, theatre_times: DataFrame
) -> DataFrame:
    """Processes and aggregates the IP baseline and model results to produce functional area outputs for IP wards,
    aggregated by model run and grouping.

    Args:
        ip_original_mapped (DataFrame): Baseline IP data
        ip_model_results (DataFrame): IP full model results
        theatre_times (DataFrame): Theatre times dataset

    Returns:
        DataFrame: IP data for each of the model runs aggregated into functional areas for theatres
    """
    ip_data_with_theatre_time = ip_original_mapped.join(
        theatre_times,
        ip_original_mapped["primary_procedure"] == theatre_times["opcs_code"],
        "left",
    )
    baseline_grouped = create_ip_theatres_groupings(ip_data_with_theatre_time)
    groupings_per_run = create_ip_theatres_groupings(
        ip_data_with_theatre_time.drop("speldur", "classpat", "model_run").join(
            ip_model_results, on="rn", how="left"
        )
    )
    final_groupings_per_run_with_baseline = groupings_per_run.unionByName(
        baseline_grouped
    )
    # Add missing groupings - we need all groupings to be present in all model runs even if value is 0
    required_ip_theatres_groupings = [grouping for grouping, _ in GROUPINGS]
    required_ip_theatres_groupings = [
        grouping + "_unknown_time" for grouping in required_ip_theatres_groupings
    ]

    final_df = add_missing_groupings(
        final_groupings_per_run_with_baseline, required_ip_theatres_groupings
    )
    return final_df
