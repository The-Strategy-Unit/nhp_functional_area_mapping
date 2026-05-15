from typing import List

import pandas as pd
import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame

from nhp.capacity.functional_areas.processing_helpers import add_missing_groupings

spark = DatabricksSession.builder.getOrCreate()


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


def create_ip_theatre_groupings(
    ip_data: DataFrame, theatre_times: DataFrame
) -> DataFrame:
    """Adds "grouping" column to the IP data with the functional areas for IP theatres

    Args:
        ip_data (DataFrame): Inpatients data
        theatre_times (DataFrame): Theatre times dataset

    Returns:
        DataFrame: IP data, aggregated by theatres functional areas
    """
    procedures_only_no_maternity = ip_data.filter(
        F.col("has_procedure") & (F.col("group") != "maternity")
    )
    procedures_only_no_maternity = procedures_only_no_maternity.join(
        theatre_times,
        procedures_only_no_maternity["primary_procedure"] == theatre_times["opcs_code"],
        "left",
    )
    df_with_grouping = (
        procedures_only_no_maternity.withColumn(
            "grouping",
            F.when(
                (F.col("pod") == "ip_elective_admission")
                & (F.col("admiage") > 17)
                & F.col("theatre_time").isNotNull(),
                "adult_elective_surgical_procedures",
            )
            .when(
                (F.col("pod") == "ip_elective_admission")
                & (F.col("admiage") <= 17)
                & F.col("theatre_time").isNotNull(),
                "paediatric_elective_surgical_procedures",
            )
            .when(
                (F.col("pod") == "ip_non-elective_admission")
                & (F.col("admiage") > 17)
                & F.col("theatre_time").isNotNull(),
                "adult_nonelective_surgical_procedures",
            )
            .when(
                (F.col("pod") == "ip_non-elective_admission")
                & (F.col("admiage") <= 17)
                & F.col("theatre_time").isNotNull(),
                "paediatric_nonelective_surgical_procedures",
            )
            .when(
                (F.col("pod") == "ip_elective_daycase")
                & (F.col("admiage") > 17)
                & F.col("theatre_time").isNotNull(),
                "adult_surgical_daycase",
            )
            .when(
                (F.col("pod") == "ip_elective_daycase")
                & (F.col("admiage") <= 17)
                & F.col("theatre_time").isNotNull(),
                "paediatric_surgical_daycase",
            )
            .when(
                (F.col("pod") == "ip_elective_admission")
                & (F.col("admiage") > 17)
                & F.col("theatre_time").isNull(),
                "adult_elective_surgical_procedures_unknown_time",
            )
            .when(
                (F.col("pod") == "ip_elective_admission")
                & (F.col("admiage") <= 17)
                & F.col("theatre_time").isNull(),
                "paediatric_elective_surgical_procedures_unknown_time",
            )
            .when(
                (F.col("pod") == "ip_non-elective_admission")
                & (F.col("admiage") > 17)
                & F.col("theatre_time").isNull(),
                "adult_nonelective_surgical_procedures_unknown_time",
            )
            .when(
                (F.col("pod") == "ip_non-elective_admission")
                & (F.col("admiage") <= 17)
                & F.col("theatre_time").isNull(),
                "paediatric_nonelective_surgical_procedures_unknown_time",
            )
            .when(
                (F.col("pod") == "ip_elective_daycase")
                & (F.col("admiage") > 17)
                & F.col("theatre_time").isNull(),
                "adult_surgical_daycase_unknown_time",
            )
            .when(
                (F.col("pod") == "ip_elective_daycase")
                & (F.col("admiage") <= 17)
                & F.col("theatre_time").isNull(),
                "paediatric_surgical_daycase_unknown_time",
            ),
        )
        .groupby("grouping", "model_run")
        .agg(F.count("rn").alias("total"), F.sum("theatre_time").alias("theatre_time"))
    )
    return df_with_grouping


def process_ip_theatres(
    ip_original_mapped: DataFrame, ip_model_results: DataFrame, theatre_times: DataFrame
) -> DataFrame:
    """Processes and aggregates the IP baseline and model results to produce functional area outputs for IP theatres,
    aggregated by model run and grouping.

    Args:
        ip_original_mapped (DataFrame): Baseline IP data
        ip_model_results (DataFrame): IP full model results
        theatre_times (DataFrame): Theatre times dataset

    Returns:
        DataFrame: IP data for each of the model runs aggregated into functional areas for IP theatres
    """
    baseline_grouped = create_ip_theatre_groupings(ip_original_mapped, theatre_times)
    groupings_per_run = create_ip_theatre_groupings(
        ip_original_mapped.drop("speldur", "classpat", "model_run").join(
            ip_model_results, on="rn", how="left"
        ),
        theatre_times,
    )
    final_groupings_per_run_with_baseline = groupings_per_run.unionByName(
        baseline_grouped
    )
    # Add missing groupings - we need all groupings to be present in all model runs even if value is 0
    required_ip_groupings = [
        "adult_elective_surgical_procedures",
        "adult_elective_surgical_procedures_unknown_time",
        "adult_nonelective_surgical_procedures",
        "adult_nonelective_surgical_procedures_unknown_time",
        "adult_surgical_daycase",
        "adult_surgical_daycase_unknown_time",
        "paediatric_elective_surgical_procedures",
        "paediatric_elective_surgical_procedures_unknown_time",
        "paediatric_nonelective_surgical_procedures",
        "paediatric_nonelective_surgical_procedures_unknown_time",
        "paediatric_surgical_daycase",
        "paediatric_surgical_daycase_unknown_time",
    ]
    final_df = add_missing_groupings(
        final_groupings_per_run_with_baseline, required_ip_groupings
    )
    return final_df


def qa_ip_theatre_results(
    default_results: DataFrame, final_ip_theatres_df: DataFrame, sites: List[str]
):
    """Quality Assurance step: checks that values in a given column produce the same mean in new functional area
    aggregations as with default model results.

    Args:
        default_results (DataFrame): Default model results
        final_ip_theatres_df (DataFrame): DataFrame of functional area pipeline outputs
        sites (List[str]):
    """
    if "ALL" not in sites:
        default_results = default_results.where((F.col("sitetret").isin(sites)))
    default_results = default_results.where(
        (F.col("measure") == "procedures") & (~F.col("pod").like("%maternity%"))
    )
    default_results_value = (
        default_results.groupBy("model_run")
        .agg(F.sum("value").alias("value"))
        .agg(F.mean("value").alias("mean"))
        .collect()[0][0]
    )
    grouped_results_value = (
        final_ip_theatres_df.filter(F.col("model_run") != 0)  # model_run 0 is baseline
        .groupBy("model_run")
        .agg(F.sum("total").alias("total"))
        .agg(F.mean("total").alias("mean_total"))
        .collect()[0][0]
    )
    try:
        assert float(default_results_value) == float(grouped_results_value)
    except AssertionError:
        print(
            "Aggregated results are not aligned with default model results. Check theatre calculations"
        )
        raise
