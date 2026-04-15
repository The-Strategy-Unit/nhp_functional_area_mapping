from typing import List

import pyspark.sql.functions as F
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame

from nhp.capacity.functional_areas.processing_helpers import add_missing_groupings

spark = DatabricksSession.builder.getOrCreate()


def create_ip_ward_groupings(ip_data: DataFrame) -> DataFrame:
    """Adds "grouping" column to the IP data with the functional areas for IP wards

    Args:
        ip_data (DataFrame): IP data

    Returns:
        DataFrame: IP data, aggregated by ward functional areas
    """
    filtered = ip_data.filter(
        F.col("pod").isin(["ip_non-elective_admission", "ip_elective_admission"])
    )
    df_with_grouping = filtered.withColumn(
        "grouping",
        F.concat_ws(
            "_",
            # age group
            F.when(F.col("admiage") > 17, "adult").otherwise("paediatric"),
            # admission type
            F.when(F.col("pod") == "ip_non-elective_admission", "nonelective").when(
                F.col("pod") == "ip_elective_admission", "elective"
            ),
            # specialty
            F.when(F.col("tretspef_type") == "Surgical", "surgical").when(
                F.col("tretspef_type") == "Medical/Other", "medical"
            ),
            # optional 0los suffix
            F.when(F.col("speldur") == 0, "0los"),
        ),
    )
    df_grouped = df_with_grouping.groupBy("model_run", "grouping").agg(
        F.sum("speldur").alias("beddays"), F.count("*").alias("spells")
    )
    df_final = (
        df_grouped.select(
            "model_run",
            F.concat(F.col("grouping"), F.lit("_beddays")).alias("grouping_beddays"),
            F.col("beddays"),
            F.concat(F.col("grouping"), F.lit("_spells")).alias("grouping_spells"),
            F.col("spells"),
        )
        .select(
            "model_run",
            F.explode(
                F.array(
                    F.struct(
                        F.col("grouping_beddays").alias("grouping"),
                        F.col("beddays").alias("total"),
                    ),
                    F.struct(
                        F.col("grouping_spells").alias("grouping"),
                        F.col("spells").alias("total"),
                    ),
                )
            ).alias("kv"),
        )
        .select("model_run", F.col("kv.grouping"), F.col("kv.total"))
    )
    return df_final


def process_ip_wards(
    ip_original_mapped: DataFrame, ip_model_results: DataFrame
) -> DataFrame:
    """Processes and aggregates the IP baseline and model results to produce functional area outputs for IP daycase,
    aggregated by model run and grouping.

    Args:
        ip_original_mapped (DataFrame): Baseline IP data
        ip_model_results (DataFrame): IP full model results

    Returns:
        DataFrame: IP data for each of the model runs aggregated into functional areas for IP wards
    """
    baseline_grouped = create_ip_ward_groupings(ip_original_mapped)
    groupings_per_run = create_ip_ward_groupings(
        ip_original_mapped.drop("speldur", "classpat", "model_run")
        .join(ip_model_results, on="rn", how="left")
        .withColumn(
            "pod",
            F.when(F.col("classpat") == "2", "ip_elective_daycase").otherwise(
                F.col("pod")
            ),
        )
    )
    final_groupings_per_run_with_baseline = groupings_per_run.unionByName(
        baseline_grouped
    )
    # Add missing groupings - we need all groupings to be present in all model runs even if value is 0
    required_ip_groupings = [
        "paediatric_nonelective_surgical_beddays",
        "paediatric_nonelective_surgical_spells",
        "adult_nonelective_medical_0los_beddays",
        "adult_nonelective_medical_0los_spells",
        "adult_nonelective_medical_beddays",
        "adult_nonelective_medical_spells",
        "adult_elective_surgical_0los_beddays",
        "adult_elective_surgical_0los_spells",
        "paediatric_nonelective_medical_0los_beddays",
        "paediatric_nonelective_medical_0los_spells",
        "adult_elective_medical_0los_beddays",
        "adult_elective_medical_0los_spells",
        "paediatric_elective_medical_0los_beddays",
        "paediatric_elective_medical_0los_spells",
        "paediatric_nonelective_surgical_0los_beddays",
        "paediatric_nonelective_surgical_0los_spells",
        "paediatric_elective_surgical_0los_beddays",
        "paediatric_elective_surgical_0los_spells",
        "adult_nonelective_surgical_0los_beddays",
        "adult_nonelective_surgical_0los_spells",
        "paediatric_elective_surgical_beddays",
        "paediatric_elective_surgical_spells",
        "adult_nonelective_surgical_beddays",
        "adult_nonelective_surgical_spells",
        "paediatric_nonelective_medical_beddays",
        "paediatric_nonelective_medical_spells",
        "paediatric_elective_medical_beddays",
        "paediatric_elective_medical_spells",
        "adult_elective_surgical_beddays",
        "adult_elective_surgical_spells",
        "adult_elective_medical_beddays",
        "adult_elective_medical_spells",
    ]
    final_df = add_missing_groupings(
        final_groupings_per_run_with_baseline, required_ip_groupings
    )
    return final_df


def qa_ip_wards_results(
    default_results: DataFrame, final_ip_wards_df: DataFrame, sites: List[str]
):
    """Quality Assurance step: checks that values in a given column produce the same mean in new functional area
    aggregations as with default model results.

    Args:
        default_results (DataFrame): Default model results
        final_ip_wards_df (DataFrame): DataFrame of functional area pipeline outputs
        sites (List[str]):
    """
    if "ALL" not in sites:
        default_results = default_results.where((F.col("sitetret").isin(sites)))
    default_results = default_results.filter(
        F.col("pod").isin(["ip_non-elective_admission", "ip_elective_admission"])
    )
    default_results_value = (
        default_results.groupBy("model_run", "measure")
        .agg(F.sum("value").alias("value"))
        .groupby("measure")
        .agg(F.mean("value").alias("mean"))
    )
    default_results_dict = dict(
        default_results_value.select("measure", "mean").collect()
    )
    grouped_value = (
        final_ip_wards_df.filter(F.col("model_run") != 0)
        .withColumn(
            "measure",
            F.when(F.col("grouping").endswith("beddays"), F.lit("beddays")).when(
                F.col("grouping").endswith("spells"), F.lit("admissions")
            ),
        )
        .groupBy("model_run", "measure")
        .agg(F.sum("total").alias("total"))
        .groupBy("measure")
        .agg(F.mean("total").alias("mean"))
    )
    grouped_dict = dict(grouped_value.select("measure", "mean").collect())
    try:
        adjusted_beddays_default = (
            default_results_dict["beddays"] - default_results_dict["admissions"]
        )
        assert float(adjusted_beddays_default) == float(grouped_dict["beddays"])
        assert float(default_results_dict["admissions"]) == float(
            grouped_dict["admissions"]
        )
    except AssertionError:
        print(
            "Aggregated results are not aligned with default model results. Check IP wards calculations"
        )
        raise
