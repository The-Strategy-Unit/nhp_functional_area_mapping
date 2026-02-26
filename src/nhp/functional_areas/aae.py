import pyspark.sql.functions as F
from pyspark.sql import DataFrame


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
