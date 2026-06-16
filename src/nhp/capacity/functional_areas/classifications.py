import pyspark.sql.functions as F
from pyspark.sql.column import Column


def class_age_adult() -> Column:
    return F.col("age") >= 18


def class_age_child() -> Column:
    return F.col("age") < 18
