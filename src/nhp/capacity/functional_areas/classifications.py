import pyspark.sql.functions as F
from pyspark.sql.column import Column


def class_age_adult() -> Column:
    return F.col("age") >= 18


def class_age_child() -> Column:
    return F.col("age") < 18


def class_ae() -> Column:
    return F.col("aedepttype") == "01"


def class_ae_resus() -> Column:
    return F.col("acuity") == "immediate-resuscitation"


def class_ae_major() -> Column:
    return F.col("acuity") == "very-urgent"


def class_ae_minor() -> Column:
    return (
        F.col("acuity").isin(
            "standard",
            "non-urgent",
            "urgent",
        )
        | F.col("acuity").isNull()
    )


def class_sdec() -> Column:
    return F.col("aedepttype") == "05"


def class_op_first() -> Column:

    return ~F.col("has_procedures") & F.col("is_first")


def class_op_follow_up() -> Column:

    return ~F.col("has_procedures") & ~F.col("is_first")


def class_has_procedure() -> Column:
    return F.col("has_procedures")


def class_op_virtual() -> Column:
    return F.col("tele_attendances")


def class_op_face_to_face() -> Column:
    return F.col("attendances")


def class_renal() -> Column:
    return F.col("tretspef") == "361"


def class_elective() -> Column:
    return F.col("admimeth").startswith("1") & F.col("classpat") == "1"


def class_zero_los() -> Column:
    return F.col("speldur") == 0


def class_regular_day_night() -> Column:
    return F.col("classpat").isin("3", "4")
