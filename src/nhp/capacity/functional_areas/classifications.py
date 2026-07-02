import pyspark.sql.functions as F
from pyspark.sql.column import Column

ENDOSCOPY_PROCEDURE_CODES = [
    "G14",
    "G15",
    "G16",
    "G17",
    "G18",
    "G19",
    "G20",
    "G42",
    "G43",
    "G44",
    "G45",
    "G46",
    "G54",
    "G55",
    "G64",
    "G65",
    "G79",
    "G80",
    "H23",
    "H24",
    "H25",
    "H26",
    "H27",
    "H28",
    "H37",
    "H69",
    "H70",
    "H71",
    "J38",
    "J39",
    "J40",
    "J41",
    "J42",
    "J43",
    "J44",
    "J45",
    "H20",
    "H21",
    "H22",
    "H68",
    "E48",
    "E49",
    "E50",
    "E51",
]


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


def class_daycase() -> Column:
    return (F.col("admimeth").isin("11", "12", "13")) & (F.col("classpat") == "2")


def class_haem_onc() -> Column:
    return F.col("tretspef").isin("253", "303", "260", "370", "800")


def class_endoscopy() -> Column:
    return F.col("primary_procedure").substr(1, 3).isin(ENDOSCOPY_PROCEDURE_CODES)


def class_medical() -> Column:
    return F.col("tretspef_type") == "Surgical"


def class_surgical() -> Column:
    return F.col("tretspef_type") == "Medical/Other"
