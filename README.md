<!-- badges: start -->

[![Project Status: Moved to http://example.com – The project has been moved to a new location, and the version at that location should be considered authoritative.](https://www.repostatus.org/badges/latest/moved.svg)](https://www.repostatus.org/#moved) to [https://github.com/The-Strategy-Unit/nhp_data](https://github.com/The-Strategy-Unit/nhp_data)

<!-- badges: end -->

# Functional Area Mapping

This repository contained the code for mapping results from NHP model runs to functional areas, for conversion to capacity, at spell level.

The notebook in this repository was run on MLCSU Databricks and generated parquet files saved to MLCSU Azure Blob Storage, storing the metadata for these generated files in MLCSU Azure Table Storage.

This repository has now been deprecated. A more up to date version of the code, implementing conversion of hospital activity to functional areas is available in the [nhp_data](https://github.com/The-Strategy-Unit/nhp_data) repository in the folder `src/nhp/data/functional_areas`.
