"""
Shared Configuration
------------------------
Single source of truth for constraints referenced across tools, agents,
and prompts -- the trained-store list and dataset date range. Keeping
these here means every prompt states them consistently, and updating
them (e.g. training more stores later) only requires a change in one
place.
"""

TRAINED_FORECAST_STORES = [1, 2, 3, 4, 5]
DATASET_START_DATE = "2013-01-01"
DATASET_END_DATE = "2015-07-31"