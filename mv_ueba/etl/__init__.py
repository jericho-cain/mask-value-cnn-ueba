"""
ETL modules for UEBA data ingestion.

This package contains data loaders for various data sources:
- cert_fixed_window: CERT Insider Threat Dataset with fixed 24-hour windows
"""

from mv_ueba.etl.cert_fixed_window import (
    CERTFixedWindowLoader,
    CERT_FEATURES,
    N_FEATURES,
)

__all__ = [
    "CERTFixedWindowLoader",
    "CERT_FEATURES",
    "N_FEATURES",
]
