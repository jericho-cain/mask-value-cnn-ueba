"""
CERT Insider Threat Dataset ETL Pipeline

This module provides data loading and preprocessing for the CERT Insider Threat
Dataset (r4.2) for use with the UEBA CNN Autoencoder and manifold learning pipeline.

The CERT dataset contains synthetic but realistic enterprise user behavior across
multiple modalities: logon, device, HTTP, email, and file operations. It includes
labeled insider threat scenarios suitable for evaluating anomaly detection methods.

Dataset source: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247

Usage:
    # Download dataset manually, then:
    from manifold_ueba.etl.cert import CERTDataLoader
    
    loader = CERTDataLoader(data_dir="data/cert/r4.2")
    train_data, test_data, test_labels = loader.load_splits(
        n_users_sample=50,  # For local testing
        bucket_hours=1,
        sequence_length=24
    )
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Feature schema for CERT data
CERT_FEATURES: list[str] = [
    "logon_count",
    "logoff_count",
    "after_hours_logon",
    "unique_pcs",
    "device_connect",
    "device_disconnect",
    "http_count",
    "unique_urls",
    "email_sent",
    "email_internal",
    "email_external",
    "file_ops",
    "file_to_removable",
]

N_FEATURES = len(CERT_FEATURES)


@dataclass
class CERTConfig:
    """Configuration for CERT data processing."""
    
    # Time bucketing
    bucket_hours: float = 1.0  # Aggregate events into N-hour buckets (0.25 = 15 min)
    sequence_length: int = 24  # Number of buckets per sequence (24 = 1 day if hourly)
    
    # Work hours definition (for after_hours feature)
    work_start_hour: int = 8
    work_end_hour: int = 18
    
    # Sampling
    n_users_sample: Optional[int] = None  # None = all users, int = sample N users
    random_seed: int = 42
    
    # Train/test split
    test_ratio: float = 0.2  # Fraction of normal user sequences for test set
    
    # Attack filtering
    min_attack_hours: Optional[float] = None  # Filter attacks by minimum duration (hours)


class CERTDataLoader:
    """
    Load and preprocess CERT Insider Threat Dataset for UEBA manifold learning.
    
    This class handles:
    - Loading raw CSV files from CERT r4.2 dataset
    - Aggregating events into time-bucketed features
    - Creating sequences suitable for CNN autoencoder input
    - Splitting into train (normal only) and test (normal + malicious) sets
    
    Parameters
    ----------
    data_dir : str or Path
        Path to directory containing CERT r4.2 CSV files
    config : CERTConfig, optional
        Configuration parameters for data processing
        
    Examples
    --------
    >>> loader = CERTDataLoader("data/cert/r4.2")
    >>> train, test, labels = loader.load_splits(n_users_sample=50)
    >>> print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    """
    
    def __init__(
        self, 
        data_dir: str | Path,
        config: Optional[CERTConfig] = None
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config = config or CERTConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        
        # Will be populated on load
        self._logon_df: Optional[pd.DataFrame] = None
        self._device_df: Optional[pd.DataFrame] = None
        self._http_df: Optional[pd.DataFrame] = None
        self._email_df: Optional[pd.DataFrame] = None
        self._file_df: Optional[pd.DataFrame] = None
        self._answers_df: Optional[pd.DataFrame] = None
        self._ldap_df: Optional[pd.DataFrame] = None
        
    def _check_data_exists(self) -> bool:
        """Check if required data files exist."""
        required_files = ["logon.csv", "device.csv", "http.csv", "email.csv", "file.csv"]
        missing = [f for f in required_files if not (self.data_dir / f).exists()]
        
        if missing:
            logger.error(f"Missing required files: {missing}")
            logger.info(
                "Download CERT r4.2 dataset from: "
                "https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247"
            )
            return False
        return True
    
    def _load_raw_csvs(self) -> None:
        """Load raw CSV files into DataFrames."""
        logger.info(f"Loading CERT data from {self.data_dir}")
        
        def load_csv_with_dates(filename: str) -> pd.DataFrame:
            """Load CSV and parse date column with multiple format attempts."""
            filepath = self.data_dir / filename
            df = pd.read_csv(filepath)
            
            if "date" in df.columns:
                # Try multiple date formats used in CERT datasets
                for fmt in ["%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", None]:
                    try:
                        df["date"] = pd.to_datetime(df["date"], format=fmt, errors="coerce")
                        break
                    except (ValueError, TypeError):
                        continue
            
            return df
        
        self._logon_df = load_csv_with_dates("logon.csv")
        logger.info(f"  Loaded logon.csv: {len(self._logon_df):,} rows")
        
        self._device_df = load_csv_with_dates("device.csv")
        logger.info(f"  Loaded device.csv: {len(self._device_df):,} rows")
        
        self._http_df = load_csv_with_dates("http.csv")
        logger.info(f"  Loaded http.csv: {len(self._http_df):,} rows")
        
        self._email_df = load_csv_with_dates("email.csv")
        logger.info(f"  Loaded email.csv: {len(self._email_df):,} rows")
        
        self._file_df = load_csv_with_dates("file.csv")
        logger.info(f"  Loaded file.csv: {len(self._file_df):,} rows")
        
        # Load answers (malicious user labels) if available
        answers_path = self.data_dir / "answers" / "insiders.csv"
        if not answers_path.exists():
            # Try alternative locations
            for alt in ["insiders.csv", "answers.csv"]:
                alt_path = self.data_dir / alt
                if alt_path.exists():
                    answers_path = alt_path
                    break
        
        if answers_path.exists():
            all_answers = pd.read_csv(answers_path)
            # Filter for r4.2 dataset only (dataset column contains "4.2")
            # The dataset column may be numeric (4.2) or string ("4.2")
            self._answers_df = all_answers[
                all_answers["dataset"].astype(str).str.contains("4.2", regex=False)
            ].copy()
            logger.info(f"  Loaded answers: {len(self._answers_df)} malicious users for r4.2")
            logger.info(f"    Scenario 1: {len(self._answers_df[self._answers_df['scenario'] == 1])} users")
            logger.info(f"    Scenario 2: {len(self._answers_df[self._answers_df['scenario'] == 2])} users")
            logger.info(f"    Scenario 3: {len(self._answers_df[self._answers_df['scenario'] == 3])} users")
        else:
            logger.warning("  No answers file found - cannot identify malicious users")
            self._answers_df = pd.DataFrame(columns=["user"])
    
    def _get_all_users(self) -> list[str]:
        """Get list of all unique users in the dataset."""
        users = set()
        if self._logon_df is not None:
            users.update(self._logon_df["user"].dropna().unique())
        if self._email_df is not None:
            users.update(self._email_df["user"].dropna().unique())
        return sorted(list(users))
    
    def _get_malicious_users(self, min_attack_hours: Optional[float] = None) -> set[str]:
        """Get set of malicious user IDs, optionally filtered by attack duration."""
        if self._answers_df is None or self._answers_df.empty:
            return set()
        
        # Handle different column names in answers file
        user_col = None
        for col in ["user", "user_id", "insider", "employee"]:
            if col in self._answers_df.columns:
                user_col = col
                break
        
        if user_col is None:
            logger.warning("Could not find user column in answers file")
            return set()
        
        df = self._answers_df.copy()
        
        # Filter by attack duration if specified
        if min_attack_hours is not None and "start" in df.columns and "end" in df.columns:
            df["start_dt"] = pd.to_datetime(df["start"])
            df["end_dt"] = pd.to_datetime(df["end"])
            df["duration_hours"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 3600
            
            original_count = len(df)
            df = df[df["duration_hours"] >= min_attack_hours]
            filtered_count = len(df)
            
            logger.info(f"Filtered attacks by duration >= {min_attack_hours}hr: {filtered_count}/{original_count}")
        
        return set(df[user_col].dropna().unique())
    
    def _get_date_range(self) -> tuple[datetime, datetime]:
        """Get the date range covered by the dataset."""
        all_dates = []
        
        for df in [self._logon_df, self._device_df, self._http_df, 
                   self._email_df, self._file_df]:
            if df is not None and "date" in df.columns:
                valid_dates = df["date"].dropna()
                if len(valid_dates) > 0:
                    all_dates.extend([valid_dates.min(), valid_dates.max()])
        
        if not all_dates:
            raise ValueError("No valid dates found in dataset")
        
        return min(all_dates), max(all_dates)
    
    def _precompute_user_aggregates(self, users: list[str]) -> dict[str, pd.DataFrame]:
        """
        Pre-compute hourly aggregates for selected users efficiently.
        
        Uses pandas groupby for fast aggregation instead of per-bucket filtering.
        Returns dict mapping user -> DataFrame with hourly feature aggregates.
        """
        bucket_hours = self.config.bucket_hours
        bucket_minutes = int(bucket_hours * 60)
        logger.info(f"Pre-computing aggregates ({bucket_minutes}-minute buckets)...")
        
        user_set = set(users)
        
        # Helper to create time bucket column
        def add_bucket_col(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or len(df) == 0:
                return df
            df = df[df["user"].isin(user_set)].copy()
            if len(df) == 0:
                return df
            # Use minutes for more precise bucketing (supports 15-min, 30-min, etc.)
            df["bucket"] = df["date"].dt.floor(f"{bucket_minutes}min")
            return df
        
        # Pre-filter and add bucket column to each DataFrame
        logon_df = add_bucket_col(self._logon_df)
        device_df = add_bucket_col(self._device_df)
        http_df = add_bucket_col(self._http_df)
        email_df = add_bucket_col(self._email_df)
        file_df = add_bucket_col(self._file_df)
        
        user_aggregates = {}
        
        for user in users:
            agg_data = {}
            
            # Logon aggregates
            if logon_df is not None and len(logon_df) > 0:
                user_logon = logon_df[logon_df["user"] == user]
                if len(user_logon) > 0:
                    # Group by bucket
                    logon_grouped = user_logon.groupby("bucket")
                    
                    agg_data["logon_count"] = logon_grouped.apply(
                        lambda x: (x["activity"] == "Logon").sum()
                    )
                    agg_data["logoff_count"] = logon_grouped.apply(
                        lambda x: (x["activity"] == "Logoff").sum()
                    )
                    agg_data["after_hours_logon"] = logon_grouped.apply(
                        lambda x: ((x["activity"] == "Logon") & 
                                   ((x["date"].dt.hour < self.config.work_start_hour) |
                                    (x["date"].dt.hour >= self.config.work_end_hour))).sum()
                    )
                    agg_data["unique_pcs"] = logon_grouped["pc"].nunique()
            
            # Device aggregates
            if device_df is not None and len(device_df) > 0:
                user_device = device_df[device_df["user"] == user]
                if len(user_device) > 0:
                    device_grouped = user_device.groupby("bucket")
                    agg_data["device_connect"] = device_grouped.apply(
                        lambda x: (x["activity"] == "Connect").sum()
                    )
                    agg_data["device_disconnect"] = device_grouped.apply(
                        lambda x: (x["activity"] == "Disconnect").sum()
                    )
            
            # HTTP aggregates
            if http_df is not None and len(http_df) > 0:
                user_http = http_df[http_df["user"] == user]
                if len(user_http) > 0:
                    http_grouped = user_http.groupby("bucket")
                    agg_data["http_count"] = http_grouped.size()
                    agg_data["unique_urls"] = http_grouped["url"].nunique()
            
            # Email aggregates
            if email_df is not None and len(email_df) > 0:
                user_email = email_df[email_df["user"] == user]
                if len(user_email) > 0:
                    email_grouped = user_email.groupby("bucket")
                    agg_data["email_sent"] = email_grouped.size()
                    # Simplified: count external as any with non-dtaa domain
                    if "to" in user_email.columns:
                        agg_data["email_external"] = email_grouped.apply(
                            lambda x: x["to"].str.contains("@(?!dtaa)", regex=True, na=False).sum()
                        )
                        agg_data["email_internal"] = email_grouped.apply(
                            lambda x: x["to"].str.contains("@dtaa", regex=False, na=False).sum()
                        )
            
            # File aggregates  
            if file_df is not None and len(file_df) > 0:
                user_file = file_df[file_df["user"] == user]
                if len(user_file) > 0:
                    file_grouped = user_file.groupby("bucket")
                    agg_data["file_ops"] = file_grouped.size()
                    # All file.csv entries are copies to removable media per readme
                    agg_data["file_to_removable"] = file_grouped.size()
            
            # Combine into DataFrame - use concat to preserve index
            if agg_data:
                # Each series in agg_data has bucket timestamps as index
                user_df = pd.concat(agg_data, axis=1)
                user_df = user_df.fillna(0)
                user_aggregates[user] = user_df
            else:
                user_aggregates[user] = pd.DataFrame()
        
        # Log some diagnostics
        non_empty = sum(1 for u, df in user_aggregates.items() if len(df) > 0)
        logger.info(f"Pre-computed aggregates for {len(user_aggregates)} users ({non_empty} with data)")
        
        # Show sample bucket range for first user with data
        for user, df in user_aggregates.items():
            if len(df) > 0:
                logger.info(f"  Sample user {user}: {len(df)} buckets, range {df.index.min()} to {df.index.max()}")
                break
        
        return user_aggregates
    
    def _get_bucket_features(
        self,
        user_agg: pd.DataFrame,
        bucket_time: datetime
    ) -> np.ndarray:
        """Get feature vector for a specific bucket from pre-computed aggregates."""
        features = np.zeros(N_FEATURES, dtype=np.float32)
        
        if user_agg is None or len(user_agg) == 0:
            return features
        
        bucket_time = pd.Timestamp(bucket_time)
        
        # Try exact match first
        if bucket_time in user_agg.index:
            row = user_agg.loc[bucket_time]
            for feat_name in CERT_FEATURES:
                if feat_name in row.index:
                    features[CERT_FEATURES.index(feat_name)] = row[feat_name]
        else:
            # Try finding closest bucket within the hour
            bucket_end = bucket_time + pd.Timedelta(hours=self.config.bucket_hours)
            mask = (user_agg.index >= bucket_time) & (user_agg.index < bucket_end)
            matching = user_agg[mask]
            if len(matching) > 0:
                # Sum all activity in this bucket window
                for feat_name in CERT_FEATURES:
                    if feat_name in matching.columns:
                        features[CERT_FEATURES.index(feat_name)] = matching[feat_name].sum()
        
        return features
    
    def _create_user_sequences_from_agg(
        self,
        user_agg: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> list[np.ndarray]:
        """
        Create all non-overlapping sequences for a user from pre-computed aggregates.
        
        Returns list of arrays, each shape (sequence_length, N_FEATURES).
        """
        sequences = []
        bucket_delta = timedelta(hours=self.config.bucket_hours)
        sequence_delta = bucket_delta * self.config.sequence_length
        
        current_start = start_date
        
        while current_start + sequence_delta <= end_date:
            sequence = []
            bucket_start = current_start
            
            for _ in range(self.config.sequence_length):
                features = self._get_bucket_features(user_agg, bucket_start)
                sequence.append(features)
                bucket_start += bucket_delta
            
            seq_array = np.stack(sequence, axis=0)  # (T, F)
            
            # Only include sequences with some activity
            if seq_array.sum() > 0:
                sequences.append(seq_array)
            
            current_start += sequence_delta  # Non-overlapping
        
        return sequences
    
    def load_splits(
        self,
        n_users_sample: Optional[int] = None,
        bucket_hours: Optional[float] = None,
        sequence_length: Optional[int] = None,
        min_attack_hours: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split CERT data into train and test sets.
        
        Training data contains only normal users.
        Test data contains both normal and malicious users.
        
        Parameters
        ----------
        n_users_sample : int, optional
            Number of users to sample (for local testing). None = all users.
        bucket_hours : float, optional
            Hours per time bucket. Use 0.25 for 15-min buckets. Default from config.
        sequence_length : int, optional
            Number of buckets per sequence. Default from config.
        min_attack_hours : float, optional
            Minimum attack duration in hours to include. Filters out shorter attacks.
            
        Returns
        -------
        train_data : np.ndarray
            Training sequences, shape (N_train, sequence_length, N_FEATURES)
        test_data : np.ndarray
            Test sequences, shape (N_test, sequence_length, N_FEATURES)
        test_labels : np.ndarray
            Binary labels for test data, shape (N_test,). 1 = malicious, 0 = normal.
        """
        # Update config if parameters provided
        if n_users_sample is not None:
            self.config.n_users_sample = n_users_sample
        if bucket_hours is not None:
            self.config.bucket_hours = bucket_hours
        if sequence_length is not None:
            self.config.sequence_length = sequence_length
        if min_attack_hours is not None:
            self.config.min_attack_hours = min_attack_hours
        
        # Check data exists
        if not self._check_data_exists():
            raise FileNotFoundError(
                f"CERT data not found in {self.data_dir}. "
                "Download from: https://kilthub.cmu.edu/articles/dataset/"
                "Insider_Threat_Test_Dataset/12841247"
            )
        
        # Load raw data
        self._load_raw_csvs()
        
        # Get users and malicious set
        all_users = self._get_all_users()
        malicious_users = self._get_malicious_users(min_attack_hours=self.config.min_attack_hours)
        normal_users = [u for u in all_users if u not in malicious_users]
        
        logger.info(f"Total users: {len(all_users)}")
        logger.info(f"Normal users: {len(normal_users)}")
        logger.info(f"Malicious users: {len(malicious_users)}")
        
        # Sample users if requested
        if self.config.n_users_sample is not None:
            n_sample = min(self.config.n_users_sample, len(all_users))
            
            # Proportional sampling to maintain ratio
            n_malicious_sample = max(1, int(n_sample * len(malicious_users) / len(all_users)))
            n_normal_sample = n_sample - n_malicious_sample
            
            if len(malicious_users) > 0:
                sampled_malicious = list(self.rng.choice(
                    list(malicious_users), 
                    size=min(n_malicious_sample, len(malicious_users)),
                    replace=False
                ))
            else:
                sampled_malicious = []
            
            sampled_normal = list(self.rng.choice(
                normal_users,
                size=min(n_normal_sample, len(normal_users)),
                replace=False
            ))
            
            normal_users = sampled_normal
            malicious_users = set(sampled_malicious)
            
            logger.info(f"Sampled {len(normal_users)} normal, {len(malicious_users)} malicious users")
        
        # Get date range
        start_date, end_date = self._get_date_range()
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Pre-compute aggregates for all selected users (fast)
        all_selected_users = normal_users + list(malicious_users)
        user_aggregates = self._precompute_user_aggregates(all_selected_users)
        
        # Create sequences for normal users
        logger.info("Creating sequences for normal users...")
        normal_sequences = []
        for i, user in enumerate(normal_users):
            if (i + 1) % 10 == 0:
                logger.info(f"  Processing user {i + 1}/{len(normal_users)}")
            user_agg = user_aggregates.get(user, pd.DataFrame())
            user_seqs = self._create_user_sequences_from_agg(user_agg, start_date, end_date)
            normal_sequences.extend(user_seqs)
        
        logger.info(f"Created {len(normal_sequences)} normal sequences")
        
        # Create sequences for malicious users
        logger.info("Creating sequences for malicious users...")
        malicious_sequences = []
        for user in malicious_users:
            user_agg = user_aggregates.get(user, pd.DataFrame())
            user_seqs = self._create_user_sequences_from_agg(user_agg, start_date, end_date)
            malicious_sequences.extend(user_seqs)
        
        logger.info(f"Created {len(malicious_sequences)} malicious sequences")
        
        # Split normal sequences into train and test
        n_normal_test = int(len(normal_sequences) * self.config.test_ratio)
        self.rng.shuffle(normal_sequences)
        
        normal_test = normal_sequences[:n_normal_test]
        normal_train = normal_sequences[n_normal_test:]
        
        # Combine test set
        test_sequences = normal_test + malicious_sequences
        test_labels = np.array(
            [0] * len(normal_test) + [1] * len(malicious_sequences),
            dtype=np.int32
        )
        
        # Shuffle test set
        test_indices = self.rng.permutation(len(test_sequences))
        test_sequences = [test_sequences[i] for i in test_indices]
        test_labels = test_labels[test_indices]
        
        # Convert to arrays
        train_data = np.stack(normal_train, axis=0) if normal_train else np.array([])
        test_data = np.stack(test_sequences, axis=0) if test_sequences else np.array([])
        
        logger.info(f"Final splits - Train: {len(train_data)}, Test: {len(test_data)}")
        logger.info(f"Test composition - Normal: {(test_labels == 0).sum()}, Malicious: {(test_labels == 1).sum()}")
        
        return train_data, test_data, test_labels
    
    def get_feature_names(self) -> list[str]:
        """Return the feature names in order."""
        return CERT_FEATURES.copy()


def download_cert_dataset(dest_dir: str | Path) -> None:
    """
    Provide instructions for downloading CERT dataset.
    
    The CERT Insider Threat Dataset requires manual download due to licensing.
    
    Parameters
    ----------
    dest_dir : str or Path
        Directory where dataset should be placed
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    instructions = """
    CERT Insider Threat Dataset Download Instructions
    ==================================================
    
    The CERT dataset requires manual download from CMU's data repository.
    
    1. Visit: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
    
    2. Download the r4.2 version (or desired version)
    
    3. Extract the archive to: {dest_dir}
    
    4. Verify you have these files:
       - logon.csv
       - device.csv
       - http.csv
       - email.csv
       - file.csv
       - answers/insiders.csv (or similar)
    
    5. Run the data loader:
       
       from manifold_ueba.etl.cert import CERTDataLoader
       loader = CERTDataLoader("{dest_dir}")
       train, test, labels = loader.load_splits(n_users_sample=50)
    
    """.format(dest_dir=dest_dir)
    
    print(instructions)
    
    # Write instructions to file
    readme_path = dest_dir / "README_DOWNLOAD.txt"
    with open(readme_path, "w") as f:
        f.write(instructions)
    
    print(f"Instructions saved to: {readme_path}")
