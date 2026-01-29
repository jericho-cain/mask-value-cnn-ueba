"""
CERT Insider Threat Dataset - Fixed-Window Approach (exp005)

This module implements fixed 24-hour window loading for multi-day attack campaign
detection (1 day < duration < 1 week) with proper temporal separation between
normal training data and attack periods.

**Key Design Principles:**
- Window = 24 hours (latent point z represents 1 day of behavior)
- Target attacks: 1 day < duration < 1 week (26 users)
- Temporal buffer: 7 days before/after attack periods
- Non-overlapping windows (no sliding)
- Train on normal windows far from attacks
- Test on normal + attack windows

**Window-Level vs Trajectory-Level:**
- Window-level: pointwise metrics (AE reconstruction, off-manifold) per 24hr window
- Trajectory-level: geodesic metrics across sequences of consecutive windows

Dataset source: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247

Usage:
    from manifold_ueba.etl.cert_fixed_window import CERTFixedWindowLoader
    
    loader = CERTFixedWindowLoader(data_dir="data/cert/r4.2")
    train_data, test_data, test_labels, test_metadata = loader.load_fixed_windows(
        bucket_hours=1.0,
        window_hours=24,
        buffer_days=7,
        train_split=0.8
    )
    
    # train_data: (N_train, 24, 13) normal windows
    # test_data: (N_test, 24, 13) normal + malicious windows
    # test_labels: (N_test,) 0=normal, 1=malicious
    # test_metadata: DataFrame with user_id, window_start, scenario
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Full CERT feature schema
# NOTE: file_to_removable removed (was duplicate of file_ops - requires proper ETL from device correlation)
CERT_FEATURES = [
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
]
N_FEATURES = len(CERT_FEATURES)


class CERTFixedWindowLoader:
    """
    Loads CERT r4.2 data for fixed 24-hour window detection.
    
    Targets multi-day attack campaigns (1 day < duration < 1 week) with
    proper temporal separation between training and attack periods.
    """
    
    def __init__(self, data_dir: str, work_start_hour: int = 8, work_end_hour: int = 17):
        """
        Initialize loader.
        
        Args:
            data_dir: Path to CERT r4.2 dataset directory
            work_start_hour: Start of work hours (for after_hours detection)
            work_end_hour: End of work hours
        """
        self.data_dir = Path(data_dir)
        self.answers_path = self.data_dir / "answers" / "insiders.csv"
        self.work_start_hour = work_start_hour
        self.work_end_hour = work_end_hour
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.answers_path.exists():
            raise FileNotFoundError(f"Answers file not found: {self.answers_path}")
    
    def load_fixed_windows(
        self,
        bucket_hours: float = 1.0,
        window_hours: int = 24,
        buffer_days: int = 7,
        train_split: float = 0.8,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Load fixed 24-hour windows with CHRONOLOGICAL temporal separation.
        
        PROTOCOL FIX: Normal windows are now split chronologically (not randomly)
        to avoid temporal leakage. Training uses earlier windows, testing uses
        later windows for each user.
        
        Args:
            bucket_hours: Time bucket size within window (Ît)
            window_hours: Window size in hours (default 24)
            buffer_days: Temporal buffer around attacks (days)
            train_split: Fraction of normal windows for training (chronologically first X%)
            random_seed: Random seed (unused since split is now deterministic, kept for API compatibility)
            
        Returns:
            train_data: (N_train, T, 12) normal windows for training (chronologically earlier)
            test_data: (N_test, T, 12) normal + malicious windows (chronologically later + attacks)
            test_labels: (N_test,) labels (0=normal, 1=malicious)
            test_metadata: DataFrame with user_id, window_start, scenario
            train_metadata: DataFrame with user_id, window_start (for temporal pairing)
        """
        # Note: random_seed unused now that split is chronological (deterministic)
        # Kept in signature for backward compatibility
        pass  # Remove seed call since split is now deterministic
        
        logger.info("Loading fixed 24-hour windows for multi-day attack detection...")
        
        # Load target users (1 day < attack < 1 week)
        users_df = self._load_target_users()
        logger.info(f"Loaded {len(users_df)} target users")
        logger.info(f"Scenario breakdown: {users_df['scenario'].value_counts().to_dict()}")
        
        # Load raw activity data
        logger.info("Loading raw activity logs...")
        raw_data = self._load_raw_data()
        
        # Filter to target users
        user_ids = set(users_df['user'].tolist())
        filtered_data = {
            modality: df[df['user'].isin(user_ids)].copy()
            for modality, df in raw_data.items()
        }
        
        logger.info("Activity counts for target users:")
        for modality, df in filtered_data.items():
            logger.info(f"  {modality}: {len(df):,} events")
        
        # Aggregate into buckets
        logger.info(f"Aggregating into {bucket_hours}-hour buckets...")
        agg_data = self._aggregate_to_buckets(filtered_data, bucket_hours)
        
        # Extract fixed windows with temporal separation
        logger.info(f"Extracting {window_hours}-hour windows with {buffer_days}-day buffer...")
        train_windows, test_windows, test_labels, test_meta, train_meta = self._extract_fixed_windows(
            agg_data,
            users_df,
            bucket_hours,
            window_hours,
            buffer_days,
            train_split
        )
        
        logger.info(f"\nWindow extraction complete:")
        logger.info(f"  Training: {len(train_windows):,} normal windows")
        logger.info(f"  Test: {len(test_windows):,} windows ({test_labels.sum()} malicious, {(~test_labels.astype(bool)).sum()} normal)")
        logger.info(f"  Test imbalance: {test_labels.mean()*100:.1f}% malicious")
        
        return train_windows, test_windows, test_labels, test_meta, train_meta
    
    def _load_target_users(self) -> pd.DataFrame:
        """Load users with attacks in 1 day < duration < 1 week range."""
        df = pd.read_csv(self.answers_path)
        r42 = df[df['dataset'].astype(str).str.contains('4.2')].copy()
        
        # Parse timestamps and calculate duration
        r42['start_dt'] = pd.to_datetime(r42['start'], format='%m/%d/%Y %H:%M:%S')
        r42['end_dt'] = pd.to_datetime(r42['end'], format='%m/%d/%Y %H:%M:%S')
        r42['duration_hours'] = (r42['end_dt'] - r42['start_dt']).dt.total_seconds() / 3600
        
        # Filter to 1 day < duration < 1 week (24h < d < 168h)
        filtered = r42[(r42['duration_hours'] > 24) & (r42['duration_hours'] < 168)].copy()
        
        logger.info(f"Filtered from {len(r42)} to {len(filtered)} users (1 day < attack < 1 week)")
        
        return filtered[['user', 'scenario', 'start_dt', 'end_dt', 'duration_hours']].reset_index(drop=True)
    
    def _load_raw_data(self) -> dict:
        """Load raw CERT activity logs with columns needed for 13 features."""
        data = {}
        
        log_specs = {
            'logon': ['date', 'user', 'pc', 'activity'],
            'device': ['date', 'user', 'activity'],
            'http': ['date', 'user', 'url'],
            'file': ['date', 'user'],
            'email': ['date', 'user', 'to']
        }
        
        for modality, columns in log_specs.items():
            filepath = self.data_dir / f"{modality}.csv"
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue
            
            try:
                df = pd.read_csv(filepath, usecols=columns, on_bad_lines='skip')
                df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                df = df.dropna(subset=['date'])
                data[modality] = df
                logger.info(f"Loaded {modality}: {len(df):,} events")
            except Exception as e:
                logger.error(f"Error loading {modality}: {e}")
        
        return data
    
    def _aggregate_to_buckets(self, data: dict, bucket_hours: float) -> pd.DataFrame:
        """
        Aggregate activity into time buckets with all 13 CERT features.
        
        Returns:
            DataFrame with columns: user, bucket_start, and all 13 CERT_FEATURES
        """
        # Add bucket timestamps
        for modality in data:
            if len(data[modality]) > 0:
                data[modality]['bucket'] = data[modality]['date'].dt.floor(f'{bucket_hours}h')
        
        # Get all unique user-bucket combinations
        all_user_buckets = []
        for modality, df in data.items():
            if len(df) > 0:
                all_user_buckets.append(df[['user', 'bucket']].drop_duplicates())
        
        if not all_user_buckets:
            raise ValueError("No data loaded")
        
        user_buckets = pd.concat(all_user_buckets).drop_duplicates().reset_index(drop=True)
        
        # Initialize with zeros
        result = user_buckets.copy()
        for feature in CERT_FEATURES:
            result[feature] = 0
        
        # Compute logon features
        if 'logon' in data and len(data['logon']) > 0:
            logon_df = data['logon']
            logon_grouped = logon_df.groupby(['user', 'bucket'])
            
            logon_count = logon_grouped.apply(
                lambda x: (x['activity'] == 'Logon').sum(), include_groups=False
            ).reset_index(name='logon_count')
            result = result.merge(logon_count, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
            result['logon_count'] = result['logon_count_new'].fillna(result['logon_count'])
            result = result.drop(columns=['logon_count_new'])
            
            logoff_count = logon_grouped.apply(
                lambda x: (x['activity'] == 'Logoff').sum(), include_groups=False
            ).reset_index(name='logoff_count')
            result = result.merge(logoff_count, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
            result['logoff_count'] = result['logoff_count_new'].fillna(result['logoff_count'])
            result = result.drop(columns=['logoff_count_new'])
            
            after_hours = logon_grouped.apply(
                lambda x: ((x['activity'] == 'Logon') & 
                          ((x['date'].dt.hour < self.work_start_hour) |
                           (x['date'].dt.hour >= self.work_end_hour))).sum(),
                include_groups=False
            ).reset_index(name='after_hours_logon')
            result = result.merge(after_hours, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
            result['after_hours_logon'] = result['after_hours_logon_new'].fillna(result['after_hours_logon'])
            result = result.drop(columns=['after_hours_logon_new'])
            
            unique_pcs = logon_df.groupby(['user', 'bucket'])['pc'].nunique().reset_index(name='unique_pcs')
            result = result.merge(unique_pcs, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
            result['unique_pcs'] = result['unique_pcs_new'].fillna(result['unique_pcs'])
            result = result.drop(columns=['unique_pcs_new'])
        
        # Compute device features
        if 'device' in data and len(data['device']) > 0:
            device_df = data['device']
            device_grouped = device_df.groupby(['user', 'bucket'])
            
            device_connect = device_grouped.apply(
                lambda x: (x['activity'] == 'Connect').sum(), include_groups=False
            ).reset_index(name='device_connect')
            result = result.merge(device_connect, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
            result['device_connect'] = result['device_connect_new'].fillna(result['device_connect'])
            result = result.drop(columns=['device_connect_new'])
            
            device_disconnect = device_grouped.apply(
                lambda x: (x['activity'] == 'Disconnect').sum(), include_groups=False
            ).reset_index(name='device_disconnect')
            result = result.merge(device_disconnect, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
            result['device_disconnect'] = result['device_disconnect_new'].fillna(result['device_disconnect'])
            result = result.drop(columns=['device_disconnect_new'])
        
        # Compute HTTP features
        if 'http' in data and len(data['http']) > 0:
            http_df = data['http']
            http_grouped = http_df.groupby(['user', 'bucket'])
            
            http_count = http_grouped.size().reset_index(name='http_count')
            result = result.merge(http_count, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
            result['http_count'] = result['http_count_new'].fillna(result['http_count'])
            result = result.drop(columns=['http_count_new'])
            
            unique_urls = http_df.groupby(['user', 'bucket'])['url'].nunique().reset_index(name='unique_urls')
            result = result.merge(unique_urls, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
            result['unique_urls'] = result['unique_urls_new'].fillna(result['unique_urls'])
            result = result.drop(columns=['unique_urls_new'])
        
        # Compute email features
        if 'email' in data and len(data['email']) > 0:
            email_df = data['email']
            email_grouped = email_df.groupby(['user', 'bucket'])
            
            email_sent = email_grouped.size().reset_index(name='email_sent')
            result = result.merge(email_sent, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
            result['email_sent'] = result['email_sent_new'].fillna(result['email_sent'])
            result = result.drop(columns=['email_sent_new'])
            
            if 'to' in email_df.columns:
                email_external = email_grouped.apply(
                    lambda x: x['to'].str.contains('@(?!dtaa)', regex=True, na=False).sum(),
                    include_groups=False
                ).reset_index(name='email_external')
                result = result.merge(email_external, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
                result['email_external'] = result['email_external_new'].fillna(result['email_external'])
                result = result.drop(columns=['email_external_new'])
                
                email_internal = email_grouped.apply(
                    lambda x: x['to'].str.contains('@dtaa', regex=False, na=False).sum(),
                    include_groups=False
                ).reset_index(name='email_internal')
                result = result.merge(email_internal, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
                result['email_internal'] = result['email_internal_new'].fillna(result['email_internal'])
                result = result.drop(columns=['email_internal_new'])
        
        # Compute file features
        if 'file' in data and len(data['file']) > 0:
            file_df = data['file']
            file_grouped = file_df.groupby(['user', 'bucket'])
            
            file_ops = file_grouped.size().reset_index(name='file_ops')
            result = result.merge(file_ops, on=['user', 'bucket'], how='left', suffixes=('', '_new'))
            result['file_ops'] = result['file_ops_new'].fillna(result['file_ops'])
            result = result.drop(columns=['file_ops_new'])
        
        # Ensure all features present
        for feature in CERT_FEATURES:
            if feature in result.columns:
                result[feature] = result[feature].fillna(0).astype(int)
            else:
                result[feature] = 0
        
        result = result.rename(columns={'bucket': 'bucket_start'})
        result = result.sort_values(['user', 'bucket_start']).reset_index(drop=True)
        
        return result
    
    def _extract_fixed_windows(
        self,
        agg_data: pd.DataFrame,
        users_df: pd.DataFrame,
        bucket_hours: float,
        window_hours: int,
        buffer_days: int,
        train_split: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Extract non-overlapping 24-hour windows with temporal separation.
        
        Returns:
            train_data, test_data, test_labels, test_metadata, train_metadata
        """
        buckets_per_window = int(window_hours / bucket_hours)
        window_delta = timedelta(hours=window_hours)
        buffer_delta = timedelta(days=buffer_days)
        
        train_windows_list = []
        train_metadata_list = []
        test_windows_list = []
        test_labels_list = []
        test_metadata_list = []
        
        # Process each user
        for _, user_info in users_df.iterrows():
            user_id = user_info['user']
            attack_start = user_info['start_dt']
            attack_end = user_info['end_dt']
            scenario = user_info['scenario']
            
            # Get user data
            user_data = agg_data[agg_data['user'] == user_id].copy()
            if len(user_data) == 0:
                continue
            
            user_min_time = user_data['bucket_start'].min()
            user_max_time = user_data['bucket_start'].max()
            
            # Define buffer zones
            buffer_start = attack_start - buffer_delta
            buffer_end = attack_end + buffer_delta
            
            # Extract normal windows BEFORE buffer
            normal_before = []
            current_time = user_min_time
            while current_time + window_delta <= buffer_start:
                window_data = user_data[
                    (user_data['bucket_start'] >= current_time) &
                    (user_data['bucket_start'] < current_time + window_delta)
                ]
                
                # Create window (fill missing buckets with zeros)
                window = self._create_window(window_data, current_time, buckets_per_window, bucket_hours)
                if window is not None:
                    normal_before.append({
                        'window': window,
                        'start': current_time  # Track timestamp!
                    })
                
                # Move to next non-overlapping window
                current_time += window_delta
            
            # Extract normal windows AFTER buffer
            normal_after = []
            current_time = buffer_end
            while current_time + window_delta <= user_max_time:
                window_data = user_data[
                    (user_data['bucket_start'] >= current_time) &
                    (user_data['bucket_start'] < current_time + window_delta)
                ]
                
                window = self._create_window(window_data, current_time, buckets_per_window, bucket_hours)
                if window is not None:
                    normal_after.append({
                        'window': window,
                        'start': current_time  # Track timestamp!
                    })
                
                current_time += window_delta
            
            # Combine normal windows for this user
            user_normal_windows = normal_before + normal_after
            
            # Extract attack windows (consecutive, covering attack period)
            attack_windows = []
            current_time = attack_start.floor(f'{bucket_hours}h')
            while current_time < attack_end:
                window_data = user_data[
                    (user_data['bucket_start'] >= current_time) &
                    (user_data['bucket_start'] < current_time + window_delta)
                ]
                
                window = self._create_window(window_data, current_time, buckets_per_window, bucket_hours)
                if window is not None:
                    attack_windows.append({
                        'window': window,
                        'user': user_id,
                        'start': current_time,
                        'scenario': scenario
                    })
                
                current_time += window_delta
            
            # Split normal windows for this user (80/20) CHRONOLOGICALLY
            # CRITICAL FIX: Sort by time to avoid temporal leakage
            # Train on earlier windows, test on later windows
            if len(user_normal_windows) > 0:
                n_train = int(train_split * len(user_normal_windows))
                
                # Sort windows chronologically by start time
                sorted_windows = sorted(user_normal_windows, key=lambda x: x['start'])
                
                # Split chronologically: first 80%  train, last 20%  test
                train_windows = sorted_windows[:n_train]
                test_windows = sorted_windows[n_train:]
                
                # Add to train set with metadata
                for window_dict in train_windows:
                    train_windows_list.append(window_dict['window'])
                    train_metadata_list.append({
                        'user_id': user_id,
                        'window_start': window_dict['start']
                    })
                
                # Add to test set (normal) with actual timestamps
                for window_dict in test_windows:
                    test_windows_list.append(window_dict['window'])
                    test_labels_list.append(0)
                    test_metadata_list.append({
                        'user_id': user_id,
                        'window_start': window_dict['start'],  # Real timestamp!
                        'scenario': 0
                    })
            
            # Add attack windows to test set
            for attack_window in attack_windows:
                test_windows_list.append(attack_window['window'])
                test_labels_list.append(1)
                test_metadata_list.append({
                    'user_id': attack_window['user'],
                    'window_start': attack_window['start'],
                    'scenario': attack_window['scenario']
                })
        
        # Convert to arrays
        train_data = np.array(train_windows_list)
        train_metadata = pd.DataFrame(train_metadata_list)
        test_data = np.array(test_windows_list)
        test_labels = np.array(test_labels_list)
        test_metadata = pd.DataFrame(test_metadata_list)
        
        return train_data, test_data, test_labels, test_metadata, train_metadata
    
    def _create_window(
        self,
        window_data: pd.DataFrame,
        start_time: pd.Timestamp,
        num_buckets: int,
        bucket_hours: float
    ) -> np.ndarray:
        """
        Create a fixed window from bucket data.
        
        Returns (num_buckets, 13) array with feature values (zeros for missing buckets)
        """
        # Create all bucket timestamps for this window
        bucket_times = pd.date_range(start_time, periods=num_buckets, freq=f'{bucket_hours}h')
        
        # Initialize feature matrix
        features = np.zeros((num_buckets, N_FEATURES))
        
        # Fill in data for buckets with activity
        for i, bucket_time in enumerate(bucket_times):
            matching = window_data[window_data['bucket_start'] == bucket_time]
            if len(matching) > 0:
                for j, feature_name in enumerate(CERT_FEATURES):
                    if feature_name in matching.columns:
                        features[i, j] = matching[feature_name].iloc[0]
        
        return features
