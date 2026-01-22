"""
Data generation and preprocessing for UEBA LSTM Autoencoder demos.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# ----------------------------
# Global schema
# ----------------------------

FEATURES: list[str] = [
    "okta_login_cnt",
    "okta_fail_rate",
    "okta_mfa_rate",
    "okta_geo_switches",
    "edr_proc_cnt",
    "edr_net_bytes",
    "edr_admin_events",
    "edr_filemods",
    "email_out_cnt",
    "email_in_cnt",
    "email_ext_ratio",
    "email_link_ratio",
    "delta_okta_to_edr_secs",
]
F = len(FEATURES)

# Reproducible RNG for synthetic data
rng = np.random.default_rng(7)


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a scalar to the closed interval [lo, hi]."""
    return max(lo, min(hi, x))


# ----------------------------
# Timestep generators
# ----------------------------


def make_timestep(kind: str = "normal") -> np.ndarray:
    """
    Generate a synthetic UEBA feature vector for one time bucket across multiple systems.

    Produces an aggregated snapshot for a single time interval (e.g., 15 minutes).
    Output includes Okta/SSO, EDR, and Email metrics, plus a cross-system timing feature.

    Modes
    -----
    - kind="normal": sample benign behavior from simple distributions.
    - kind in {'spray_and_exfil', 'geo_impossible_travel'}: start from normal, then
      modify a subset of features to simulate suspicious cross-system patterns.

    Parameters
    ----------
    kind : str, optional
        "normal" or a supported anomaly label. Unsupported strings fall back to "normal".

    Returns
    -------
    np.ndarray
        Shape (13,), dtype float32. See `FEATURES` for order.
    """
    if kind == "normal":
        return _generate_normal()
    return _generate_anomalous(kind)


def _generate_normal() -> np.ndarray:
    """Helper: create a normal baseline timestep (shape (13,), float32)."""
    okta_login_cnt = rng.poisson(8)
    okta_fail_rate = _clamp(rng.normal(0.05, 0.03), 0.0, 0.3)
    okta_mfa_rate = _clamp(rng.normal(0.70, 0.07), 0.3, 0.95)
    okta_geo_switches = rng.poisson(0.2)

    edr_proc_cnt = max(0, rng.poisson(250))
    edr_net_bytes = max(0, int(rng.normal(8e6, 2e6)))
    edr_admin_events = rng.poisson(0.3)
    edr_filemods = rng.poisson(30)

    email_out_cnt = rng.poisson(4)
    email_in_cnt = rng.poisson(12)
    email_ext_ratio = _clamp(rng.normal(0.35, 0.10), 0.0, 1.0)
    email_link_ratio = _clamp(rng.normal(0.12, 0.05), 0.0, 1.0)

    delta_okta_to_edr_secs = max(0, rng.normal(45, 15))

    return np.array(
        [
            okta_login_cnt,
            okta_fail_rate,
            okta_mfa_rate,
            okta_geo_switches,
            edr_proc_cnt,
            edr_net_bytes,
            edr_admin_events,
            edr_filemods,
            email_out_cnt,
            email_in_cnt,
            email_ext_ratio,
            email_link_ratio,
            delta_okta_to_edr_secs,
        ],
        dtype=np.float32,
    )


def make_diverse_timestep(
    archetype: str = "general", 
    time_of_day: int = 12, 
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate diverse synthetic UEBA feature vectors with user archetypes and temporal patterns.
    
    This creates more realistic behavioral diversity needed for manifold learning,
    where different user types and temporal patterns create geometric structure
    in the latent space.
    
    Parameters
    ----------
    archetype : str
        User type: 'developer', 'executive', 'support', 'analyst', 'general'
    time_of_day : int  
        Hour of day (0-23) for temporal variation
    noise_level : float
        Amount of random noise to add (0.0 to 1.0)
        
    Returns
    -------
    np.ndarray
        Shape (13,), diverse behavioral features
    """
    # Start with base normal pattern
    okta_login_cnt = rng.poisson(8)
    okta_fail_rate = _clamp(rng.normal(0.05, 0.03), 0.0, 0.3)
    okta_mfa_rate = _clamp(rng.normal(0.70, 0.07), 0.3, 0.95)
    okta_geo_switches = rng.poisson(0.2)

    edr_proc_cnt = max(0, rng.poisson(250))
    edr_net_bytes = max(0, int(rng.normal(8e6, 2e6)))
    edr_admin_events = rng.poisson(0.3)
    edr_filemods = rng.poisson(30)

    email_out_cnt = rng.poisson(4)
    email_in_cnt = rng.poisson(12)
    email_ext_ratio = _clamp(rng.normal(0.35, 0.10), 0.0, 1.0)
    email_link_ratio = _clamp(rng.normal(0.12, 0.05), 0.0, 1.0)

    delta_okta_to_edr_secs = max(0, rng.normal(45, 15))
    
    # Apply archetype-specific patterns
    if archetype == "developer":
        # Developers: High EDR activity, more logins, less email
        edr_proc_cnt = max(0, rng.poisson(400))  # More processes
        edr_filemods = rng.poisson(80)  # More file modifications  
        okta_login_cnt = rng.poisson(12)  # More frequent logins
        email_out_cnt = rng.poisson(2)  # Less email
        email_ext_ratio = _clamp(rng.normal(0.15, 0.05), 0.0, 1.0)  # Less external
        
    elif archetype == "executive":  
        # Executives: High email volume, lower EDR, more external contacts
        email_out_cnt = rng.poisson(15)  # Much more email
        email_in_cnt = rng.poisson(25)
        email_ext_ratio = _clamp(rng.normal(0.60, 0.10), 0.0, 1.0)  # More external
        edr_proc_cnt = max(0, rng.poisson(150))  # Less EDR activity
        okta_login_cnt = rng.poisson(6)  # Fewer logins
        
    elif archetype == "support":
        # Support: Moderate across all systems, more admin events
        edr_admin_events = rng.poisson(2)  # More admin activity
        email_out_cnt = rng.poisson(8)  # Moderate email
        okta_login_cnt = rng.poisson(10)  # Frequent logins
        
    elif archetype == "analyst":
        # Analysts: High email, moderate EDR, data-focused
        email_out_cnt = rng.poisson(7)
        email_in_cnt = rng.poisson(18)  # Reading lots of reports
        edr_net_bytes = max(0, int(rng.normal(12e6, 3e6)))  # More data transfer
        email_link_ratio = _clamp(rng.normal(0.20, 0.08), 0.0, 1.0)  # More links
        
    # Apply time-of-day effects
    work_hour_factor = _get_work_hour_factor(time_of_day)
    
    okta_login_cnt = int(okta_login_cnt * work_hour_factor)
    email_out_cnt = int(email_out_cnt * work_hour_factor)  
    email_in_cnt = int(email_in_cnt * work_hour_factor)
    edr_proc_cnt = int(edr_proc_cnt * (0.3 + 0.7 * work_hour_factor))  # Some background activity
    
    # Add controlled random noise for manifold diversity
    features = np.array([
        okta_login_cnt,
        okta_fail_rate,  
        okta_mfa_rate,
        okta_geo_switches,
        edr_proc_cnt,
        edr_net_bytes,
        edr_admin_events,
        edr_filemods,
        email_out_cnt,
        email_in_cnt,
        email_ext_ratio,
        email_link_ratio,
        delta_okta_to_edr_secs,
    ], dtype=np.float32)
    
    # Apply multiplicative noise to maintain feature relationships
    if noise_level > 0:
        noise_multipliers = rng.normal(1.0, noise_level, size=features.shape)
        features = features * np.maximum(noise_multipliers, 0.1)  # Prevent negative
        
    # Ensure non-negative where appropriate
    features[0] = max(0, features[0])  # okta_login_cnt
    features[3] = max(0, features[3])  # okta_geo_switches  
    features[4] = max(0, features[4])  # edr_proc_cnt
    features[5] = max(0, features[5])  # edr_net_bytes
    features[6] = max(0, features[6])  # edr_admin_events
    features[7] = max(0, features[7])  # edr_filemods
    features[8] = max(0, features[8])  # email_out_cnt
    features[9] = max(0, features[9])  # email_in_cnt
    features[12] = max(0, features[12])  # delta_okta_to_edr_secs
    
    # Clamp ratios
    features[1] = _clamp(features[1], 0.0, 1.0)  # okta_fail_rate
    features[2] = _clamp(features[2], 0.0, 1.0)  # okta_mfa_rate  
    features[10] = _clamp(features[10], 0.0, 1.0)  # email_ext_ratio
    features[11] = _clamp(features[11], 0.0, 1.0)  # email_link_ratio
    
    return features


def _get_work_hour_factor(hour: int) -> float:
    """Get activity factor based on hour of day (0-23)."""
    if 9 <= hour <= 17:  # Core work hours
        return 1.0
    elif 7 <= hour <= 19:  # Extended work hours  
        return 0.7
    elif 6 <= hour <= 21:  # Early/late work
        return 0.3
    else:  # Off hours
        return 0.1


def generate_diverse_sequence(
    archetype: str = "general",
    seq_len: int = 24, 
    anomalous: bool = False,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate a diverse behavioral sequence with temporal patterns.
    
    Parameters
    ----------
    archetype : str
        User archetype for consistent behavioral patterns
    seq_len : int
        Sequence length (number of timesteps) 
    anomalous : bool
        Whether to inject coordinated attack pattern
    noise_level : float
        Random variation level
        
    Returns  
    -------
    np.ndarray
        Shape (seq_len, 13), diverse behavioral sequence
    """
    sequence = []
    
    # Generate realistic daily pattern starting at random hour
    start_hour = rng.integers(0, 24)
    
    for t in range(seq_len):
        # Progress through day (15-minute intervals)
        current_hour = (start_hour + (t // 4)) % 24
        
        if anomalous and seq_len // 2 <= t < seq_len // 2 + 4:
            # Inject coordinated attack pattern in middle 1-hour window
            if t == seq_len // 2:  # Initial compromise
                timestep = make_diverse_timestep(archetype, current_hour, noise_level)
                timestep[FEATURES.index("okta_fail_rate")] = 0.8  # Authentication issues
                timestep[FEATURES.index("okta_login_cnt")] *= 3  # Multiple attempts
            elif t == seq_len // 2 + 1:  # Successful login
                timestep = make_diverse_timestep(archetype, current_hour, noise_level)  
                timestep[FEATURES.index("okta_fail_rate")] = 0.0  # Success
                timestep[FEATURES.index("delta_okta_to_edr_secs")] = 5  # Quick correlation
            elif t == seq_len // 2 + 2:  # Lateral movement
                timestep = make_diverse_timestep(archetype, current_hour, noise_level)
                timestep[FEATURES.index("edr_proc_cnt")] *= 4  # Process spike
                timestep[FEATURES.index("edr_admin_events")] = 5  # Admin activity
            else:  # Data exfiltration
                timestep = make_diverse_timestep(archetype, current_hour, noise_level)
                timestep[FEATURES.index("email_out_cnt")] *= 8  # Email spike
                timestep[FEATURES.index("email_ext_ratio")] = 0.9  # External
                timestep[FEATURES.index("edr_net_bytes")] *= 10  # Data transfer
        else:
            # Normal behavior with temporal and archetype patterns
            timestep = make_diverse_timestep(archetype, current_hour, noise_level)
            
        sequence.append(timestep)
        
    return np.stack(sequence, axis=0)


def generate_subtle_anomalous_sequence(
    archetype: str = "general",
    seq_len: int = 24,
    attack_type: str = "insider_threat",
    noise_level: float = 0.15
) -> np.ndarray:
    """
    Generate subtle anomalous sequences that require manifold learning to detect.
    
    These are designed to be geometrically distinct in latent space but not
    easily separable by reconstruction error alone - similar to the gravitational
    wave detection problem where manifold geometry provides key discriminative power.
    
    Parameters
    ----------
    archetype : str
        Base user archetype
    seq_len : int
        Sequence length
    attack_type : str  
        Type of subtle attack: 'insider_threat', 'account_takeover', 'data_harvesting'
    noise_level : float
        Random variation level
        
    Returns
    -------
    np.ndarray
        Subtly anomalous sequence that maintains realistic behavioral patterns
    """
    sequence = []
    start_hour = rng.integers(0, 24)
    
    for t in range(seq_len):
        current_hour = (start_hour + (t // 4)) % 24
        
        # Start with normal behavior
        timestep = make_diverse_timestep(archetype, current_hour, noise_level)
        
        if attack_type == "insider_threat":
            # Subtle pattern: Gradual increase in data access, staying within normal bounds
            if t > seq_len // 3:
                progress = (t - seq_len // 3) / (2 * seq_len // 3)
                # Gradually increase activity but stay within 2x normal (subtle)
                timestep[FEATURES.index("edr_filemods")] *= (1 + 0.8 * progress)
                timestep[FEATURES.index("email_out_cnt")] *= (1 + 0.6 * progress)
                # Very subtle timing correlations
                if progress > 0.7:
                    timestep[FEATURES.index("delta_okta_to_edr_secs")] *= 0.7  # Faster response
                    timestep[FEATURES.index("email_ext_ratio")] *= (1 + 0.3 * progress)
                    
        elif attack_type == "account_takeover":  
            # Subtle pattern: Slight behavioral shifts that indicate compromise
            if seq_len // 4 <= t <= 3 * seq_len // 4:
                # Slight increase in authentication activity
                timestep[FEATURES.index("okta_login_cnt")] *= rng.uniform(1.1, 1.4)  
                # Minor increase in fail rate (not obvious)
                timestep[FEATURES.index("okta_fail_rate")] *= rng.uniform(1.2, 1.8)
                # Subtle geographic inconsistencies
                if rng.random() < 0.3:  # 30% of time
                    timestep[FEATURES.index("okta_geo_switches")] += rng.poisson(0.5)
                # Cross-system correlation changes
                timestep[FEATURES.index("delta_okta_to_edr_secs")] *= rng.uniform(0.6, 1.4)
                
        elif attack_type == "data_harvesting":
            # Subtle pattern: Consistent small increases across multiple systems
            if t > seq_len // 4:
                multiplier = rng.uniform(1.15, 1.35)  # Small but consistent
                timestep[FEATURES.index("edr_net_bytes")] *= multiplier
                timestep[FEATURES.index("edr_proc_cnt")] *= rng.uniform(1.05, 1.25)
                # Slight email pattern changes
                if rng.random() < 0.4:
                    timestep[FEATURES.index("email_out_cnt")] *= rng.uniform(1.1, 1.3)
                    timestep[FEATURES.index("email_link_ratio")] *= rng.uniform(1.1, 1.4)
        
        sequence.append(timestep)
    
    return np.stack(sequence, axis=0)


def _generate_anomalous(kind: str) -> np.ndarray:
    """Helper: create an anomalous timestep by modifying a normal one."""
    x = _generate_normal()

    if kind == "spray_and_exfil":
        # Credential spray + exfil pattern
        x[FEATURES.index("okta_fail_rate")] = _clamp(rng.normal(0.4, 0.05), 0.2, 0.8)
        x[FEATURES.index("email_out_cnt")] = rng.poisson(25)
        x[FEATURES.index("edr_net_bytes")] = int(rng.normal(5e7, 1e7))
        x[FEATURES.index("delta_okta_to_edr_secs")] = max(0, rng.normal(5, 2))

    elif kind == "geo_impossible_travel":
        # Impossible travel + risky admin activity
        x[FEATURES.index("okta_geo_switches")] = rng.poisson(4)
        x[FEATURES.index("okta_mfa_rate")] = _clamp(rng.normal(0.2, 0.05), 0.05, 0.6)
        x[FEATURES.index("edr_admin_events")] = rng.poisson(3)
        x[FEATURES.index("delta_okta_to_edr_secs")] = max(0, rng.normal(1800, 600))

    return x.astype(np.float32)


# ----------------------------
# Sequence construction
# ----------------------------


def make_sequence(T: int = 24, anomalous: bool = False, kind: str | None = None) -> np.ndarray:
    """
    Generate a sequence of UEBA timesteps.

    Parameters
    ----------
    T : int, optional
        Number of time buckets in the sequence (default 24).
    anomalous : bool, optional
        If True, inject a short anomalous window centered in the sequence.
    kind : str or None, optional
        Anomaly type when `anomalous=True`. Supported: {'spray_and_exfil', 'geo_impossible_travel'}.

    Returns
    -------
    np.ndarray
        Array of shape (T, F) with unscaled raw features for each time step.
    """
    xs = []
    window = {T // 2, T // 2 + 1, T // 2 + 2}
    for t in range(T):
        if anomalous and t in window:
            xs.append(make_timestep(kind if kind else "normal"))
        else:
            xs.append(make_timestep("normal"))
    return np.stack(xs, axis=0)


# ----------------------------
# Preprocessing
# ----------------------------


def compute_stats(train_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean and std from training data.

    Parameters
    ----------
    train_array : np.ndarray
        Shape (N, F) where N is total timesteps across training sequences.

    Returns
    -------
    (mu, sigma) : Tuple[np.ndarray, np.ndarray]
        Per-feature mean and std; sigma has a small epsilon added for stability.
    """
    mu = train_array.mean(axis=0)
    sigma = train_array.std(axis=0) + 1e-6
    return mu, sigma


def standardize(seq_np: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Standardize a UEBA feature sequence using precomputed mean and standard deviation.

    Parameters
    ----------
    seq_np : np.ndarray
        2D array, shape (T, F), raw values.
    mu : np.ndarray
        1D array, shape (F,), feature means from training data.
    sigma : np.ndarray
        1D array, shape (F,), feature stds from training data (+epsilon).

    Returns
    -------
    np.ndarray
        Standardized array, shape (T, F), dtype float32.
    """
    out: np.ndarray = ((seq_np - mu) / sigma).astype(np.float32)
    return out


class SeqDataset(Dataset):
    """
    PyTorch Dataset for UEBA sequences (standardized on init).

    Parameters
    ----------
    seq_list : list of np.ndarray
        Each element shape (T, F), raw values.
    mu : np.ndarray
        Feature means (F,), from training data.
    sigma : np.ndarray
        Feature stds (F,), from training data.

    Returns (via indexing)
    ----------------------
    torch.Tensor
        Standardized tensor of shape (T, F), dtype float32.
    """

    def __init__(self, seq_list: list[np.ndarray], mu: np.ndarray, sigma: np.ndarray) -> None:
        self.data = [torch.tensor(standardize(s, mu, sigma), dtype=torch.float32) for s in seq_list]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# ----------------------------
# Anomaly sequence generators
# ----------------------------


def create_isolated_okta_anomaly(seq_len: int = 24) -> np.ndarray:
    """Create sequence with isolated Okta anomaly (legitimate user behavior).

    Simulates legitimate scenarios that cause Okta anomalies without affecting other systems:
    - User traveling (geographic location change, authentication issues)
    - Network connectivity problems causing login failures
    - User forgetting password and retrying multiple times

    Args:
        seq_len: Length of sequence in timesteps (default 24 = 6 hours of 15-min windows)

    Returns:
        np.ndarray: Shape (seq_len, 13) with isolated Okta anomaly in middle timestep

    Anomaly characteristics:
        - okta_fail_rate: 0.7 (70% vs normal ~5%)
        - okta_login_cnt: 15 (vs normal ~8)
        - okta_geo_switches: 2 (location change vs normal ~0.2)
        - EDR and Email systems remain normal
    """
    sequence = []

    for t in range(seq_len):
        if t == seq_len // 2:  # Anomaly in middle
            # High Okta failures, but normal EDR/Email
            timestep = make_timestep("normal")
            timestep[FEATURES.index("okta_fail_rate")] = 0.7  # High failure rate
            timestep[FEATURES.index("okta_login_cnt")] = 15  # More login attempts
            timestep[FEATURES.index("okta_geo_switches")] = 2  # Location change
            sequence.append(timestep)
        else:
            sequence.append(make_timestep("normal"))

    return np.stack(sequence, axis=0)


def create_isolated_edr_anomaly(seq_len: int = 24) -> np.ndarray:
    """Create sequence with isolated EDR anomaly (legitimate system activity).

    Simulates legitimate scenarios that cause EDR anomalies without affecting other systems:
    - Software updates or patches being installed
    - System maintenance or cleanup tasks
    - Large file operations (backup, sync, etc.)
    - Development work (compilation, testing)

    Args:
        seq_len: Length of sequence in timesteps (default 24 = 6 hours of 15-min windows)

    Returns:
        np.ndarray: Shape (seq_len, 13) with isolated EDR anomaly in middle timestep

    Anomaly characteristics:
        - edr_proc_cnt: 500 (vs normal ~250)
        - edr_net_bytes: 25MB (vs normal ~8MB)
        - edr_filemods: 80 (vs normal ~30)
        - Okta and Email systems remain normal
    """
    sequence = []

    for t in range(seq_len):
        if t == seq_len // 2:  # Anomaly in middle
            # High EDR activity, but normal Okta/Email
            timestep = make_timestep("normal")
            timestep[FEATURES.index("edr_proc_cnt")] = 500  # High process count
            timestep[FEATURES.index("edr_net_bytes")] = 25000000  # High network
            timestep[FEATURES.index("edr_filemods")] = 80  # Many file changes
            sequence.append(timestep)
        else:
            sequence.append(make_timestep("normal"))

    return np.stack(sequence, axis=0)


def create_isolated_email_anomaly(seq_len: int = 24) -> np.ndarray:
    """Create sequence with isolated Email anomaly (legitimate communication activity).

    Simulates legitimate scenarios that cause Email anomalies without affecting other systems:
    - Sending company newsletter or announcements
    - Bulk communication to external partners/customers
    - Marketing campaigns or event invitations
    - End-of-quarter reporting to external stakeholders

    Args:
        seq_len: Length of sequence in timesteps (default 24 = 6 hours of 15-min windows)

    Returns:
        np.ndarray: Shape (seq_len, 13) with isolated Email anomaly in middle timestep

    Anomaly characteristics:
        - email_out_cnt: 25 (vs normal ~4)
        - email_ext_ratio: 0.8 (80% external vs normal ~35%)
        - Okta and EDR systems remain normal
    """
    sequence = []

    for t in range(seq_len):
        if t == seq_len // 2:  # Anomaly in middle
            # High email activity, but normal Okta/EDR
            timestep = make_timestep("normal")
            timestep[FEATURES.index("email_out_cnt")] = 25  # High outbound emails
            timestep[FEATURES.index("email_ext_ratio")] = 0.8  # Mostly external
            sequence.append(timestep)
        else:
            sequence.append(make_timestep("normal"))

    return np.stack(sequence, axis=0)


def create_coordinated_attack(seq_len: int = 24) -> np.ndarray:
    """Create sequence with coordinated cross-system attack pattern.

    Simulates a realistic multi-stage cyber attack that spans multiple security systems
    over time, demonstrating the type of coordinated behavior that UEBA excels at detecting.

    Attack Timeline (4 consecutive timesteps in sequence middle):

    T1 - Initial Credential Spray:
        - okta_fail_rate: 0.6 (60% failures - attacker trying multiple passwords)
        - okta_login_cnt: 20 (high attempt volume)
        - Other systems normal (attack hasn't succeeded yet)

    T2 - Spray Continues + Reconnaissance:
        - okta_fail_rate: 0.8 (80% failures - attack intensifying)
        - okta_login_cnt: 25 (more attempts)
        - edr_proc_cnt: 400 (reconnaissance tools running)
        - edr_net_bytes: 20MB (network scanning, enumeration)

    T3 - Success + Privilege Escalation + Data Access:
        - okta_fail_rate: 0.1 (10% failures - SUCCESS! attacker got in)
        - edr_admin_events: 3 (privilege escalation attempts)
        - edr_filemods: 60 (accessing sensitive files)
        - email_out_cnt: 15 (preparing for exfiltration)
        - delta_okta_to_edr_secs: 5 (FAST correlation - automated attack)

    T4 - Data Exfiltration:
        - email_out_cnt: 20 (high volume data exfiltration)
        - email_ext_ratio: 0.9 (90% external - sending data outside org)
        - edr_net_bytes: 35MB (large data transfers)

    Args:
        seq_len: Length of sequence in timesteps (default 24 = 6 hours of 15-min windows)

    Returns:
        np.ndarray: Shape (seq_len, 13) with 4-timestep coordinated attack pattern

    Key Correlations Demonstrated:
        - Temporal: Attack escalates over consecutive timesteps
        - Cross-system: Okta success immediately followed by EDR/Email activity
        - Behavioral: Realistic attack progression (spray → recon → access → exfil)
        - Timing: Fast cross-system correlation (5 sec vs normal 45 sec)

    This pattern should score much higher in UEBA than isolated anomalies because
    it violates multiple behavioral norms simultaneously across systems and time.
    """
    sequence = []
    attack_window = {seq_len // 2 - 1, seq_len // 2, seq_len // 2 + 1, seq_len // 2 + 2}

    for t in range(seq_len):
        if t in attack_window:
            timestep = make_timestep("normal")

            if t == seq_len // 2 - 1:  # T1: Initial credential spray
                timestep[FEATURES.index("okta_fail_rate")] = 0.6
                timestep[FEATURES.index("okta_login_cnt")] = 20

            elif t == seq_len // 2:  # T2: Spray continues + reconnaissance
                timestep[FEATURES.index("okta_fail_rate")] = 0.8
                timestep[FEATURES.index("okta_login_cnt")] = 25
                timestep[FEATURES.index("edr_proc_cnt")] = 400
                timestep[FEATURES.index("edr_net_bytes")] = 20000000

            elif t == seq_len // 2 + 1:  # T3: Success + privilege escalation + data access
                timestep[FEATURES.index("okta_fail_rate")] = 0.1  # Success!
                timestep[FEATURES.index("edr_admin_events")] = 3
                timestep[FEATURES.index("edr_filemods")] = 60
                timestep[FEATURES.index("email_out_cnt")] = 15
                timestep[FEATURES.index("delta_okta_to_edr_secs")] = 5  # Fast correlation

            elif t == seq_len // 2 + 2:  # T4: Data exfiltration
                timestep[FEATURES.index("email_out_cnt")] = 20
                timestep[FEATURES.index("email_ext_ratio")] = 0.9
                timestep[FEATURES.index("edr_net_bytes")] = 35000000

            sequence.append(timestep)
        else:
            sequence.append(make_timestep("normal"))

    return np.stack(sequence, axis=0)
