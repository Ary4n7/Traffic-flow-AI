"""
src/data_loader.py
------------------
STAGE 1 — Data loading and initial preprocessing.

Responsibilities
----------------
1. Load Indian Driving Dataset (IDD).h5 → pandas DataFrame of shape (34272, 207)
   - Rows   = timestamps (5-min intervals, March–June 2012)
   - Columns = sensor IDs (207 LA highway loop detectors)
   - Values  = speed in miles per hour (mph)

2. Load distances_delhi_ncr.csv → pairwise sensor distances (meters)
   - Columns: from, to, cost

3. Load graph_sensor_ids.txt → ordered list of 207 sensor IDs
   (defines the column order in Indian Driving Dataset (IDD).h5)

4. Validate all data and report a human-readable summary.

5. Save a cleaned Parquet snapshot for downstream stages.

Notes
-----
- Speed units are preserved as-is (mph).  No unit conversion is applied.
- Missing values (NaN / zeros) are flagged but NOT silently dropped here;
  imputation decisions are left to preprocessing.py.
- The from/to columns in distances_delhi_ncr.csv represent physical road
  proximity, NOT traffic flow direction.
"""

import os
import sys
import pickle
import logging

import numpy as np
import pandas as pd
import h5py

# Allow running as a script from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_loader")


# ── 1. Speed data ─────────────────────────────────────────────────────────────



def _load_h5_with_h5py(path: str) -> pd.DataFrame:
    """
    Read a Indian Driving Dataset (IDD) style HDF5 file using h5py directly.

    The Indian Driving Dataset (IDD) h5 file has the structure:
        /df/axis0  → column labels (sensor IDs as bytes)
        /df/axis1  → row index (timestamps as int64 ns)
        /df/block0_items  → column labels for the data block
        /df/block0_values → the actual (207, T) float array

    Returns pd.DataFrame (T × 207) or None on failure.
    """
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        log.info("HDF5 top-level keys: %s", keys)

        # ── Try pandas-style layout under 'df' ───────────────────────────
        if "df" in f:
            grp = f["df"]
            grp_keys = list(grp.keys())
            log.info("HDF5 '/df' keys: %s", grp_keys)

            # Timestamps
            if "axis1" in grp:
                ts_raw = grp["axis1"][:]
                # Pandas stores timestamps as int64 nanoseconds
                index = pd.to_datetime(ts_raw, unit="ns")
            else:
                index = None

            # Column labels (sensor IDs)
            if "axis0" in grp:
                cols_raw = grp["axis0"][:]
                cols = [
                    c.decode("utf-8") if isinstance(c, bytes) else str(c)
                    for c in cols_raw
                ]
            elif "block0_items" in grp:
                cols_raw = grp["block0_items"][:]
                cols = [
                    c.decode("utf-8") if isinstance(c, bytes) else str(c)
                    for c in cols_raw
                ]
            else:
                cols = None

            # Data values
            if "block0_values" in grp:
                values = grp["block0_values"][:]  # shape: (n_cols, T) or (T, n_cols)
                # Transpose if needed so shape is (T, n_cols)
                if values.ndim == 2 and cols and values.shape[0] == len(cols):
                    values = values.T
            else:
                values = None

            if values is not None and index is not None and cols is not None:
                df = pd.DataFrame(values, index=index, columns=cols)
                log.info(
                    "Loaded via h5py pandas layout: shape=%s", df.shape
                )
                return df

        # ── Fallback: try pandas.read_hdf (requires tables) ─────────────
        try:
            import tables  # noqa: F401
            df = pd.read_hdf(path)
            log.info("Loaded via pandas.read_hdf: shape=%s", df.shape)
            return df
        except ImportError:
            log.debug("tables not available, skipping pandas.read_hdf fallback")
        except Exception as exc:
            log.warning("pandas.read_hdf failed: %s", exc)

        return None


def load_speed_data(path: str = config.METR_LA_H5) -> pd.DataFrame:

    """
    Load the Indian Driving Dataset (IDD) HDF5 speed file.

    Returns
    -------
    pd.DataFrame, shape (T, 207)
        Index   : pd.DatetimeIndex (5-min intervals)
        Columns : sensor IDs (integers as strings)
        Values  : speed in mph (float32)

    Raises
    ------
    FileNotFoundError if Indian Driving Dataset (IDD).h5 is absent.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\n[DATA MISSING] {path} does not exist.\n"
            "Please download Indian Driving Dataset (IDD).h5 from one of:\n"
            "  • https://github.com/leilin-research/GCGRNN (raw)\n"
            "  • https://github.com/liyaguang/DCRNN (Google Drive link in README)\n"
            "and place it in:  data/raw/Indian Driving Dataset (IDD).h5\n"
        )

    log.info("Loading %s …", path)
    df = _load_h5_with_h5py(path)
    if df is None:
        raise RuntimeError(
            f"Failed to load {path} — the file may be corrupt or in an "
            "unsupported format."
        )

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Cast to float32 to save memory
    df = df.astype("float32")

    # Ensure column names are strings (some files store as int)
    df.columns = df.columns.astype(str)

    log.info(
        "Speed data loaded: %d timesteps × %d sensors  |  "
        "%s → %s",
        len(df), len(df.columns),
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
    )
    return df


# ── 2. Sensor ID list ─────────────────────────────────────────────────────────

def load_sensor_ids(path: str = config.SENSOR_IDS_TXT) -> list:
    """
    Load the ordered list of 207 sensor IDs from graph_sensor_ids.txt.

    Returns
    -------
    list of str  — e.g. ['773869', '767541', ...]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[DATA MISSING] {path} not found.\n"
            "Download from: https://github.com/liyaguang/DCRNN/tree/master/"
            "data/sensor_graph/graph_sensor_ids.txt"
        )
    with open(path, "r") as f:
        raw = f.read().strip()
    ids = [s.strip() for s in raw.split(",") if s.strip()]
    log.info("Loaded %d sensor IDs from %s", len(ids), path)
    return ids


# ── 3. Distance graph ─────────────────────────────────────────────────────────

def load_distances(path: str = config.DISTANCES_CSV) -> pd.DataFrame:
    """
    Load the pairwise road-network distances file.

    Returns
    -------
    pd.DataFrame with columns: from, to, cost
        'from' and 'to' are sensor IDs (str)
        'cost' is road-network distance in meters (float)

    Notes
    -----
    - The 'from'/'to' columns represent physical road proximity.
      They do NOT encode traffic flow direction.
    - Self-distances (from == to, cost == 0) are retained here and
      filtered in graph_builder.py.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[DATA MISSING] {path} not found.\n"
            "Download from: https://github.com/liyaguang/DCRNN/tree/master/"
            "data/sensor_graph/distances_delhi_ncr.csv"
        )
    dist_df = pd.read_csv(
        path,
        dtype={"from": str, "to": str, "cost": float},
    )
    dist_df.columns = dist_df.columns.str.strip()
    log.info(
        "Distance file loaded: %d rows  |  unique sensors: %d",
        len(dist_df),
        dist_df["from"].nunique(),
    )
    return dist_df


# ── 4. Validation and summary ─────────────────────────────────────────────────

def validate_and_summarise(
    speed_df: pd.DataFrame,
    sensor_ids: list,
    dist_df: pd.DataFrame,
) -> dict:
    """
    Cross-check all three data sources and return a summary dict.
    Also logs a human-readable report.
    """
    summary = {}

    # ── Speed DataFrame ────────────────────────────────────────────────────
    summary["n_timesteps"]    = len(speed_df)
    summary["n_sensors"]      = len(speed_df.columns)
    summary["time_start"]     = str(speed_df.index[0])
    summary["time_end"]       = str(speed_df.index[-1])
    summary["speed_unit"]     = config.SPEED_UNIT

    # Infer sample interval from first two timestamps
    if len(speed_df) > 1:
        delta = (speed_df.index[1] - speed_df.index[0]).total_seconds() / 60
        summary["sample_interval_min"] = delta
    else:
        summary["sample_interval_min"] = None

    # Speed statistics (global, across all sensors & timesteps)
    flat = speed_df.values.flatten()
    valid = flat[~np.isnan(flat)]
    summary["speed_min_mph"]  = float(np.min(valid))
    summary["speed_max_mph"]  = float(np.max(valid))
    summary["speed_mean_mph"] = float(np.mean(valid))

    # Missing / zero values
    nan_count  = int(np.isnan(flat).sum())
    zero_count = int((flat == 0).sum())
    total_vals = len(flat)
    summary["nan_count"]  = nan_count
    summary["zero_count"] = zero_count
    summary["nan_pct"]    = round(100 * nan_count  / total_vals, 3)
    summary["zero_pct"]   = round(100 * zero_count / total_vals, 3)

    # Congestion rate using default threshold
    thr = config.CONGESTION_THRESHOLD_MPH
    congested = int((valid < thr).sum())
    summary["congestion_rate_pct"] = round(100 * congested / len(valid), 2)
    summary["congestion_threshold_mph"] = thr

    # ── Sensor ID cross-check ──────────────────────────────────────────────
    speed_col_ids   = set(speed_df.columns)
    sensor_id_set   = set(sensor_ids)
    summary["sensor_ids_match_columns"] = speed_col_ids == sensor_id_set
    if speed_col_ids != sensor_id_set:
        diff = speed_col_ids.symmetric_difference(sensor_id_set)
        log.warning(
            "Sensor ID mismatch: %d IDs differ between .h5 columns "
            "and graph_sensor_ids.txt: %s",
            len(diff), list(diff)[:10],
        )
    else:
        log.info("✓ Sensor IDs in .h5 columns match graph_sensor_ids.txt")

    # ── Distance file cross-check ──────────────────────────────────────────
    dist_sensor_ids = set(dist_df["from"].unique()) | set(dist_df["to"].unique())
    in_speed_not_dist = speed_col_ids - dist_sensor_ids
    in_dist_not_speed = dist_sensor_ids - speed_col_ids
    summary["dist_sensors_in_speed"]   = len(dist_sensor_ids & speed_col_ids)
    summary["speed_sensors_not_in_dist"] = len(in_speed_not_dist)
    summary["dist_sensors_not_in_speed"] = len(in_dist_not_speed)

    if in_speed_not_dist:
        log.warning(
            "%d speed sensors have no entry in distances file: %s …",
            len(in_speed_not_dist), list(in_speed_not_dist)[:5],
        )

    # ── Pretty log ────────────────────────────────────────────────────────
    log.info(
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Indian Driving Dataset (IDD) Dataset Summary\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Timesteps     : %d\n"
        "  Sensors       : %d\n"
        "  Time range    : %s  →  %s\n"
        "  Sample rate   : %.0f min\n"
        "  Speed unit    : %s\n"
        "  Speed range   : %.1f – %.1f %s (mean %.1f)\n"
        "  NaN values    : %d  (%.3f%%)\n"
        "  Zero values   : %d  (%.3f%%)\n"
        "  Congested (<%.0f mph): %.2f%% of readings\n"
        "  Sensor ID check: %s\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        summary["n_timesteps"],
        summary["n_sensors"],
        summary["time_start"],
        summary["time_end"],
        summary["sample_interval_min"],
        summary["speed_unit"],
        summary["speed_min_mph"],
        summary["speed_max_mph"],
        summary["speed_unit"],
        summary["speed_mean_mph"],
        summary["nan_count"],
        summary["nan_pct"],
        summary["zero_count"],
        summary["zero_pct"],
        summary["congestion_threshold_mph"],
        summary["congestion_rate_pct"],
        "✓ match" if summary["sensor_ids_match_columns"] else "✗ MISMATCH",
    )
    return summary


# ── 5. Save processed snapshot ────────────────────────────────────────────────

def save_processed_speeds(speed_df: pd.DataFrame) -> None:
    """
    Save a clean Parquet snapshot for downstream stages.

    Parquet is used (not CSV) because:
    - Preserves the DatetimeIndex dtype
    - ~3× smaller than CSV for float data
    - Reads/writes significantly faster
    """
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    out = config.PROCESSED_SPEEDS
    speed_df.to_parquet(out, index=True)
    size_mb = os.path.getsize(out) / 1e6
    log.info("Saved processed speeds → %s  (%.1f MB)", out, size_mb)


# ── 6. Public load_all convenience function ───────────────────────────────────

def load_all(save_snapshot: bool = True) -> dict:
    """
    Load all three Indian Driving Dataset (IDD) data sources, validate, and optionally
    save a processed Parquet snapshot.

    Returns
    -------
    dict with keys:
        'speed_df'   : pd.DataFrame (T × 207)
        'sensor_ids' : list of str
        'dist_df'    : pd.DataFrame (from, to, cost)
        'summary'    : dict of validation statistics
    """
    speed_df   = load_speed_data()
    sensor_ids = load_sensor_ids()
    dist_df    = load_distances()
    summary    = validate_and_summarise(speed_df, sensor_ids, dist_df)

    if save_snapshot:
        save_processed_speeds(speed_df)

    return {
        "speed_df":   speed_df,
        "sensor_ids": sensor_ids,
        "dist_df":    dist_df,
        "summary":    summary,
    }


# ── 7. Quick inspection helper ────────────────────────────────────────────────

def inspect_speed_sample(speed_df: pd.DataFrame, n_sensors: int = 5) -> None:
    """Print a head/tail sample of the speed DataFrame for quick inspection."""
    sample_cols = list(speed_df.columns[:n_sensors])
    print("\n── Speed DataFrame Head (first 5 rows, first 5 sensors) ──")
    print(speed_df[sample_cols].head().to_string())
    print("\n── Speed DataFrame Tail (last 5 rows, first 5 sensors) ──")
    print(speed_df[sample_cols].tail().to_string())
    print(f"\n── dtypes ──\n{speed_df.dtypes.value_counts()}")
    print(f"\n── Index type: {type(speed_df.index).__name__} ──")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TrafficFlow AI — STAGE 1: Data Loader")
    print("=" * 60 + "\n")

    result = load_all(save_snapshot=True)
    inspect_speed_sample(result["speed_df"])

    print("\n── Sample distance rows ──")
    print(result["dist_df"].head(10).to_string(index=False))

    print("\n── First 10 sensor IDs ──")
    print(result["sensor_ids"][:10])

    print("\n── Summary ──")
    for k, v in result["summary"].items():
        print(f"  {k:<40} {v}")

    print("\n✓ STAGE 1 complete.\n")
