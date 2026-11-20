"""
extract_data.py
---------------
Inspects the NASA Li-ion Battery Aging .mat files, extracts discharge-cycle
measurements, and saves them as battery_temperature_data.csv.

Run this script first, before main.py.

NASA dataset files required (place in the same directory):
    B0005.mat, B0006.mat, B0007.mat, B0018.mat

Download from:
    https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/
    pcoe/pcoe-data-set-repository/  (search "Li-ion Battery Aging Datasets")
"""

import sys
import os
import scipy.io
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
MAT_FILES  = {
    "B0005": os.path.join(DATA_DIR, "B0005.mat"),
    "B0006": os.path.join(DATA_DIR, "B0006.mat"),
    "B0007": os.path.join(DATA_DIR, "B0007.mat"),
    "B0018": os.path.join(DATA_DIR, "B0018.mat"),
}
OUTPUT_CSV = os.path.join(DATA_DIR, "battery_temperature_data.csv")

# ── Helpers ───────────────────────────────────────────────────────────────────

def inspect_mat(mat_path: str) -> None:
    """Print the top-level and nested structure of a .mat file."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {os.path.basename(mat_path)}")
    print('='*60)

    raw = scipy.io.loadmat(mat_path, simplify_cells=True)
    # Filter out scipy metadata keys
    keys = [k for k in raw.keys() if not k.startswith("__")]
    print(f"Top-level keys: {keys}")

    for key in keys:
        obj = raw[key]
        print(f"\n  raw['{key}'] type: {type(obj)}")

        # The NASA files typically use a struct array under a key like 'B0005'
        if isinstance(obj, dict):
            print(f"    dict keys: {list(obj.keys())[:20]}")

        elif isinstance(obj, np.ndarray):
            print(f"    shape: {obj.shape}, dtype: {obj.dtype}")
            if obj.dtype.names:
                print(f"    struct fields: {obj.dtype.names}")
            # Peek at first element if it's an object array
            if obj.size > 0 and obj.dtype == object:
                elem = obj.flat[0]
                print(f"    first element type: {type(elem)}")
                if isinstance(elem, np.ndarray) and elem.dtype.names:
                    print(f"    first element fields: {elem.dtype.names}")
                elif isinstance(elem, dict):
                    print(f"    first element dict keys: {list(elem.keys())[:20]}")

        # Try to show cycle/type structure one level deeper
        if isinstance(obj, np.ndarray) and obj.dtype.names and 'cycle' in obj.dtype.names:
            cycles = obj['cycle'].flat[0]
            print(f"\n  'cycle' field — shape: {cycles.shape}, dtype: {cycles.dtype}")
            if cycles.dtype.names:
                print(f"    cycle fields: {cycles.dtype.names}")
            if cycles.size > 0:
                c0 = cycles.flat[0]
                print(f"    cycle[0] type: {type(c0)}")
                if isinstance(c0, np.ndarray) and c0.dtype.names:
                    print(f"    cycle[0] fields: {c0.dtype.names}")
                    # Try data sub-field
                    if 'data' in c0.dtype.names:
                        d0 = c0['data'].flat[0]
                        print(f"    cycle[0].data type: {type(d0)}")
                        if isinstance(d0, np.ndarray) and d0.dtype.names:
                            print(f"    cycle[0].data fields: {d0.dtype.names}")
                        elif isinstance(d0, dict):
                            print(f"    cycle[0].data dict keys: {list(d0.keys())[:20]}")


def safe_flat(arr, default=None):
    """Return arr.flat[0] if the array is non-empty, else default."""
    if arr is None:
        return default
    a = np.asarray(arr)
    return a.flat[0] if a.size > 0 else default


def to_1d(arr):
    """Flatten a nested numpy array to a plain 1-D float array."""
    if arr is None:
        return np.array([])
    a = np.asarray(arr, dtype=float)
    return a.flatten()


def extract_battery(battery_id: str, mat_path: str) -> pd.DataFrame:
    """
    Load one .mat file and extract per-time-step discharge measurements.

    Returns a DataFrame with columns:
        battery_id, cycle_id, voltage_V, current_A,
        time_s, temp_ambient_C, temp_battery_C

    Columns that cannot be found in the file are filled with NaN and
    a warning is printed — no data is fabricated.
    """
    print(f"\nExtracting {battery_id} …")

    raw = scipy.io.loadmat(mat_path, simplify_cells=True)
    keys = [k for k in raw.keys() if not k.startswith("__")]

    # Find the top-level battery struct (usually the same name as the file)
    batt_key = None
    for k in keys:
        if k.upper().startswith("B"):
            batt_key = k
            break
    if batt_key is None:
        batt_key = keys[0]

    batt = raw[batt_key]

    # Navigate to the cycle array -------------------------------------------
    # Typical NASA structure: batt['cycle']  — an array of cycle structs
    if isinstance(batt, dict):
        cycle_arr = batt.get("cycle", None)
    elif isinstance(batt, np.ndarray) and batt.dtype.names and "cycle" in batt.dtype.names:
        cycle_arr = batt["cycle"].flat[0]
    else:
        print(f"  WARNING: Cannot locate 'cycle' field in {battery_id}. Skipping.")
        return pd.DataFrame()

    if cycle_arr is None or (hasattr(cycle_arr, "__len__") and len(cycle_arr) == 0):
        print(f"  WARNING: 'cycle' array is empty in {battery_id}. Skipping.")
        return pd.DataFrame()

    # Ensure it's iterable as a flat sequence of cycle structs
    cycle_arr = np.asarray(cycle_arr)
    if cycle_arr.ndim == 0:
        cycle_arr = cycle_arr.reshape(1)

    rows = []
    discharge_count = 0

    for idx, cyc in enumerate(cycle_arr.flat):
        # Each cycle has a 'type' field ('charge', 'discharge', 'impedance')
        if isinstance(cyc, dict):
            cycle_type = str(cyc.get("type", "")).strip().lower()
            data_obj   = cyc.get("data", None)
        elif isinstance(cyc, np.ndarray) and cyc.dtype.names:
            cycle_type = str(safe_flat(cyc["type"]) if "type" in cyc.dtype.names else "").strip().lower()
            data_obj   = safe_flat(cyc["data"]) if "data" in cyc.dtype.names else None
        else:
            continue

        if cycle_type != "discharge":
            continue

        discharge_count += 1

        # ── Extract individual measurement channels ────────────────────────
        # Field name mapping: try several plausible NASA field names
        def get_channel(d, *candidates):
            if d is None:
                return None
            if isinstance(d, dict):
                for c in candidates:
                    if c in d:
                        return d[c]
            elif isinstance(d, np.ndarray) and d.dtype.names:
                for c in candidates:
                    if c in d.dtype.names:
                        return d[c].flat[0] if d[c].size > 0 else None
            return None

        voltage      = get_channel(data_obj, "Voltage_measured", "Voltage", "voltage")
        current      = get_channel(data_obj, "Current_measured", "Current", "current")
        time_s       = get_channel(data_obj, "Time", "time", "time_s")
        temp_batt    = get_channel(data_obj, "Temperature_measured",
                                             "Temp_measured", "Temperature", "temperature")
        temp_ambient = get_channel(data_obj, "Ambient_measured",
                                             "Ambient_temperature", "Ambient", "ambient")

        # Convert to plain 1-D arrays
        voltage      = to_1d(voltage)
        current      = to_1d(current)
        time_s       = to_1d(time_s)
        temp_batt    = to_1d(temp_batt)
        temp_ambient = to_1d(temp_ambient)

        # Determine number of time steps from temperature (target)
        n = len(temp_batt)
        if n == 0:
            # Try voltage length instead
            n = len(voltage)
        if n == 0:
            continue  # No usable measurements in this cycle

        def pad_or_trim(arr, length):
            """Make arr exactly `length` elements, padding with NaN if shorter."""
            if len(arr) == 0:
                return np.full(length, np.nan)
            if len(arr) >= length:
                return arr[:length]
            return np.concatenate([arr, np.full(length - len(arr), np.nan)])

        voltage      = pad_or_trim(voltage,      n)
        current      = pad_or_trim(current,      n)
        time_s       = pad_or_trim(time_s,        n)
        temp_batt    = pad_or_trim(temp_batt,    n)
        temp_ambient = pad_or_trim(temp_ambient, n)

        for i in range(n):
            rows.append({
                "battery_id":    battery_id,
                "cycle_id":      idx,          # original cycle index (not discharge count)
                "voltage_V":     voltage[i],
                "current_A":     current[i],
                "time_s":        time_s[i],
                "temp_ambient_C": temp_ambient[i],
                "temp_battery_C": temp_batt[i],
            })

    print(f"  Found {discharge_count} discharge cycles, {len(rows)} time-step rows.")
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Check that at least one .mat file is present
    present = {bid: p for bid, p in MAT_FILES.items() if os.path.isfile(p)}
    missing = [bid for bid in MAT_FILES if bid not in present]

    if missing:
        print(f"\nWARNING: The following .mat files were not found in {DATA_DIR}:")
        for m in missing:
            print(f"    {m}.mat")
        print("Place the NASA dataset files in the project folder and re-run.")

    if not present:
        print("\nERROR: No .mat files found. Cannot extract data. Exiting.")
        sys.exit(1)

    # 2. Inspect structure of the first available file
    first_path = next(iter(present.values()))
    inspect_mat(first_path)

    # 3. Extract data from every available file
    frames = []
    for battery_id, mat_path in present.items():
        df = extract_battery(battery_id, mat_path)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("\nERROR: Extraction yielded no rows. Check the inspect output above.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # 4. Basic quality checks
    print(f"\n{'='*60}")
    print("Dataset summary")
    print('='*60)
    print(f"Total rows       : {len(combined):,}")
    print(f"Batteries        : {combined['battery_id'].unique().tolist()}")
    print(f"Discharge cycles : {combined.groupby(['battery_id','cycle_id']).ngroups}")
    print(f"\nColumn-level stats:")
    print(combined.describe().to_string())

    # Missing values
    print(f"\nMissing values per column:")
    print(combined.isnull().sum().to_string())

    # Duplicate rows
    n_dup = combined.duplicated().sum()
    print(f"\nDuplicate rows: {n_dup}")
    if n_dup > 0:
        combined = combined.drop_duplicates()
        print(f"  → Removed. Rows remaining: {len(combined):,}")

    # Ambient temperature variation
    if combined["temp_ambient_C"].notna().any():
        amb_std = combined["temp_ambient_C"].std()
        amb_mean = combined["temp_ambient_C"].mean()
        print(f"\nAmbient temperature — mean: {amb_mean:.2f} °C, std: {amb_std:.3f} °C")
        if amb_std < 0.5:
            print("  NOTE: Ambient temperature is nearly constant across the dataset.")
            print("  It carries little predictive information and will be flagged in main.py.")

    # Temperature distribution
    print(f"\nBattery temperature (target) range: "
          f"{combined['temp_battery_C'].min():.1f} – {combined['temp_battery_C'].max():.1f} °C")

    # 5. Save CSV
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(combined):,} rows × {len(combined.columns)} columns)")


if __name__ == "__main__":
    main()
