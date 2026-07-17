from __future__ import annotations
import re
import os
import json
import tempfile
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import gzip  # Add this import at the top of your file

try:
    import eccodes
except ImportError:
    eccodes = None

NOMADS_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_blend.pl"
REGION = "co"

SUBREGION_PARAMS = {
    "subregion": "on",
    "leftlon": "278",
    "rightlon": "280",
    "toplat": "29.8",
    "bottomlat": "26.5"
}

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nbm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_get(kind: str, key: str) -> bytes | None:
    path = os.path.join(CACHE_DIR, f"{kind}__{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

def cache_put(kind: str, key: str, data: bytes) -> None:
    path = os.path.join(CACHE_DIR, f"{kind}__{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}")
    with open(path, "wb") as f:
        f.write(data)

def purge_old_grib_cache(current_ymd: str, current_cc: str) -> None:
    """Deletes all cached GRIB files that do not match the current cycle."""
    log("=== Phase 0: Cleaning up old cached GRIB payloads ===")
    prefix = f"grib__{current_ymd}_{current_cc}z_"
    purged_count = 0
    
    for filename in os.listdir(CACHE_DIR):
        if filename.startswith("grib__") and not filename.startswith(prefix):
            try:
                os.remove(os.path.join(CACHE_DIR, filename))
                purged_count += 1
            except Exception as e:
                log(f" Failed to delete cached file {filename}: {e}")
                
    if purged_count > 0:
        log(f" Successfully purged {purged_count} stale GRIB files from previous cycles.")
    else:
        log(" Cache directory is already clean for this cycle.")

CITIES = [
    { "name": "Orlando Intl Airport* (Orange)", "lat": 28.429444, "lon": -81.308889, "sid": "MCOthr 9" },
    { "name": "Downtown Orlando (Orange)", "lat": 28.538, "lon": -81.379 },
    { "name": "Apopka (Orange)", "lat": 28.701, "lon": -81.531 },
    { "name": "Bithlo (Orange)", "lat": 28.552, "lon": -81.105 },
    { "name": "Ocoee (Orange)", "lat": 28.574, "lon": -81.53 },
    { "name": "Winter Park (Orange)", "lat": 28.596, "lon": -81.346 },
    { "name": "Lake Buena Vista (Orange)", "lat": 28.366, "lon": -81.526 },
    { "name": "UCF (Orange)", "lat": 28.589, "lon": -81.197 },
    { "name": "Zellwood (Orange)", "lat": 28.728, "lon": -81.6 },
    { "name": "Lake Nona (Orange)", "lat": 28.391, "lon": -81.269 },
    { "name": "Daytona Beach* (Volusia)", "lat": 29.178321, "lon": -81.060861, "sid": "DABthr 9" },
    { "name": "DeLand (Volusia)", "lat": 29.028, "lon": -81.303 },
    { "name": "Deltona (Volusia)", "lat": 28.892, "lon": -81.259 },
    { "name": "Ormond Beach (Volusia)", "lat": 29.284, "lon": -81.056 },
    { "name": "New Smyrna Beach (Volusia)", "lat": 29.025, "lon": -80.927 },
    { "name": "Port Orange (Volusia)", "lat": 29.117, "lon": -80.999 },
    { "name": "Pierson (Volusia)", "lat": 29.191, "lon": -81.419 },
    { "name": "Sanford* (Seminole)", "lat": 28.7760, "lon": -81.2345, "sid": "SFBthr 9" },
    { "name": "Oviedo (Seminole)", "lat": 28.665, "lon": -81.189 },
    { "name": "Altamonte Springs (Seminole)", "lat": 28.66, "lon": -81.393 },
    { "name": "Longwood (Seminole)", "lat": 28.697, "lon": -81.338 },
    { "name": "Winter Springs (Seminole)", "lat": 28.695, "lon": -81.306 },
    { "name": "Lake Mary (Seminole)", "lat": 28.752, "lon": -81.32 },
    { "name": "Geneva (Seminole)", "lat": 28.733, "lon": -81.113 },
    { "name": "Leesburg* (Lake)", "lat": 28.8265, "lon": -81.8084, "sid": "LEEthr 9" },
    { "name": "Clermont (Lake)", "lat": 28.56, "lon": -81.776 },
    { "name": "Eustis (Lake)", "lat": 28.855, "lon": -81.688 },
    { "name": "Mount Dora (Lake)", "lat": 28.805, "lon": -81.643 },
    { "name": "Tavares (Lake)", "lat": 28.81, "lon": -81.728 },
    { "name": "Astor (Lake)", "lat": 29.167, "lon": -81.53 },
    { "name": "Altoona (Lake)", "lat": 28.968, "lon": -81.648 },
    { "name": "Lady Lake (Lake)", "lat": 28.939, "lon": -81.942 },
    { "name": "Melbourne* (Brevard)", "lat": 28.10275, "lon": -80.64525, "sid": "MLBthr 9" },
    { "name": "Titusville (Brevard)", "lat": 28.609, "lon": -80.813 },
    { "name": "Cocoa (Brevard)", "lat": 28.369, "lon": -80.743 },
    { "name": "Palm Bay (Brevard)", "lat": 28, "lon": -80.67 },
    { "name": "Satellite Beach (Brevard)", "lat": 28.173, "lon": -80.596 },
    { "name": "Mims (Brevard)", "lat": 28.668, "lon": -80.847 },
    { "name": "Barefoot Bay (Brevard)", "lat": 27.886, "lon": -80.53 },
    { "name": "Viera - Suntree (Brevard)", "lat": 28.256, "lon": -80.733 },
    { "name": "Cocoa Beach (Brevard)", "lat": 28.3126, "lon": -80.6136 },
    { "name": "Kissimmee (Osceola)", "lat": 28.292, "lon": -81.412 },
    { "name": "Celebration (Osceola)", "lat": 28.324, "lon": -81.543 },
    { "name": "St Cloud (Osceola)", "lat": 28.241, "lon": -81.282 },
    { "name": "Poinciana (Osceola)", "lat": 28.153, "lon": -81.467 },
    { "name": "Holopaw (Osceola)", "lat": 28.132, "lon": -81.076 },
    { "name": "Kenansville (Osceola)", "lat": 27.875, "lon": -80.989 },
    { "name": "Vero Beach* (Indian River)", "lat": 27.655556, "lon": -80.417944, "sid": "VRBthr 9" },
    { "name": "Sebastian (Indian River)", "lat": 27.815, "lon": -80.474 },
    { "name": "Fellsmere (Indian River)", "lat": 27.763, "lon": -80.602 },
    { "name": "South Beach (Indian River)", "lat": 27.591, "lon": -80.332 },
    { "name": "Wabasso (Indian River)", "lat": 27.751, "lon": -80.438 },
    { "name": "Blue Cypress Lake (Indian River)", "lat": 27.725, "lon": -80.776 },
    { "name": "Fort Pierce* (St. Lucie)", "lat": 27.443, "lon": -80.336, "sid": "FPRthr 9" },
    { "name": "Port St Lucie (St. Lucie)", "lat": 27.27, "lon": -80.385 },
    { "name": "Tradition (St. Lucie)", "lat": 27.265, "lon": -80.439 },
    { "name": "White City (St. Lucie)", "lat": 27.37, "lon": -80.335 },
    { "name": "St Lucie Fairgrounds (St. Lucie)", "lat": 27.368, "lon": -80.486 },
    { "name": "Lakewood Park (St. Lucie)", "lat": 27.546, "lon": -80.4 },
    { "name": "Okeechobee (Okeechobee)", "lat": 27.2476, "lon": -80.835 },
    { "name": "Okee-Tantee (Okeechobee)", "lat": 27.158, "lon": -80.865 },
    { "name": "Basinger (Okeechobee)", "lat": 27.398, "lon": -81.016 },
    { "name": "Fort Drum (Okeechobee)", "lat": 27.522, "lon": -80.807 },
    { "name": "Taylor Creek (Okeechobee)", "lat": 27.212, "lon": -80.792 },
    { "name": "Stuart (Martin)", "lat": 27.184, "lon": -80.222 },
    { "name": "Indiantown (Martin)", "lat": 27.028, "lon": -80.478 },
    { "name": "Palm City (Martin)", "lat": 27.166, "lon": -80.268 },
    { "name": "Hobe Sound (Martin)", "lat": 27.06, "lon": -80.136 },
    { "name": "Port Salerno (Martin)", "lat": 27.142, "lon": -80.201 },
    { "name": "Tequesta (Martin)", "lat": 26.98, "lon": -80.12 },
    { "name": "Jensen Beach (Martin)", "lat": 27.256, "lon": -80.232 }
]

def http_get(url: str, timeout: int = 45) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()

def build_nomads_url(run_dt: datetime, f_hour: int) -> str:
    ymd, cc = run_dt.strftime("%Y%m%d"), f"{run_dt.hour:02d}"
    params = {
        "file": f"blend.t{cc}z.qmd.f{f_hour:03d}.{REGION}.grib2",
        "dir": f"/blend.{ymd}/{cc}/qmd",
        "var_APCP": "on",
        "var_APTMP": "on",
        "var_TMP": "on",
        "lev_2_m_above_ground": "on",
        "lev_surface": "on",
        **SUBREGION_PARAMS
    }
    return f"{NOMADS_URL}?{urllib.parse.urlencode(params)}"

def check_run_exists(run_dt: datetime) -> bool:
    # A run is only usable once BOTH ends of the lead-time range are posted: f024 (short lead,
    # posts quickly) and f192 (the longest lead time this pipeline needs, and the slowest to
    # post). Checking f024 alone previously let the pipeline lock onto a run that looked live
    # but hadn't finished posting its longer lead times yet, silently truncating day 8.
    for probe_fhour in (24, 192):
        url = build_nomads_url(run_dt, probe_fhour)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read(100)
                if not (b"GRIB" in body or resp.status in (200, 206)):
                    log(f" Run {run_dt.strftime('%Y-%m-%d')} {run_dt.hour:02d}Z rejected: f{probe_fhour:03d} not available yet")
                    return False
        except Exception as e:
            log(f" Run {run_dt.strftime('%Y-%m-%d')} {run_dt.hour:02d}Z rejected: f{probe_fhour:03d} probe failed ({e})")
            return False
    return True

def latest_complete_qmd_run(max_attempts: int = 6) -> datetime:
    now = datetime.now(timezone.utc)
    cycle = (now.hour // 6) * 6
    run = now.replace(hour=cycle, minute=0, second=0, microsecond=0)
    
    for _ in range(max_attempts):
        log(f"Probing NOMADS run cycle {run.strftime('%Y-%m-%d')} {run.hour:02d}Z...")
        if check_run_exists(run):
            log(f" Successfully locked onto live NOMADS run: {run.strftime('%Y-%m-%d')} {run.hour:02d}Z")
            return run
        run -= timedelta(hours=6)
        
    fallback_run = now.replace(hour=cycle, minute=0, second=0, microsecond=0) - timedelta(hours=6)
    return fallback_run

def fetch_filtered_grib(run_dt: datetime, f_hour: int):
    cc = f"{run_dt.hour:02d}"
    cache_key = f"{run_dt.strftime('%Y%m%d')}_{cc}z_f{f_hour:03d}_filtered"

    cached = cache_get("grib", cache_key)
    if cached is not None:
        return cached, "cached"

    url = build_nomads_url(run_dt, f_hour)
    try:
        body = http_get(url)
        if body.startswith(b"GRIB"):
            cache_put("grib", cache_key, body)
            return body, "downloaded"
        return b"", "invalid"
    except Exception as e:
        return b"", f"failed ({e})"

def anchor_forecast_hours(run_hour: int, end_hour_utc: int, window_hours: int, n_periods: int = 8) -> list[int]:
    base = (end_hour_utc - run_hour) % 24
    if base == 0:
        base = 24
    hours = [base + 24 * k for k in range(n_periods)]
    return [f for f in hours if (f - window_hours) >= 0]

def maxt_valid_date(run_dt: datetime, f_hour: int):
    return ((run_dt + timedelta(hours=f_hour)) - timedelta(hours=12)).date()

def mint_valid_date(run_dt: datetime, f_hour: int):
    return (run_dt + timedelta(hours=f_hour)).date()

_grid_box_cache: dict | None = None  # {"fingerprint": (...), "indices": {city_name: {"nearest": int, "box": [int,...]}}}

_GRID_FINGERPRINT_KEYS = (
    "Nx", "Ny", "jPointsAreConsecutive",
    "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
    "gridType",
)

# Cities to log full box contents for on every run, as a manual spot-check that the
# 3x3 neighborhood is actually a compact square of physically adjacent grid cells
# around the city -- one interior point, one nearer the domain edge.
_DEBUG_SPOTCHECK_CITIES = {"Downtown Orlando (Orange)", "Melbourne* (Brevard)"}

def _grid_fingerprint(gid: int) -> tuple:
    """A cheap signature of the grid's geometry. If this ever differs between
    messages/files within a run, cached flat-array indices from an earlier grid
    are no longer valid and must not be reused."""
    vals = []
    for key in _GRID_FINGERPRINT_KEYS:
        try:
            vals.append(eccodes.codes_get(gid, key))
        except Exception:
            vals.append(None)
    return tuple(vals)

def _flat_to_rowcol(idx: int, nx: int, ny: int, j_consecutive: bool) -> tuple[int, int]:
    if j_consecutive:
        return idx % ny, idx // ny  # (row, col)
    return idx // nx, idx % nx      # (row, col)

def _rowcol_to_flat(row: int, col: int, nx: int, ny: int, j_consecutive: bool) -> int:
    if j_consecutive:
        return col * ny + row
    return row * nx + col

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2
    r = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlmb / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))

def get_city_box_indices(gid: int, cities: list[dict]) -> dict[str, dict]:
    """For each city, returns the flat index of the nearest grid point plus the
    flat indices of the surrounding 3x3 box (fewer than 9 if the nearest point is
    right at the edge of the cropped subregion domain). Cached for the run, but
    keyed on a grid-geometry fingerprint: if a later message/file turns out to be
    on a differently-shaped or differently-anchored grid, the cache is invalidated
    and recomputed rather than silently reused."""
    global _grid_box_cache

    fingerprint = _grid_fingerprint(gid)

    if _grid_box_cache is not None and _grid_box_cache["fingerprint"] == fingerprint:
        return _grid_box_cache["indices"]

    if _grid_box_cache is not None:
        log(f" WARNING: grid geometry changed mid-run (was {_grid_box_cache['fingerprint']}, "
            f"now {fingerprint}) -- recomputing city grid box indices.")

    nx = eccodes.codes_get(gid, "Nx")
    ny = eccodes.codes_get(gid, "Ny")
    try:
        j_consecutive = bool(eccodes.codes_get(gid, "jPointsAreConsecutive"))
    except Exception:
        j_consecutive = False

    lats = [c["lat"] for c in cities]
    lons = [c["lon"] for c in cities]
    nearest_points = eccodes.codes_grib_find_nearest_multiple(gid, False, lats, lons)

    grid_lats = eccodes.codes_get_array(gid, "latitudes")
    grid_lons = eccodes.codes_get_array(gid, "longitudes")

    # Don't just trust the semantic reading of jPointsAreConsecutive -- derive which
    # axis actually varies fastest directly from the grid's own lat/lon arrays.
    # Moving from flat index 0 to index 1 either shifts latitude or shifts longitude;
    # that's an observable fact about this specific file, not an assumption about
    # what the WMO flag is supposed to mean. This is a hard, deterministic check
    # (unlike the statistical step-consistency check below, which has a known blind
    # spot on near-square grids) and takes precedence if the two disagree.
    if len(grid_lats) > 1:
        d_lat_01 = abs(grid_lats[1] - grid_lats[0])
        d_lon_01 = abs(grid_lons[1] - grid_lons[0])
        empirical_j_consecutive = d_lat_01 > d_lon_01
        if empirical_j_consecutive != j_consecutive:
            log(f" WARNING: eccodes reports jPointsAreConsecutive={j_consecutive}, but the "
                f"grid's own lat/lon arrays disagree (index 0->1 moves lat by {d_lat_01:.5f} deg, "
                f"lon by {d_lon_01:.5f} deg). Trusting the empirical evidence over the flag.")
            j_consecutive = empirical_j_consecutive

    # Secondary self-check below: on a projected grid, the immediate N/S/E/W neighbor
    # of any point should be roughly one grid-step away, and that step size should be
    # roughly the same for every city in a given direction. Kept as a second line of
    # defense in case Nx/Ny themselves are wrong (which the empirical check above
    # doesn't cover), even though it's weaker on near-square grids.

    indices: dict[str, dict] = {}
    # Track step distances separately per direction (x=E/W, y=N/S): a real projected
    # grid can legitimately have a different step size in x vs y, so those two must
    # not be pooled together, or a systemic wrong-direction bug can hide inside the
    # blended spread. Within a single direction, though, step size should be tight
    # and consistent across the whole domain.
    x_steps: list[float] = []
    y_steps: list[float] = []
    per_city_dir_steps: dict[str, dict[str, list[float]]] = {}

    for c, pt in zip(cities, nearest_points):
        row, col = _flat_to_rowcol(pt.index, nx, ny, j_consecutive)
        box = []
        city_dir_steps = {"x": [], "y": []}
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, cc = row + dr, col + dc
                if 0 <= r < ny and 0 <= cc < nx:
                    member_idx = _rowcol_to_flat(r, cc, nx, ny, j_consecutive)
                    box.append(member_idx)
                    if abs(dr) + abs(dc) == 1:  # orthogonal (N/S/E/W) neighbor, not diagonal
                        dist_km = _haversine_km(
                            grid_lats[pt.index], grid_lons[pt.index],
                            grid_lats[member_idx], grid_lons[member_idx],
                        )
                        if dc != 0:
                            city_dir_steps["x"].append(dist_km)
                            x_steps.append(dist_km)
                        else:
                            city_dir_steps["y"].append(dist_km)
                            y_steps.append(dist_km)
        indices[c["name"]] = {"nearest": pt.index, "box": box}
        per_city_dir_steps[c["name"]] = city_dir_steps

    geometry_ok = True
    for direction, pooled in (("x/E-W", x_steps), ("y/N-S", y_steps)):
        if not pooled:
            continue
        median_step = sorted(pooled)[len(pooled) // 2]
        bad_cities = []
        for name, dir_steps in per_city_dir_steps.items():
            key = "x" if direction.startswith("x") else "y"
            for d in dir_steps[key]:
                if median_step > 0 and abs(d - median_step) / median_step > 0.2:
                    bad_cities.append(name)
                    break
        if len(bad_cities) > max(1, len(cities) // 10):
            geometry_ok = False
            log(f" WARNING: box index geometry self-check FAILED in the {direction} direction. "
                f"Median single-step distance is {median_step:.2f} km, but "
                f"{len(bad_cities)}/{len(cities)} cities have an inconsistent step there "
                f"(e.g. {bad_cities[:5]}). The row/col scan-order assumption "
                f"(jPointsAreConsecutive={j_consecutive}, Nx={nx}, Ny={ny}) likely doesn't "
                f"match this grid's actual layout -- box results are NOT trustworthy "
                f"until this is fixed.")

    if geometry_ok:
        log(" Box index geometry self-check passed: step sizes are consistent across the domain.")

    for c in cities:
        if c["name"] in _DEBUG_SPOTCHECK_CITIES:
            info = indices[c["name"]]
            nearest_lat, nearest_lon = grid_lats[info["nearest"]], grid_lons[info["nearest"]]
            nearest_dist = _haversine_km(c["lat"], c["lon"], nearest_lat, nearest_lon)
            log(f" [spot-check] {c['name']}: target=({c['lat']:.4f},{c['lon']:.4f}) "
                f"nearest_idx={info['nearest']} at ({nearest_lat:.4f},{nearest_lon:.4f}), "
                f"{nearest_dist:.2f} km from target -- box has {len(info['box'])} members")
            for member_idx in info["box"]:
                m_lat, m_lon = grid_lats[member_idx], grid_lons[member_idx]
                d_from_nearest = _haversine_km(nearest_lat, nearest_lon, m_lat, m_lon)
                marker = " (=nearest)" if member_idx == info["nearest"] else ""
                log(f"     idx={member_idx} lat={m_lat:.4f} lon={m_lon:.4f} "
                    f"dist_from_nearest={d_from_nearest:.2f} km{marker}")

    _grid_box_cache = {"fingerprint": fingerprint, "indices": indices}
    return indices

def decode_nomads_file(raw_bytes: bytes, cities: list[dict], f_hour: int | None = None) -> dict:
    if not raw_bytes or eccodes is None:
        return {}

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    extracted_data = {
        "TMP_max": {}, "TMP_min": {}, "TMP_instant": {}, "APTMP_max": {}, "APTMP_min": {},
        "APCP_24": {}, "APCP_48": {}, "APCP_72": {}
    }

    try:
        with open(tmp_path, "rb") as f:
            while True:
                gid = eccodes.codes_grib_new_from_file(f)
                
                if gid is None:
                    break

                try:
                    raw_param = eccodes.codes_get(gid, "shortName")
                    param = raw_param.lower() if raw_param else ""

                    if param not in ("2t", "max_2t", "min_2t", "aptmp", "tp"):
                        continue

                    box_indices = get_city_box_indices(gid, cities)
                    values = eccodes.codes_get_values(gid)

                    # Nearest-point-only sample (used for QPF, which doesn't get box treatment)
                    city_nearest = {c["name"]: float(values[box_indices[c["name"]]["nearest"]]) for c in cities}

                    def city_box_extreme(reducer):
                        return {
                            c["name"]: float(reducer(values[i] for i in box_indices[c["name"]]["box"]))
                            for c in cities
                        }

                    city_samples = city_nearest

                    if param == "tp":
                        try:
                            upper_scale = eccodes.codes_get(gid, "scaleFactorOfUpperLimit")
                            upper_val = eccodes.codes_get(gid, "scaledValueOfUpperLimit")
                            
                            thresh_mm = upper_val / (10 ** upper_scale)
                            thresh_in = str(round(thresh_mm / 25.4, 2))
                        except Exception as e:
                           # log(f" [f{f_hour}] Skipped tp message: could not decode threshold ({e})")
                            continue

                        step_range = str(eccodes.codes_get(gid, "stepRange"))
                        if "-" in step_range:
                            parts = step_range.split("-")
                            hours_diff = int(parts[1]) - int(parts[0])
                        else:
                            hours_diff = int(step_range)

                        if hours_diff == 24:
                            extracted_data["APCP_24"].setdefault(thresh_in, city_samples)
                        elif hours_diff == 48:
                            extracted_data["APCP_48"].setdefault(thresh_in, city_samples)
                        elif hours_diff == 72:
                            extracted_data["APCP_72"].setdefault(thresh_in, city_samples)
                    else:
                        if param in ("max_2t", "min_2t"):
                            try:
                                pct = str(eccodes.codes_get(gid, "percentileValue"))
                            except Exception as e:
                             #   log(f" [f{f_hour}] Skipped {param} message: no percentileValue ({e})")
                                continue

                            if param == "max_2t":
                                # Hottest value within each city's 3x3 grid-box neighborhood
                                extracted_data["TMP_max"].setdefault(pct, city_box_extreme(max))
                            elif param == "min_2t":
                                # Coldest value within each city's 3x3 grid-box neighborhood
                                extracted_data["TMP_min"].setdefault(pct, city_box_extreme(min))

                        elif param == "2t":
                            continue

                        elif param == "aptmp":
                            try:
                                pct = str(eccodes.codes_get(gid, "percentileValue"))
                            except Exception as e:
                            #    log(f" [f{f_hour}] Skipped aptmp message: no percentileValue ({e})")
                                continue

                            # Same hourly message feeds both the daily-high (hottest-in-box) and
                            # daily-low (coldest-in-box) apparent-temp series downstream.
                            extracted_data["APTMP_max"].setdefault(pct, city_box_extreme(max))
                            extracted_data["APTMP_min"].setdefault(pct, city_box_extreme(min))

                except Exception as e:
                    log(f" [f{f_hour}] Message failed: {e}")
                finally:
                    eccodes.codes_release(gid)
    finally:
        os.unlink(tmp_path)

    return extracted_data

def calculate_shifted_cdf(envelope_map: dict[int, float], p50_values: list[float], is_heat: bool) -> list[dict]:
    if not envelope_map:
        return []

    # NOTE: This used to apply a statistical "daily max from hourly samples" stretch here
    # (driver = max(temp_range, ensemble_spread) -> effective_n -> probability warp). That was
    # dropped: temp_range (diurnal amplitude) dominated the max() on nearly every day regardless
    # of actual forecast confidence, so the stretch wasn't behaving as intended. The
    # "true instantaneous value could exceed the hourly sample" adjustment is now applied
    # client-side as a flat +/-1.5F nudge to the NDFD anchor used in bias correction
    # (see getBiasCorrectedPoints in heatnbm.html), rather than as a CDF-shape transform here.
    raw_xp = sorted(envelope_map.keys())
    raw_fp = [envelope_map[p] for p in raw_xp]

    target_increments = list(range(0, 101, 5))
    smooth_cdf = []

    for target_p in target_increments:
        val = round(float(np.interp(target_p, raw_xp, raw_fp)), 1)
        smooth_cdf.append({
            "p": target_p,
            "v": val
        })

    return smooth_cdf

def build_dataset() -> dict:
    run_dt = latest_complete_qmd_run()
    
    # Clean cache targets before beginning downloads
    purge_old_grib_cache(run_dt.strftime("%Y%m%d"), f"{run_dt.hour:02d}")
    
    maxt_fhours = anchor_forecast_hours(run_dt.hour, 6, 18)
    mint_fhours = anchor_forecast_hours(run_dt.hour, 18, 18)
    qpf_windows = {
        24: anchor_forecast_hours(run_dt.hour, 12, 24),
        48: anchor_forecast_hours(run_dt.hour, 12, 48),
        72: anchor_forecast_hours(run_dt.hour, 12, 72)
    }
    
    appt_fhours = list(range(1, 193))
    all_qpf_hours = set(qpf_windows[24] + qpf_windows[48] + qpf_windows[72])
    all_target_hours = sorted(list(set(maxt_fhours + mint_fhours + appt_fhours + list(all_qpf_hours))))
    
    dataset = {
        "run": {"date": run_dt.strftime("%Y-%m-%d"), "cycle": f"{run_dt.hour:02d}"},
        "cities": {c["name"]: {"lat": c["lat"], "lon": c["lon"], "sid": c.get("sid", None), "high": {}, "low": {}, "appt_hourly": {}, "qpf": {}, "appt_daily_cdf": {}} for c in CITIES},
    }

    total_steps = len(all_target_hours)
    
    log(f"=== Phase 1: Processing files (Concurrent Workers: 6) ===")
    downloaded_payloads: dict[int, bytes] = {}
    
    # Execute network requests concurrently up to 6 threads
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_hour = {executor.submit(fetch_filtered_grib, run_dt, f): f for f in all_target_hours}
        
        completed_count = 0
        for future in as_completed(future_to_hour):
            f = future_to_hour[future]
            completed_count += 1
            try:
                raw_grib, status = future.result()
                if raw_grib:
                    downloaded_payloads[f] = raw_grib
                log(f" [{completed_count}/{total_steps}] Processed f{f:03d} ({status})")
            except Exception as e:
                log(f" [{completed_count}/{total_steps}] Failed downloading f{f:03d}: {e}")

    missing_hours = [f for f in all_target_hours if f not in downloaded_payloads]
    log(f"=== Phase 1 complete: {len(downloaded_payloads)}/{total_steps} target hours downloaded ===")
    if missing_hours:
        log(f" Missing {len(missing_hours)} hour(s): {missing_hours}")
    if 192 in all_target_hours and 192 not in downloaded_payloads:
        log(" NOTE: f192 (longest lead time) did not come through this run.")

    log(f"=== Phase 2: Generating output dataset ===")
    for idx, f in enumerate(all_target_hours, 1):
        if f not in downloaded_payloads:
            continue
            
        hour_data = decode_nomads_file(downloaded_payloads[f], CITIES, f_hour=f)

        if f in maxt_fhours:
            date_str = maxt_valid_date(run_dt, f).isoformat()
            target_source = hour_data["TMP_max"] if hour_data["TMP_max"] else hour_data["TMP_instant"]
            for pct, city_vals in target_source.items():
                for name, k in city_vals.items():
                    temp_f = round((k - 273.15) * 9 / 5 + 32, 1)
                    dataset["cities"][name]["high"].setdefault(date_str, {"pcts": {}})
                    current_high = dataset["cities"][name]["high"][date_str]["pcts"].get(pct, -999)
                    if temp_f > current_high:
                        dataset["cities"][name]["high"][date_str]["pcts"][pct] = temp_f

        if f in mint_fhours:
            date_str = mint_valid_date(run_dt, f).isoformat()
            target_source = hour_data["TMP_min"] if hour_data["TMP_min"] else hour_data["TMP_instant"]
            for pct, city_vals in target_source.items():
                for name, k in city_vals.items():
                    temp_f = round((k - 273.15) * 9 / 5 + 32, 1)
                    dataset["cities"][name]["low"].setdefault(date_str, {"pcts": {}})
                    current_low = dataset["cities"][name]["low"][date_str]["pcts"].get(pct, 999)
                    if temp_f < current_low:
                        dataset["cities"][name]["low"][date_str]["pcts"][pct] = temp_f

        for window, fhours in qpf_windows.items():
            if f in fhours:
                date_str = (run_dt + timedelta(hours=f)).date().isoformat()
                target_key = f"APCP_{window}"
                if hour_data.get(target_key):
                    for thresh_in, city_prob_vals in hour_data[target_key].items():
                        for name, prob_val in city_prob_vals.items():
                            entry = dataset["cities"][name]["qpf"].setdefault(date_str, {})
                            win_entry = entry.setdefault(f"{window}hr", {"exceed": {}})
                            win_entry["exceed"][thresh_in] = round(prob_val, 1)

    log(f"=== Phase 3: Processing Daily Apparent Temperature Shifted CDFs ===")
    # Two parallel hourly series per city: the hottest-in-box reading (feeds the daily
    # "highest high" apparent-temp CDF) and the coldest-in-box reading (feeds the daily
    # "lowest low" apparent-temp CDF). These come from the same GRIB message per hour,
    # just reduced across each city's 3x3 grid-box neighborhood in opposite directions.
    temp_hourly_store_max = {c["name"]: {} for c in CITIES}
    temp_hourly_store_min = {c["name"]: {} for c in CITIES}
    for f in appt_fhours:
        if f not in downloaded_payloads:
            continue
        hour_data = decode_nomads_file(downloaded_payloads[f], CITIES, f_hour=f)
        valid_str = (run_dt + timedelta(hours=f)).strftime("%Y-%m-%dT%H:%M:%SZ")

        if hour_data.get("APTMP_max"):
            for pct, city_vals in hour_data["APTMP_max"].items():
                for name, k in city_vals.items():
                    temp_hourly_store_max[name].setdefault(valid_str, {"pcts": {}})
                    temp_hourly_store_max[name][valid_str]["pcts"][pct] = round((k - 273.15) * 9 / 5 + 32, 1)

        if hour_data.get("APTMP_min"):
            for pct, city_vals in hour_data["APTMP_min"].items():
                for name, k in city_vals.items():
                    temp_hourly_store_min[name].setdefault(valid_str, {"pcts": {}})
                    temp_hourly_store_min[name][valid_str]["pcts"][pct] = round((k - 273.15) * 9 / 5 + 32, 1)

    def group_by_day(hourly_records: dict) -> dict:
        groups: dict[str, list] = {}
        for timestamp, payload in hourly_records.items():
            day_str = timestamp.split("T")[0]
            groups.setdefault(day_str, []).append(payload["pcts"])
        return groups

    def build_envelope(hourly_pct_list: list, pick) -> dict:
        available_pcts = set()
        for hour_pcts in hourly_pct_list:
            for p_str in hour_pcts.keys():
                available_pcts.add(int(p_str))

        env = {}
        for p in available_pcts:
            p_str = str(p)
            vals = [h[p_str] for h in hourly_pct_list if p_str in h]
            if vals:
                env[p] = pick(vals)
        return env

    for name in dataset["cities"]:
        city_data = dataset["cities"][name]

        if "appt_hourly" in city_data:
            del city_data["appt_hourly"]

        max_daily_groups = group_by_day(temp_hourly_store_max.get(name, {}))
        min_daily_groups = group_by_day(temp_hourly_store_min.get(name, {}))

        for day_str in set(max_daily_groups) | set(min_daily_groups):
            max_hourly_pct_list = max_daily_groups.get(day_str, [])
            min_hourly_pct_list = min_daily_groups.get(day_str, [])

            # Day's running max, per percentile, of the hottest-in-box hourly readings
            max_envelope = build_envelope(max_hourly_pct_list, max)
            # Day's running min, per percentile, of the coldest-in-box hourly readings
            min_envelope = build_envelope(min_hourly_pct_list, min)

            max_p50_values = [h["50"] for h in max_hourly_pct_list if "50" in h]
            min_p50_values = [h["50"] for h in min_hourly_pct_list if "50" in h]

            max_cdf = calculate_shifted_cdf(max_envelope, max_p50_values, is_heat=True)
            min_cdf = calculate_shifted_cdf(min_envelope, min_p50_values, is_heat=False)

            city_data["appt_daily_cdf"][day_str] = {
                "max_apparent_tw": max_cdf,
                "min_apparent_tw": min_cdf
            }

    return dataset

if __name__ == "__main__":
    start_time = time.time()
    result = build_dataset()
    
    run_info = result["run"]
    date_obj = datetime.strptime(run_info["date"], "%Y-%m-%d")
    yymmddhh = f"{date_obj.strftime('%y%m%d')}{run_info['cycle']}"
    
    # Save as a .json.gz file extension
    filename = f"nbm_qmd_output_{yymmddhh}.json.gz"
    
    # Compress the JSON on the fly
    json_str = json.dumps(result, indent=2)
    with gzip.open(filename, "wb") as f:
        f.write(json_str.encode("utf-8"))
    
    elapsed = time.time() - start_time
    log(f"Compressed and wrote complete weather dataset to {filename} in {elapsed:.2f} seconds")
    
    # Update Phase 4 cleanup to track the .json.gz pattern
    log("=== Phase 4: Cleaning up older JSON cycle files ===")
    all_files = glob.glob("nbm_qmd_output_*.json.gz")
    cycle_file_pattern = re.compile(r"^nbm_qmd_output_\d{8}\.json\.gz$")
    matched_files = [f for f in all_files if cycle_file_pattern.match(f)]
    matched_files.sort()
    
    if len(matched_files) > 10:
        files_to_delete = matched_files[:-10]
        for f_del in files_to_delete:
            try:
                os.remove(f_del)
                log(f" Pruned old cycle file: {f_del}")
            except Exception as e:
                log(f" Failed to delete {f_del}: {e}")
    else:
        log(" Active JSON datasets total 10 or fewer. Retention boundaries met.")
