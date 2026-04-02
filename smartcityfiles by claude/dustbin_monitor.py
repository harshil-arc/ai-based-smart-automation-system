"""
dustbin_monitor.py
─────────────────────────────────────────────────────────────────
Smart City Dustbin Fill Level Monitor

Simulates a network of IoT-enabled smart dustbins across city zones.
Each bin has:
  • Unique ID and GPS location
  • Fill level (0–100%)
  • Status: EMPTY / QUARTER / HALF / FULL / CRITICAL
  • Zone, capacity, last update timestamp
  • Simulated sensor drift over time
"""

import random
import math
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────
# CITY ZONE DEFINITIONS
# ─────────────────────────────────────────────────────────────────
ZONES = {
    "Central Plaza":   {"fill_bias": 0.75, "capacity": 240},
    "Market Block":    {"fill_bias": 0.85, "capacity": 180},
    "North Gate":      {"fill_bias": 0.45, "capacity": 120},
    "South Corridor":  {"fill_bias": 0.50, "capacity": 120},
    "East Park":       {"fill_bias": 0.30, "capacity": 200},
    "West Residency":  {"fill_bias": 0.40, "capacity": 120},
    "Bus Terminal":    {"fill_bias": 0.80, "capacity": 300},
    "Hospital Zone":   {"fill_bias": 0.55, "capacity": 150},
    "School District": {"fill_bias": 0.60, "capacity": 120},
    "Industrial Area": {"fill_bias": 0.35, "capacity": 480},
}

BIN_LOCATIONS = {
    "Central Plaza":   ["CP-Gate-1", "CP-Fountain", "CP-Parking", "CP-Bench-Row"],
    "Market Block":    ["MKT-Entrance", "MKT-Row-A", "MKT-Row-B", "MKT-Exit", "MKT-Corner"],
    "North Gate":      ["NG-Entry", "NG-Road-1"],
    "South Corridor":  ["SC-Left", "SC-Right", "SC-Junction"],
    "East Park":       ["EP-Main", "EP-Jogging", "EP-Picnic"],
    "West Residency":  ["WR-Block1", "WR-Block2", "WR-Corner"],
    "Bus Terminal":    ["BT-Platform1", "BT-Platform2", "BT-Entry", "BT-Exit"],
    "Hospital Zone":   ["HZ-OPD", "HZ-Parking", "HZ-Emergency"],
    "School District": ["SD-Main", "SD-Side", "SD-Sports"],
    "Industrial Area": ["IA-Gate1", "IA-Gate2"],
}


def _status_from_level(fill_level: float) -> str:
    if fill_level >= 95:  return "CRITICAL"
    if fill_level >= 80:  return "FULL"
    if fill_level >= 50:  return "HALF"
    if fill_level >= 25:  return "QUARTER"
    return "EMPTY"


def _time_ago(minutes: int) -> str:
    t = datetime.now() - timedelta(minutes=minutes)
    return t.strftime("%H:%M")


# ─────────────────────────────────────────────────────────────────
# BIN FACTORY
# ─────────────────────────────────────────────────────────────────
def get_all_bins() -> list:
    """
    Generate all simulated bins with randomised fill levels
    weighted by zone activity bias.
    """
    bins = []
    bin_counter = 1

    for zone, locations in BIN_LOCATIONS.items():
        zone_info = ZONES.get(zone, {"fill_bias": 0.5, "capacity": 120})
        bias = zone_info["fill_bias"]
        cap  = zone_info["capacity"]

        for loc in locations:
            # Gaussian-distributed fill level centred on zone bias
            fill = random.gauss(bias * 100, 18)
            fill = max(0, min(100, fill))

            # Small chance of critical bin
            if random.random() < 0.08:
                fill = random.uniform(88, 100)

            # Small chance of freshly emptied bin
            if random.random() < 0.10:
                fill = random.uniform(0, 15)

            sensor_id  = f"SN{1000 + bin_counter:04d}"
            minutes_ago = random.randint(1, 45)

            bins.append({
                "bin_id":         f"BIN-{bin_counter:03d}",
                "zone":           zone,
                "location":       loc,
                "fill_level":     round(fill, 1),
                "status":         _status_from_level(fill),
                "capacity_litres": cap,
                "sensor_id":      sensor_id,
                "last_updated":   _time_ago(minutes_ago),
                "_minutes_ago":   minutes_ago,
                "_fill_rate":     random.uniform(0.05, 0.3),  # %/min
            })
            bin_counter += 1

    return bins


# ─────────────────────────────────────────────────────────────────
# REFRESH — simulate bins filling/emptying over time
# ─────────────────────────────────────────────────────────────────
def refresh_bin_levels(bins: list) -> list:
    """
    Advance fill levels based on simulated IoT sensor readings.
    Bins near CRITICAL randomly get serviced (emptied).
    """
    for b in bins:
        # Simulate fill increase
        delta = b["_fill_rate"] * random.uniform(0.5, 2.0)
        b["fill_level"] = min(100, b["fill_level"] + delta)

        # Simulate random emptying (garbage collection)
        if b["fill_level"] >= 90 and random.random() < 0.15:
            b["fill_level"] = random.uniform(2, 12)

        b["fill_level"] = round(b["fill_level"], 1)
        b["status"]      = _status_from_level(b["fill_level"])
        b["last_updated"] = datetime.now().strftime("%H:%M")

    return bins


# ─────────────────────────────────────────────────────────────────
# SINGLE BIN STATUS
# ─────────────────────────────────────────────────────────────────
def get_dustbin_status(bin_id: str, bins: list) -> dict | None:
    """Return a single bin dict by ID, or None if not found."""
    for b in bins:
        if b["bin_id"] == bin_id:
            return b
    return None


# ─────────────────────────────────────────────────────────────────
# ZONE SUMMARY
# ─────────────────────────────────────────────────────────────────
def get_zone_summary(bins: list) -> dict:
    """
    Returns per-zone aggregated stats:
    {zone: {avg_fill, max_fill, full_count, total_bins}}
    """
    summary = {}
    for b in bins:
        z = b["zone"]
        if z not in summary:
            summary[z] = {"fills": [], "full_count": 0, "total": 0}
        summary[z]["fills"].append(b["fill_level"])
        summary[z]["total"] += 1
        if b["status"] in ("FULL", "CRITICAL"):
            summary[z]["full_count"] += 1

    result = {}
    for z, s in summary.items():
        fills = s["fills"]
        result[z] = {
            "avg_fill":   round(sum(fills) / len(fills), 1),
            "max_fill":   round(max(fills), 1),
            "full_count": s["full_count"],
            "total_bins": s["total"],
        }
    return result


# ─────────────────────────────────────────────────────────────────
# COLLECTION PRIORITY QUEUE
# ─────────────────────────────────────────────────────────────────
def get_collection_priority(bins: list) -> list:
    """Return bins sorted by urgency (CRITICAL first, then by fill %)."""
    priority_order = {"CRITICAL": 0, "FULL": 1, "HALF": 2, "QUARTER": 3, "EMPTY": 4}
    return sorted(
        bins,
        key=lambda b: (priority_order.get(b["status"], 5), -b["fill_level"])
    )
