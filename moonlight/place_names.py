# SPDX-License-Identifier: Apache-2.0
"""Maldivian place-name reference database (Phase B3).

Populated from GeoNames MV data (islands, atolls, cities, reefs, shoals)
plus a hardcoded Wikipedia 20-atoll traditional-name supplement.

Primary use: `lookup_place_names_for_text` scans a Dhivehi (Thaana) source
text and returns canonical English romanisations, which the translator then
injects as a per-text reference block — replacing the static rule-of-thumb
it previously used.

Build command: `kahzaabu translate build-place-names`
"""
from __future__ import annotations

import io
import logging
import re
import sqlite3
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Thaana script range U+0780–U+07BF
_THAANA_RE = re.compile(r"[ހ-޿]")

_GEONAMES_MAIN_URL = "http://download.geonames.org/export/dump/MV.zip"
_GEONAMES_ALT_URL  = "http://download.geonames.org/export/dump/alternatenames/MV.zip"

PLACE_NAMES_SCHEMA = """
CREATE TABLE IF NOT EXISTS place_names (
    geonameid    INTEGER PRIMARY KEY,
    en_name      TEXT NOT NULL,
    dv_thaana    TEXT,
    dv_latin     TEXT,
    en_name_po   TEXT,
    feature_code TEXT,
    atoll_code   TEXT,
    latitude     REAL,
    longitude    REAL,
    source       TEXT NOT NULL DEFAULT 'geonames'
);
CREATE INDEX IF NOT EXISTS idx_place_names_thaana
    ON place_names(dv_thaana);
CREATE INDEX IF NOT EXISTS idx_place_names_en
    ON place_names(en_name);
CREATE INDEX IF NOT EXISTS idx_place_names_feature
    ON place_names(feature_code);
"""

# Wikipedia 20-atoll supplement.
# Keys are the GeoNames admin names; values are the traditional English names
# used in PO press releases and geographic literature.
# Source: https://en.wikipedia.org/wiki/Administrative_divisions_of_the_Maldives
ATOLL_TRADITIONAL: dict[str, str] = {
    "Haa Alifu Atoll":   "North Thiladhunmathi Atoll",
    "Haa Dhaalu Atoll":  "South Thiladhunmathi Atoll",
    "Shaviyani Atoll":   "North Miladhunmadulu Atoll",
    "Noonu Atoll":       "South Miladhunmadulu Atoll",
    "Raa Atoll":         "North Maalhosmadulu Atoll",
    "Baa Atoll":         "South Maalhosmadulu Atoll",
    "Lhaviyani Atoll":   "Faadhippolhu Atoll",
    "Kaafu Atoll":       "North Malé Atoll",
    "Alifu Alifu Atoll": "North Ari Atoll",
    "Alifu Dhaalu Atoll":"South Ari Atoll",
    "Vaavu Atoll":       "Felidhu Atoll",
    "Meemu Atoll":       "Mulaku Atoll",
    "Faafu Atoll":       "North Nilandhe Atoll",
    "Dhaalu Atoll":      "South Nilandhe Atoll",
    "Thaa Atoll":        "Kolhumadulu Atoll",
    "Laamu Atoll":       "Hadhdhunmathi Atoll",
    "Gaafu Alifu Atoll": "North Huvadhoo Atoll",
    "Gaafu Dhaalu Atoll":"South Huvadhoo Atoll",
    "Gnaviyani Atoll":   "Fuvahmulah",
    "Seenu Atoll":       "Addu Atoll",
}

# PO corpus spelling corrections: GeoNames en_name → PO canonical form.
# Add entries here whenever the corpus reveals a GeoNames name that differs
# from what the Presidency Office actually publishes.
_PO_CORRECTIONS: dict[str, str] = {
    "Male":         "Malé",
    "Kanditheemu":  "Kan'ditheemu",
    "Hanimaadhoo":  "Hanimaadhoo",
    "Fuvahmulah":   "Fuvahmulah",
    "Addu City":    "Addu City",
    "Hulhumale":    "Hulhumalé",
    "Maafushi":     "Maafushi",
    "Kudahuvadhoo": "Kudahuvadhoo",
}


def init_place_names(conn: sqlite3.Connection) -> bool:
    """Create place_names table + indexes. Idempotent. Returns True if newly created."""
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE name='place_names'"
    ).fetchone()
    conn.executescript(PLACE_NAMES_SCHEMA)
    conn.commit()
    return existing is None


def _is_thaana(s: str) -> bool:
    return bool(_THAANA_RE.search(s))


def _parse_main(data: bytes) -> dict[int, dict]:
    """Parse GeoNames MV.txt → {geonameid: row_dict}."""
    rows: dict[int, dict] = {}
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 11:
            continue
        try:
            gid = int(parts[0])
        except ValueError:
            continue
        name = parts[1].strip()       # name (may have diacritics)
        ascii_name = parts[2].strip() # asciiname (ASCII-only)
        # Prefer the name with diacritics; fall back to asciiname
        en_name = name if name else ascii_name
        try:
            lat = float(parts[4]) if parts[4] else None
            lon = float(parts[5]) if parts[5] else None
        except ValueError:
            lat = lon = None
        feature_code = parts[7].strip()
        admin1_code = parts[10].strip()
        rows[gid] = {
            "en_name": en_name,
            "feature_code": feature_code,
            "atoll_code": f"MV.{admin1_code}" if admin1_code else None,
            "latitude": lat,
            "longitude": lon,
        }
    return rows


def _parse_altnames(data: bytes) -> dict[int, dict]:
    """Parse alternatenames/MV.txt → {geonameid: {dv_thaana?, dv_latin?}}.

    dv rows come in pairs per geonameid — one Thaana, one Latin romanisation.
    We pick the shortest Latin form when multiple exist.
    """
    per_id: dict[int, dict] = {}
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        if parts[2].strip().lower() != "dv":
            continue
        try:
            gid = int(parts[1])
        except ValueError:
            continue
        name = parts[3].strip()
        if not name:
            continue
        bucket = per_id.setdefault(gid, {})
        if _is_thaana(name):
            bucket["dv_thaana"] = name
        else:
            existing = bucket.get("dv_latin", "")
            # Prefer shorter / simpler Latin form when multiple exist
            if not existing or len(name) < len(existing):
                bucket["dv_latin"] = name
    return per_id


def _download_zip(url: str, *, timeout: int = 30) -> bytes:
    """Download a GeoNames zip URL; return the first contained file's bytes."""
    log.info("downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "kahzaabu-place-names/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        first = zf.namelist()[0]
        return zf.read(first)


def _load_or_download(url: str, cache_path: Path, *, timeout: int = 30) -> bytes:
    if cache_path.exists():
        log.info("using cached %s", cache_path)
        return cache_path.read_bytes()
    data = _download_zip(url, timeout=timeout)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    log.info("cached %d bytes → %s", len(data), cache_path)
    return data


def build_place_names(conn: sqlite3.Connection, *,
                       data_dir: Optional[Path] = None,
                       timeout: int = 30) -> dict:
    """Download GeoNames MV data and populate the place_names table.

    Idempotent: upserts on geonameid so re-running refreshes data.
    Downloads are cached under data_dir to avoid repeat HTTP fetches.

    Returns:
        {"upserted": int, "total": int, "with_thaana": int}
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data" / "geonames"
    data_dir.mkdir(parents=True, exist_ok=True)

    main_data = _load_or_download(
        _GEONAMES_MAIN_URL,
        data_dir / "MV.txt",
        timeout=timeout,
    )
    alt_data = _load_or_download(
        _GEONAMES_ALT_URL,
        data_dir / "alternatenames_MV.txt",
        timeout=timeout,
    )

    main_rows = _parse_main(main_data)
    alt_rows  = _parse_altnames(alt_data)

    # Build a case-insensitive lookup for atoll supplement
    _atoll_lower = {k.lower(): v for k, v in ATOLL_TRADITIONAL.items()}

    upserted = 0
    for gid, row in main_rows.items():
        alt = alt_rows.get(gid, {})
        dv_thaana = alt.get("dv_thaana")
        dv_latin  = alt.get("dv_latin")
        en_name   = row["en_name"]
        feature_code = row["feature_code"]

        # PO corpus correction (specific island spellings)
        en_name_po = _PO_CORRECTIONS.get(en_name)

        # Wikipedia atoll supplement (ADM1 + ATOL feature codes)
        if feature_code in ("ADM1", "ATOL") and en_name_po is None:
            en_name_po = _atoll_lower.get(en_name.lower())

        conn.execute(
            """INSERT INTO place_names
               (geonameid, en_name, dv_thaana, dv_latin, en_name_po,
                feature_code, atoll_code, latitude, longitude, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'geonames')
               ON CONFLICT(geonameid) DO UPDATE SET
                 en_name      = excluded.en_name,
                 dv_thaana    = excluded.dv_thaana,
                 dv_latin     = excluded.dv_latin,
                 en_name_po   = excluded.en_name_po,
                 feature_code = excluded.feature_code,
                 atoll_code   = excluded.atoll_code,
                 latitude     = excluded.latitude,
                 longitude    = excluded.longitude""",
            (gid, en_name, dv_thaana, dv_latin, en_name_po,
             feature_code, row["atoll_code"], row["latitude"], row["longitude"]),
        )
        upserted += 1

    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM place_names").fetchone()[0]
    with_thaana = conn.execute(
        "SELECT COUNT(*) FROM place_names WHERE dv_thaana IS NOT NULL"
    ).fetchone()[0]
    return {"upserted": upserted, "total": total, "with_thaana": with_thaana}


def lookup_place_names_for_text(conn: sqlite3.Connection,
                                 source_text: str,
                                 *, max_results: int = 15) -> list[dict]:
    """Find known Maldivian place names whose Thaana form appears in source_text.

    Uses SQLite INSTR for substring matching — efficient for the ~1,100 dv rows
    in the MV dataset against a typical press-release body (~300–600 chars Thaana).

    Returns list of dicts ordered longest-match first (more-specific names
    take priority over common prefixes):
        {"dv_thaana": str, "en_name": str, "feature_code": str}

    `en_name` is the PO-calibrated form (en_name_po) when available, else the
    raw GeoNames romanisation.
    """
    if not source_text or not source_text.strip():
        return []

    rows = conn.execute(
        """SELECT dv_thaana, en_name, en_name_po, feature_code
           FROM place_names
           WHERE dv_thaana IS NOT NULL
             AND INSTR(?, dv_thaana) > 0
           ORDER BY LENGTH(dv_thaana) DESC
           LIMIT ?""",
        (source_text, max_results),
    ).fetchall()

    result = []
    for row in rows:
        dv_thaana, en_name_raw, en_name_po, feature_code = (
            row[0], row[1], row[2], row[3]
        )
        result.append({
            "dv_thaana":    dv_thaana,
            "en_name":      en_name_po if en_name_po else en_name_raw,
            "feature_code": feature_code,
        })
    return result


def format_place_name_block(places: list[dict]) -> str:
    """Format a list of place-name matches as a translator prompt block.

    Returns an empty string when places is empty (so the prompt is unaffected
    when the source text doesn't contain any recognised place names).
    """
    if not places:
        return ""

    _FEAT_LABEL = {
        "ISL":  "island",
        "ATOL": "atoll",
        "ADM1": "atoll (administrative)",
        "PPL":  "city/town",
        "HTL":  "hotel/resort",
        "SHOL": "shoal",
        "RF":   "reef",
        "PPLC": "capital city",
    }

    lines = ["\nMALDIVIAN PLACE NAMES DETECTED IN THIS TEXT:"]
    lines.append("Use EXACTLY these canonical English romanisations:")
    for p in places:
        label = _FEAT_LABEL.get(p["feature_code"], "place")
        lines.append(f"  {p['dv_thaana']} → {p['en_name']} ({label})")
    lines.append(
        "Do NOT drop final 'u', apostrophes, or diacritics (e.g. Malé, "
        "Kan'ditheemu). Use 'North/South [Name] Atoll' for atoll names."
    )
    return "\n".join(lines) + "\n"
