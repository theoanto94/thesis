from __future__ import annotations

import csv
import glob
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = "identifier.db"

# The thesis panel. Overridable so a throwaway database can be built over a
# different set of files without touching the thesis build.
CLAIMS_GLOB = "data/*_postcode3_en.csv"

# Forced to VARCHAR rather than left to the type sniffer: age_class holds '90+'
# alongside plain integers, and in the years Vektis doesn't pad the numeric ages
# the sample looks like clean BIGINT until the first '90+' blows up.
# _aggregate_claims TRIMs and re-casts these anyway.
_KEY_COLUMN_TYPES = {
    "gender": "VARCHAR",
    "age_class": "VARCHAR",
    "postal_code_3": "VARCHAR",
}

# Raw Dutch Vektis headers -> the English names used throughout the pipeline.
# Source names vary by year and are truncated in the 2023 file, so several
# Dutch keys can map to one English column.
TRANSLATION_MAP = {
    # Demographic columns
    'geslacht': 'gender',
    'leeftijdsklasse': 'age_class',
    'postcode_3': 'postal_code_3',
    'aantal_bsn': 'number_of_insured',
    'aantal_verzekerdejaren': 'insured_years',

    # Medical specialist care
    'kosten_medisch_specialistische_zorg': 'costs_medical_specialist_care',
    'kosten_medisch_specialistische_z': 'costs_medical_specialist_care',  # truncated in 2023

    # Pharmacy
    'kosten_farmacie': 'costs_pharmacy',

    # Mental health care (GGZ)
    'kosten_tweedelijns_ggz': 'costs_secondary_mental_health',           # 2011-2013
    'kosten_specialistische_ggz': 'costs_specialist_mental_health',      # 2014-2021
    'kosten_generalistische_basis_ggz': 'costs_general_basic_mental_health',  # 2014-2021
    'kosten_langdurige_ggz': 'costs_longterm_mental_health',             # 2015-2021
    'kosten_consulten_ggz': 'costs_mental_health_consultations',         # 2022+
    'kosten_intramuraal_verblijf_ggz': 'costs_inpatient_mental_health',  # 2022+
    'kosten_overige_prestaties_ggz': 'costs_other_mental_health_services',  # 2022+
    'kosten_innovatiegelden_ggz': 'costs_mental_health_innovation',      # 2021+

    # General practitioner (GP / Huisarts)
    'kosten_huisarts_inschrijftarief': 'costs_gp_registration_fee',
    'kosten_huisarts_consult': 'costs_gp_consultation',
    'kosten_huisarts_mdz': 'costs_gp_multidisciplinary_care',  # 2015+
    'kosten_huisarts_overig': 'costs_gp_other',

    # Medical aids and dental
    'kosten_hulpmiddelen': 'costs_medical_aids',
    'kosten_mondzorg': 'costs_dental_care',

    # Paramedical care
    'kosten_paramedische_zorg_fysiotherapie': 'costs_physiotherapy',
    'kosten_paramedische_zorg_fysioth': 'costs_physiotherapy',  # truncated in 2023
    'kosten_paramedische_zorg_overig': 'costs_paramedical_other',

    # Patient transport
    'kosten_ziekenvervoer_zittend': 'costs_patient_transport_seated',
    'kosten_ziekenvervoer_liggend': 'costs_patient_transport_ambulance',

    # Maternity and obstetric care
    'kosten_kraamzorg': 'costs_maternity_care',
    'kosten_verloskundige_zorg': 'costs_obstetric_care',
    'kosten_integrale_geboortezorg': 'costs_integrated_birth_care',  # 2021+

    # Psychology and primary care
    'kosten_eerstelijns_psychologische_zorg': 'costs_primary_psychological_care',  # 2011-2013
    'kosten_eerstelijns_ondersteuning': 'costs_primary_care_support',
    'kosten_eerstelijnsverblijf': 'costs_primary_care_residence',  # 2018+

    # Cross-border and geriatric care
    'kosten_grensoverschrijdende_zorg': 'costs_cross_border_care',
    'kosten_geriatrische_revalidatiezorg': 'costs_geriatric_rehabilitation',  # 2013+
    'kosten_geriatrische_revalidatiez': 'costs_geriatric_rehabilitation',  # truncated in 2023

    # Nursing and specific patient groups
    'kosten_verpleging_en_verzorging': 'costs_nursing_care',  # 2015+
    'kosten_gzsp': 'costs_medical_care_specific_patients',    # 2020+

    # Other costs
    'kosten_overig': 'costs_other',
}


def build(
    db_path: str = DB_PATH,
    translate_source: str | None = None,
    claims_glob: str | Sequence[str] = CLAIMS_GLOB,
) -> None:
    """Build the DuckDB database from the translated claims CSVs.

    If ``translate_source`` is given (a glob of raw, Dutch-header Vektis CSVs),
    the headers are translated to English via ``translate_headers`` first,
    producing the ``*_postcode3_en.csv`` files the rest of the pipeline reads.

    ``claims_glob`` selects which CSVs make up the panel. It defaults to the
    thesis data set; pass a wider set to build a throwaway database that
    includes years held outside ``data/``.
    """
    if translate_source is not None:
        translate_headers(translate_source)

    con = duckdb.connect(db_path)
    _load_claims(con, claims_glob)
    _load_cpi(con)
    _load_postcodes(con)
    _load_gebieden(con)
    _clean_claims(con)
    _aggregate_claims(con)
    _enrich_claims(con)
    con.close()
    print(f"Done. Database written to {db_path}")


# Step 0 – translate raw Vektis headers (Dutch -> English)

def _translate_header_line(header: str) -> tuple[str, list[str]]:
    """Translate one semicolon-delimited header line.

    Returns the translated header and a list of any source column names that
    had no entry in ``TRANSLATION_MAP`` (left unchanged so nothing is dropped).
    """
    fields = header.lstrip('﻿').rstrip('\r\n').split(';')
    translated: list[str] = []
    unknown: list[str] = []
    for field in fields:
        key = field.strip().lower()
        if key in TRANSLATION_MAP:
            translated.append(TRANSLATION_MAP[key])
        else:
            translated.append(field)
            if key:
                unknown.append(field)
    return ';'.join(translated), unknown


def translate_headers(
    src_glob: str,
    out_dir: str | None = None,
    suffix: str = '_en',
) -> list[Path]:
    """Rewrite raw Vektis CSVs with English headers, non-destructively.

    For each file matching ``src_glob``, the first (header) line is translated
    via ``TRANSLATION_MAP`` and a copy is written as ``<stem><suffix>.csv``
    (e.g. ``2011_postcode3.csv`` -> ``2011_postcode3_en.csv``). Data rows are
    copied verbatim. Files whose stem already ends in ``suffix`` are skipped so
    re-running is idempotent. Returns the list of paths written.
    """
    src_paths = sorted(Path(p) for p in glob.glob(src_glob))
    if not src_paths:
        print(f"No files matched {src_glob!r}; nothing to translate.")
        return []

    written: list[Path] = []
    for src in src_paths:
        if src.stem.endswith(suffix):
            continue  # already translated
        out_dir_path = Path(out_dir) if out_dir else src.parent
        dest = out_dir_path / f"{src.stem}{suffix}.csv"

        with src.open('r', encoding='utf-8') as fin:
            header = fin.readline()
            translated, unknown = _translate_header_line(header)
            if unknown:
                print(f"  WARNING: {src.name}: {len(unknown)} untranslated "
                      f"header(s) left as-is: {unknown}")
            out_dir_path.mkdir(parents=True, exist_ok=True)
            with dest.open('w', encoding='utf-8') as fout:
                fout.write(translated + '\n')
                for line in fin:
                    fout.write(line)
        print(f"  {src.name} -> {dest.name}")
        written.append(dest)

    print(f"Translated {len(written)} file(s).")
    return written


# Step 1 – raw claims

def _load_claims(
    con: duckdb.DuckDBPyConnection,
    claims_glob: str | Sequence[str] = CLAIMS_GLOB,
) -> None:
    """Read the claims CSVs into the ``claims`` table.

    DuckDB unions the matches by name, so the year-varying mental-health
    schemas coexist.
    """
    globs = [claims_glob] if isinstance(claims_glob, str) else list(claims_glob)
    print(f"Loading raw claims from {', '.join(globs)}...")
    types = ", ".join(f"'{col}': '{sql}'" for col, sql in _KEY_COLUMN_TYPES.items())
    con.execute(rf"""
        CREATE OR REPLACE TABLE claims AS
        SELECT
            *,
            -- From the basename, so a directory with digits in it cannot
            -- hijack the year.
            CAST(substr(regexp_extract(regexp_extract(filename, '[^/]+$'), '\d+'), 1, 4)
                 AS INTEGER) AS year,
            COALESCE(costs_secondary_mental_health, 0)
            + COALESCE(costs_primary_psychological_care, 0)
            + COALESCE(costs_specialist_mental_health, 0)
            + COALESCE(costs_general_basic_mental_health, 0)
            + COALESCE(costs_longterm_mental_health, 0)
            + COALESCE(costs_mental_health_innovation, 0)
            + COALESCE(costs_mental_health_consultations, 0)
            + COALESCE(costs_inpatient_mental_health, 0)
            + COALESCE(costs_other_mental_health_services, 0)
                AS costs_mental_health_total
        FROM read_csv(
            {globs!r},
            delim         = ';',
            header        = true,
            union_by_name = true,
            null_padding  = true,
            types         = {{{types}}}
        )
    """)


# Step 2 – CPI index (2020-rebased)

def _load_cpi(con: duckdb.DuckDBPyConnection) -> None:
    print("Loading CPI...")
    con.execute("""
        CREATE OR REPLACE TABLE cpi AS
        WITH raw AS (
            SELECT
                CAST("Periods" AS INTEGER) AS year,
                CAST(replace("CPI (2015 = 100)", ',', '.') AS DOUBLE)         AS cpi_2015_base,
                CAST(replace("Derived CPI (2015 = 100)", ',', '.') AS DOUBLE) AS derived_cpi_2015_base
            FROM read_csv(
                'data/cpi_price_index_health_cbs_2011_2025.csv',
                delim       = ';',
                header      = true,
                all_varchar = true
            )
            WHERE TRY_CAST("Periods" AS INTEGER) IS NOT NULL
        ),
        anchor AS (
            SELECT cpi_2015_base AS cpi_anchor, derived_cpi_2015_base AS derived_anchor
            FROM raw WHERE year = 2020
        )
        SELECT
            r.year,
            r.cpi_2015_base,
            r.derived_cpi_2015_base,
            ROUND(r.cpi_2015_base         / a.cpi_anchor    * 100, 4) AS cpi_2020_base,
            ROUND(r.derived_cpi_2015_base / a.derived_anchor * 100, 4) AS derived_cpi_2020_base
        FROM raw r, anchor a
        ORDER BY year
    """)


# Step 3 – postcode → municipality lookup

def _load_postcodes(con: duckdb.DuckDBPyConnection) -> None:
    print("Loading postcodes...")
    con.execute("""
        CREATE OR REPLACE TABLE postcodes AS
        -- From 2023, Hellevoetsluis, Brielle, Oostvoorne, Oudenhoorn, Rockanje, Tinte,
        -- Vierpolders, and Zwartewaal were merged into Voorne aan Zee.
        WITH raw AS (
            SELECT
                LEFT(postal_code, 3)                                   AS postal_code_3,
                COALESCE(NULLIF(TRIM(gementee), ''), 'Voorne aan Zee') AS gementee,
                TRY_CAST(latitude   AS DOUBLE)                         AS latitude,
                TRY_CAST(longtitude AS DOUBLE)                         AS longitude
            FROM read_csv(
                'data/NL_full.csv',
                header      = true,
                all_varchar = true
            )
            WHERE postal_code IS NOT NULL
        )
        SELECT
            postal_code_3,
            mode(gementee)           AS gementee,
            ROUND(AVG(latitude),  6) AS latitude,
            ROUND(AVG(longitude), 6) AS longitude
        FROM raw
        GROUP BY postal_code_3
    """)


# Step 4 – Gebieden → gementee_code lookup

def _norm_gebieden(name: str) -> str:
    name = re.sub(r'\s*\([A-Za-z\.\s]{1,15}\)\s*$', '', name.strip())
    name = re.sub(r',\s+.+$', '', name)  # e.g. 'Nuenen, Gerwen en Nederwetten' → 'Nuenen'
    return name.strip()


def _norm_nl(name: str) -> str:
    name = re.sub(r'\s+Municipalit[a-z]+$', '', name.strip(), flags=re.IGNORECASE)
    name = re.sub(r'^Gemeente\s+', '', name, flags=re.IGNORECASE)
    return name.rstrip('.').strip()


# NL_full names that normalization alone cannot resolve.
_OVERRIDES: dict = {
    'Bergen':                    'GM0373',  # Bergen (NH.), North Holland
    'Bergen Municipality':       'GM0893',  # Bergen (L.), Limburg
    "Gemeente 's-Hertogenbosch": 'GM0796',
    'Gemeente Den Haag':         'GM0518',  # officially 's-Gravenhage
    'Katwijk Municipalty':       'GM0537',  # typo in NL_full source
    'Nuenen':                    'GM0820',  # full name: Nuenen, Gerwen en Nederwetten
    'Sûdwest Fryslân':           'GM1900',  # Gebieden spelling: Súdwest-Fryslân
    'Voorne aan Zee':            'GM1992',  # new 2023 municipality (merged)
    'Hoogland Municipality':     'GM0307',  # dissolved into Amersfoort
}

# Source name -> display name in the output.
_RENAMES: dict[str, str] = {
    'Hoogland Municipality': 'Amersfoort Municipality',
}


def _load_gebieden(con: duckdb.DuckDBPyConnection) -> None:
    print("Loading Gebieden and building gementee_code lookup...")

    gebieden_rows: list[dict] = []
    with open('data/Gebieden_in_Nederland_2023.csv') as f:
        for row in csv.DictReader(f, delimiter=';'):
            region = row['Region'].strip()
            code   = row['gementee_code'].strip()
            if not region or not code:
                continue
            if 'GM0518' in region:  # malformed 's-Gravenhage row in source CSV
                gebieden_rows.append({"region": "'s-Gravenhage (gemeente)", "region_norm": "'s-Gravenhage", "gementee_code": "GM0518"})
            elif 'GM0796' in region:  # malformed 's-Hertogenbosch row in source CSV
                gebieden_rows.append({"region": "'s-Hertogenbosch", "region_norm": "'s-Hertogenbosch", "gementee_code": "GM0796"})
            else:
                gebieden_rows.append({"region": region, "region_norm": _norm_gebieden(region), "gementee_code": code})

    df_gebieden = pd.DataFrame(gebieden_rows)
    con.register('_df_gebieden', df_gebieden)
    con.execute("CREATE OR REPLACE TABLE gebieden AS SELECT * FROM _df_gebieden")

    gementees = con.execute(
        "SELECT DISTINCT gementee FROM postcodes WHERE gementee IS NOT NULL"
    ).fetchall()

    lookup_rows: list[dict] = []
    unresolved: list[str] = []
    for (g,) in gementees:
        g = g.strip()
        display = _RENAMES.get(g, g)
        if g in _OVERRIDES:
            lookup_rows.append({"gementee": g, "gementee_display": display, "gementee_code": _OVERRIDES[g]})
        else:
            norm = _norm_nl(g)
            result = con.execute(
                "SELECT gementee_code FROM gebieden WHERE region_norm = ?", [norm]
            ).fetchone()
            if result is None:
                unresolved.append(g)
            lookup_rows.append({"gementee": g, "gementee_display": display, "gementee_code": result[0] if result else None})

    if unresolved:
        print(f"  WARNING: {len(unresolved)} gementee(s) could not be resolved to a code: {unresolved}")

    df_lookup = pd.DataFrame(lookup_rows)
    con.register('_df_lookup', df_lookup)
    con.execute("CREATE OR REPLACE TABLE gementee_lookup AS SELECT * FROM _df_lookup")


# Step 5 – remove null-gender rows

def _clean_claims(con: duckdb.DuckDBPyConnection) -> None:
    print("Cleaning claims...")
    con.execute("""
        DELETE FROM claims
        WHERE gender IS NULL
           OR LOWER(TRIM(gender)) = '(null)'
    """)


# Step 6 – aggregate to 5-year age bands and apply CPI deflation

def _aggregate_claims(con: duckdb.DuckDBPyConnection) -> None:
    print("Aggregating and deflating claims...")
    con.execute("""
        CREATE OR REPLACE TABLE claims AS
        WITH grouped AS (
            SELECT
                gender,
                postal_code_3,
                year,
                CASE
                    WHEN TRIM(age_class::VARCHAR) = '90+' THEN '90+'
                    WHEN TRY_CAST(TRIM(age_class::VARCHAR) AS INTEGER) IS NOT NULL THEN
                        -- '//' is integer division in DuckDB; '/' returns DOUBLE and breaks LPAD
                        LPAD(CAST((TRY_CAST(TRIM(age_class::VARCHAR) AS INTEGER) // 5) * 5 AS VARCHAR), 2, '0')
                        || '-'
                        || LPAD(CAST((TRY_CAST(TRIM(age_class::VARCHAR) AS INTEGER) // 5) * 5 + 4 AS VARCHAR), 2, '0')
                    ELSE NULL
                END AS age_group,
                SUM(number_of_insured)                                        AS number_of_insured,
                ROUND(SUM(insured_years),                        2)           AS insured_years,
                ROUND(SUM(costs_medical_specialist_care),        2)           AS costs_medical_specialist_care,
                ROUND(SUM(costs_pharmacy),                       2)           AS costs_pharmacy,
                ROUND(SUM(costs_gp_registration_fee),            2)           AS costs_gp_registration_fee,
                ROUND(SUM(costs_gp_consultation),                2)           AS costs_gp_consultation,
                ROUND(SUM(costs_gp_multidisciplinary_care),      2)           AS costs_gp_multidisciplinary_care,
                ROUND(SUM(costs_gp_other),                       2)           AS costs_gp_other,
                ROUND(SUM(costs_medical_aids),                   2)           AS costs_medical_aids,
                ROUND(SUM(costs_dental_care),                    2)           AS costs_dental_care,
                ROUND(SUM(costs_physiotherapy),                  2)           AS costs_physiotherapy,
                ROUND(SUM(costs_paramedical_other),              2)           AS costs_paramedical_other,
                ROUND(SUM(costs_patient_transport_seated),       2)           AS costs_patient_transport_seated,
                ROUND(SUM(costs_patient_transport_ambulance),    2)           AS costs_patient_transport_ambulance,
                ROUND(SUM(costs_maternity_care),                 2)           AS costs_maternity_care,
                ROUND(SUM(costs_obstetric_care),                 2)           AS costs_obstetric_care,
                ROUND(SUM(costs_cross_border_care),              2)           AS costs_cross_border_care,
                ROUND(SUM(costs_primary_care_support),           2)           AS costs_primary_care_support,
                ROUND(SUM(costs_geriatric_rehabilitation),       2)           AS costs_geriatric_rehabilitation,
                ROUND(SUM(costs_primary_care_residence),         2)           AS costs_primary_care_residence,
                ROUND(SUM(costs_nursing_care),                   2)           AS costs_nursing_care,
                ROUND(SUM(costs_medical_care_specific_patients), 2)           AS costs_medical_care_specific_patients,
                ROUND(SUM(costs_integrated_birth_care),          2)           AS costs_integrated_birth_care,
                ROUND(SUM(costs_mental_health_innovation),       2)           AS costs_mental_health_innovation,
                ROUND(SUM(costs_other),                          2)           AS costs_other,
                ROUND(SUM(costs_mental_health_consultations),    2)           AS costs_mental_health_consultations,
                ROUND(SUM(costs_inpatient_mental_health),        2)           AS costs_inpatient_mental_health,
                ROUND(SUM(costs_other_mental_health_services),   2)           AS costs_other_mental_health_services,
                ROUND(SUM(costs_mental_health_total),            2)           AS costs_mental_health_total
            FROM claims
            GROUP BY gender, postal_code_3, year, age_group
        )
        SELECT
            gender,
            postal_code_3,
            g.year,
            age_group,
            number_of_insured,
            insured_years,
            ROUND(g.costs_medical_specialist_care        / (c.cpi_2020_base / 100), 2) AS costs_medical_specialist_care_real,
            ROUND(g.costs_pharmacy                       / (c.cpi_2020_base / 100), 2) AS costs_pharmacy_real,
            ROUND(g.costs_gp_registration_fee            / (c.cpi_2020_base / 100), 2) AS costs_gp_registration_fee_real,
            ROUND(g.costs_gp_consultation                / (c.cpi_2020_base / 100), 2) AS costs_gp_consultation_real,
            ROUND(g.costs_gp_multidisciplinary_care      / (c.cpi_2020_base / 100), 2) AS costs_gp_multidisciplinary_care_real,
            ROUND(g.costs_gp_other                       / (c.cpi_2020_base / 100), 2) AS costs_gp_other_real,
            ROUND(g.costs_medical_aids                   / (c.cpi_2020_base / 100), 2) AS costs_medical_aids_real,
            ROUND(g.costs_dental_care                    / (c.cpi_2020_base / 100), 2) AS costs_dental_care_real,
            ROUND(g.costs_physiotherapy                  / (c.cpi_2020_base / 100), 2) AS costs_physiotherapy_real,
            ROUND(g.costs_paramedical_other              / (c.cpi_2020_base / 100), 2) AS costs_paramedical_other_real,
            ROUND(g.costs_patient_transport_seated       / (c.cpi_2020_base / 100), 2) AS costs_patient_transport_seated_real,
            ROUND(g.costs_patient_transport_ambulance    / (c.cpi_2020_base / 100), 2) AS costs_patient_transport_ambulance_real,
            ROUND(g.costs_maternity_care                 / (c.cpi_2020_base / 100), 2) AS costs_maternity_care_real,
            ROUND(g.costs_obstetric_care                 / (c.cpi_2020_base / 100), 2) AS costs_obstetric_care_real,
            ROUND(g.costs_cross_border_care              / (c.cpi_2020_base / 100), 2) AS costs_cross_border_care_real,
            ROUND(g.costs_primary_care_support           / (c.cpi_2020_base / 100), 2) AS costs_primary_care_support_real,
            ROUND(g.costs_geriatric_rehabilitation       / (c.cpi_2020_base / 100), 2) AS costs_geriatric_rehabilitation_real,
            ROUND(g.costs_primary_care_residence         / (c.cpi_2020_base / 100), 2) AS costs_primary_care_residence_real,
            ROUND(g.costs_nursing_care                   / (c.cpi_2020_base / 100), 2) AS costs_nursing_care_real,
            ROUND(g.costs_medical_care_specific_patients / (c.cpi_2020_base / 100), 2) AS costs_medical_care_specific_patients_real,
            ROUND(g.costs_integrated_birth_care          / (c.cpi_2020_base / 100), 2) AS costs_integrated_birth_care_real,
            ROUND(g.costs_mental_health_innovation       / (c.cpi_2020_base / 100), 2) AS costs_mental_health_innovation_real,
            ROUND(g.costs_other                          / (c.cpi_2020_base / 100), 2) AS costs_other_real,
            ROUND(g.costs_mental_health_consultations    / (c.cpi_2020_base / 100), 2) AS costs_mental_health_consultations_real,
            ROUND(g.costs_inpatient_mental_health        / (c.cpi_2020_base / 100), 2) AS costs_inpatient_mental_health_real,
            ROUND(g.costs_other_mental_health_services   / (c.cpi_2020_base / 100), 2) AS costs_other_mental_health_services_real,
            ROUND(g.costs_mental_health_total            / (c.cpi_2020_base / 100), 2) AS costs_mental_health_total_real
        FROM grouped g
        JOIN cpi c ON g.year = c.year
        ORDER BY g.year, g.postal_code_3, g.gender, g.age_group
    """)


# Step 7 – enrich claims with gementee, gementee_code, and coordinates

def _enrich_claims(con: duckdb.DuckDBPyConnection) -> None:
    print("Enriching claims with municipality and coordinates...")
    con.execute("""
        CREATE OR REPLACE TABLE claims AS
        -- Some round-hundred postcode-3 areas (e.g. 100, 300, 500) have no direct match in
        -- the postcodes lookup; fall back to postcode_3 + 1 (e.g. 101, 301, 501) in that case.
        SELECT
            c.*,
            COALESCE(gl.gementee_display, p1.gementee, p2.gementee) AS gementee,
            gl.gementee_code,
            COALESCE(p1.latitude,  p2.latitude)  AS latitude,
            COALESCE(p1.longitude, p2.longitude) AS longitude
        FROM claims c
        LEFT JOIN postcodes p1
            ON c.postal_code_3 = p1.postal_code_3
        LEFT JOIN postcodes p2
            ON p1.postal_code_3 IS NULL
           AND LPAD(CAST(TRY_CAST(c.postal_code_3 AS INTEGER) + 1 AS VARCHAR), 3, '0') = p2.postal_code_3
        LEFT JOIN gementee_lookup gl
            ON COALESCE(p1.gementee, p2.gementee) = gl.gementee
    """)

    row_count = con.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    print(f"  claims table: {row_count:,} rows")



if __name__ == "__main__":
    # Usage:
    #   python preprocessing.py [db_path]
    #       Build the DB from the existing *_postcode3_en.csv files.
    #   python preprocessing.py translate <src_glob> [out_dir]
    #       Translate raw Dutch-header Vektis CSVs to *_en.csv only (no build).
    #   python preprocessing.py build-from <src_glob> [db_path]
    #       Translate raw Vektis CSVs, then build the DB.
    #   python preprocessing.py build-glob <db_path> <glob> [<glob> ...]
    #       Build a DB from an explicit set of *_en.csv globs (no translation).
    args = sys.argv[1:]
    if args and args[0] == "build-glob":
        if len(args) < 3:
            sys.exit("usage: python preprocessing.py build-glob <db_path> <glob> [<glob> ...]")
        build(args[1], claims_glob=args[2:])
        sys.exit(0)
    if args and args[0] == "translate":
        if len(args) < 2:
            sys.exit("usage: python preprocessing.py translate <src_glob> [out_dir]")
        translate_headers(args[1], out_dir=args[2] if len(args) > 2 else None)
    elif args and args[0] == "build-from":
        if len(args) < 2:
            sys.exit("usage: python preprocessing.py build-from <src_glob> [db_path]")
        build(args[2] if len(args) > 2 else DB_PATH, translate_source=args[1])
    else:
        build(args[0] if args else DB_PATH)
