"""Marimo notebook for exploring the claims panel in identifier.db.

    uv run marimo edit eda_analysis.py
"""

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import warnings
    return duckdb, mo, warnings


@app.cell
def _(warnings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
    return


@app.cell
def _(mo):
    mo.md("""
    # Healthcare costs exploration
    """)
    return


@app.cell
def _(duckdb):
    # One table from all yearly CSVs, with 'year' taken from the filename.
    con = duckdb.connect("identifier.db")
    con.execute("""
        CREATE OR REPLACE TABLE claims AS
        SELECT
            *,
            CAST(substr(regexp_extract(filename, '\d+'), 1, 4) AS INTEGER) AS year,
            -- MH columns were renamed twice; sum all variants so NULLs (absent in a given year) become 0
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
            'data/*_postcode3_en.csv',
            delim          = ';',
            header         = true,
            union_by_name  = true,
            null_padding   = true
        )
    """)
    return (con,)


@app.cell
def _(con):
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
    return


@app.cell
def _(con):
    con.execute("""
        CREATE OR REPLACE TABLE postcodes AS
        -- From 2023, Hellevoetsluis, Brielle, Oostvoorne, Oudenhoorn, Rockanje, Tinte,
        -- Vierpolders, and Zwartewaal were merged into the new municipality Voorne aan Zee.
        -- NULL/empty gementee values are therefore filled with 'Voorne aan Zee'.
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
    return


@app.cell
def _(con):
    def _():
        import re
        import csv
        import pandas as pd

        def _norm_gebieden(name):
            """Strip province qualifiers and long official suffixes from Gebieden region names."""
            name = re.sub(r'\s*\([A-Za-z\.\s]{1,15}\)\s*$', '', name.strip())
            name = re.sub(r',\s+.+$', '', name)  # e.g. 'Nuenen, Gerwen en Nederwetten' → 'Nuenen'
            return name.strip()

        def _norm_nl(name):
            """Strip 'Municipality' suffix and 'Gemeente' prefix from NL_full gementee names."""
            name = re.sub(r'\s+Municipalit[a-z]+$', '', name.strip(), flags=re.IGNORECASE)
            name = re.sub(r'^Gemeente\s+', '', name, flags=re.IGNORECASE)
            return name.rstrip('.').strip()

        _gebieden_rows = []
        with open('data/Gebieden_in_Nederland_2023.csv') as f:
            for _row in csv.DictReader(f, delimiter=';'):
                _region = _row['Region'].strip()
                _code   = _row['gementee_code'].strip()
                if not _region or not _code:
                    continue
                if 'GM0518' in _region:  # malformed 's-Gravenhage row
                    _gebieden_rows.append({"region": "'s-Gravenhage (gemeente)", "region_norm": "'s-Gravenhage", "gementee_code": "GM0518"})
                elif 'GM0796' in _region:  # malformed 's-Hertogenbosch row
                    _gebieden_rows.append({"region": "'s-Hertogenbosch", "region_norm": "'s-Hertogenbosch", "gementee_code": "GM0796"})
                else:
                    _gebieden_rows.append({"region": _region, "region_norm": _norm_gebieden(_region), "gementee_code": _code})

        _df_gebieden = pd.DataFrame(_gebieden_rows)
        con.register('_df_gebieden', _df_gebieden)
        con.execute("CREATE OR REPLACE TABLE gebieden AS SELECT * FROM _df_gebieden")

        # Entries that name normalization alone can't resolve.
        _overrides = {
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

        # Source name -> canonical display name.
        _renames = {
            'Hoogland Municipality': 'Amersfoort Municipality',
        }

        _gementees = con.execute(
            "SELECT DISTINCT gementee FROM postcodes WHERE gementee IS NOT NULL"
        ).fetchall()

        _lookup_rows = []
        for (_g,) in _gementees:
            _g = _g.strip()
            _display = _renames.get(_g, _g)
            if _g in _overrides:
                _lookup_rows.append({"gementee": _g, "gementee_display": _display, "gementee_code": _overrides[_g]})
            else:
                _norm = _norm_nl(_g)
                _result = con.execute(
                    "SELECT gementee_code FROM gebieden WHERE region_norm = ?", [_norm]
                ).fetchone()
                _lookup_rows.append({"gementee": _g, "gementee_display": _display, "gementee_code": _result[0] if _result else None})

        _df_lookup = pd.DataFrame(_lookup_rows)
        con.register('_df_lookup', _df_lookup)
        return con.execute("CREATE OR REPLACE TABLE gementee_lookup AS SELECT * FROM _df_lookup")


    _()
    return


@app.cell
def _(con):
    con.execute("""
        DELETE FROM claims
        WHERE gender IS NULL
           OR LOWER(TRIM(gender)) = '(null)'
    """)
    return


@app.cell
def _(con):
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
                SUM(number_of_insured)                                          AS number_of_insured,
                ROUND(SUM(insured_years),                          2)          AS insured_years,
                ROUND(SUM(costs_medical_specialist_care),          2)          AS costs_medical_specialist_care,
                ROUND(SUM(costs_pharmacy),                         2)          AS costs_pharmacy,
                ROUND(SUM(costs_gp_registration_fee),              2)          AS costs_gp_registration_fee,
                ROUND(SUM(costs_gp_consultation),                  2)          AS costs_gp_consultation,
                ROUND(SUM(costs_gp_multidisciplinary_care),        2)          AS costs_gp_multidisciplinary_care,
                ROUND(SUM(costs_gp_other),                         2)          AS costs_gp_other,
                ROUND(SUM(costs_medical_aids),                     2)          AS costs_medical_aids,
                ROUND(SUM(costs_dental_care),                      2)          AS costs_dental_care,
                ROUND(SUM(costs_physiotherapy),                    2)          AS costs_physiotherapy,
                ROUND(SUM(costs_paramedical_other),                2)          AS costs_paramedical_other,
                ROUND(SUM(costs_patient_transport_seated),         2)          AS costs_patient_transport_seated,
                ROUND(SUM(costs_patient_transport_ambulance),      2)          AS costs_patient_transport_ambulance,
                ROUND(SUM(costs_maternity_care),                   2)          AS costs_maternity_care,
                ROUND(SUM(costs_obstetric_care),                   2)          AS costs_obstetric_care,
                ROUND(SUM(costs_cross_border_care),                2)          AS costs_cross_border_care,
                ROUND(SUM(costs_primary_care_support),             2)          AS costs_primary_care_support,
                ROUND(SUM(costs_geriatric_rehabilitation),         2)          AS costs_geriatric_rehabilitation,
                ROUND(SUM(costs_primary_care_residence),           2)          AS costs_primary_care_residence,
                ROUND(SUM(costs_nursing_care),                     2)          AS costs_nursing_care,
                ROUND(SUM(costs_medical_care_specific_patients),   2)          AS costs_medical_care_specific_patients,
                ROUND(SUM(costs_integrated_birth_care),            2)          AS costs_integrated_birth_care,
                ROUND(SUM(costs_mental_health_innovation),         2)          AS costs_mental_health_innovation,
                ROUND(SUM(costs_other),                            2)          AS costs_other,
                ROUND(SUM(costs_mental_health_consultations),      2)          AS costs_mental_health_consultations,
                ROUND(SUM(costs_inpatient_mental_health),          2)          AS costs_inpatient_mental_health,
                ROUND(SUM(costs_other_mental_health_services),     2)          AS costs_other_mental_health_services,
                ROUND(SUM(costs_mental_health_total),              2)          AS costs_mental_health_total
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
        -- inflation-adjusted costs in 2020 euros: nominal / (cpi_2020_base / 100)
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
    return


@app.cell
def _(con):
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
    return


@app.cell
def _(claims, con, mo):
    _df = mo.sql(
        """
        SELECT 
        year,
        sum(costs_primary_care_support_real),
        SUM(costs_primary_care_residence_real)
        FROM claims
        group by 1

        """,
        engine=con
    )
    return


if __name__ == "__main__":
    app.run()
