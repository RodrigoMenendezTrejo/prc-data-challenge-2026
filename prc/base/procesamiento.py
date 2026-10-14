"""Ficheros oficiales -> salidas y llegadas limpias por año (`Data/interim/`).

Es el SQL del cuaderno original `notebooks/01_procesamiento.ipynb`, sin cambios:
tokens de nulo, aeropuerto observado, hora local IANA, dique, banderas de calidad y `apto_entrenamiento`.
"""
import pandas as pd

from prc import config as C
from prc import data

from . import config as K

TAXI_MIN_SEC = 60                     # por debajo de un minuto el taxi-out es físicamente imposible
TAXI_MAX_SEC = 7200                   # criterio ATXOT de EUROCONTROL
TOLERANCIA_INCONSISTENCIA_SEC = 60    # |TAXITIME − (despegue − off-block)| por encima de esto: inconsistente

COLUMNAS_TEXTO = [
    "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt", "STAND_mvt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
]
TOKENS = ", ".join(f"'{t}'" for t in C.TOKENS_NULOS)
SQL_TEXTO = ",\n".join(
    f"CASE WHEN trim({c}) IN ({TOKENS}) THEN NULL ELSE trim({c}) END AS {c}" for c in COLUMNAS_TEXTO)

SQL_DERIVADAS = f"""
    strftime(m.MVT_TIME_UTC_mvt, '%Y-%m')                   AS mes_utc,
    timezone(a.tz, m.MVT_TIME_UTC_mvt)                      AS mvt_time_local,
    CAST(timezone(a.tz, m.MVT_TIME_UTC_mvt) AS DATE)        AS fecha_local,
    hour(timezone(a.tz, m.MVT_TIME_UTC_mvt))                AS hora_local,
    isodow(timezone(a.tz, m.MVT_TIME_UTC_mvt))              AS dia_semana_local,   -- 1 = lunes
    {C.DIQUE_SQL}                                           AS dique,
    date_diff('second', m.BLOCK_TIME_UTC_mvt, m.MVT_TIME_UTC_mvt) AS taxi_por_resta_sec
"""

SQL_BANDERAS = f"""
    TAXITIME_SEC_mvt IS NULL                                        AS f_sin_target,
    coalesce(TAXITIME_SEC_mvt <= 0, false)                          AS f_taxi_no_positivo,
    coalesce(TAXITIME_SEC_mvt < {TAXI_MIN_SEC}, false)              AS f_taxi_corto,
    coalesce(TAXITIME_SEC_mvt > {TAXI_MAX_SEC}, false)              AS f_taxi_largo,
    coalesce(abs(TAXITIME_SEC_mvt - taxi_por_resta_sec) > {TOLERANCIA_INCONSISTENCIA_SEC}, false)
                                                                    AS f_taxi_inconsistente,
    STAND_mvt IS NULL                                               AS f_sin_stand,
    RUNWAY_mvt IS NULL                                              AS f_sin_pista,
    BLOCK_TIME_UTC_mvt IS NULL                                      AS f_sin_block,
    FLIGHT_ID_mvt IS NULL                                           AS f_sin_flight_id,
    AOBT_3_flt IS NULL                                              AS f_sin_aobt3,
    coalesce(ADEP_flt <> ADEP_mvt, false)                           AS f_adep_distinto,
    coalesce(AIRCRAFT_TYPE_flt <> AIRCRAFT_TYPE_mvt, false)         AS f_tipo_avion_distinto
"""
BANDERAS = [linea.split(" AS ")[-1].strip() for linea in SQL_BANDERAS.split(",\n") if " AS " in linea]


def conectar():
    """DuckDB en UTC con las vistas `training`, `ranking` y `submitting` sobre `data/raw/`."""
    return data.connect()


def construir_movimientos(con) -> None:
    """Crea la tabla `movimientos`: training (2025) y ranking (2026) limpios, con derivadas y banderas."""
    aeropuertos = pd.DataFrame(
        [(icao, iata, tz) for icao, (iata, _, tz) in C.AIRPORTS.items()], columns=["airport", "iata", "tz"])
    con.register("aeropuertos", aeropuertos)
    con.sql(f"""
    CREATE OR REPLACE TABLE movimientos AS
    WITH crudo AS (
        SELECT 2025 AS dataset, * FROM training
        UNION ALL
        SELECT 2026 AS dataset, * FROM ranking
    ),
    limpio AS (
        SELECT dataset,
               CAST(MVT_ID_mvt    AS BIGINT) AS MVT_ID_mvt,
               CAST(FLIGHT_ID_mvt AS BIGINT) AS FLIGHT_ID_mvt,
               {SQL_TEXTO},
               MVT_TIME_UTC_mvt, BLOCK_TIME_UTC_mvt, SCHED_TIME_UTC_mvt, TAXITIME_SEC_mvt,
               LOBT_flt, IOBT_flt, EOBT_1_flt, ARVT_1_flt, AOBT_3_flt, ARVT_3_flt
        FROM crudo
    ),
    con_aeropuerto AS (
        SELECT *, {C.AIRPORT_SQL} AS airport FROM limpio
    ),
    derivado AS (
        SELECT m.*, a.tz, {SQL_DERIVADAS}
        FROM con_aeropuerto m
        LEFT JOIN aeropuertos a USING (airport)
    ),
    banderas AS (
        SELECT *, {SQL_BANDERAS} FROM derivado
    )
    SELECT *,
           NOT (f_sin_target OR f_taxi_no_positivo OR f_taxi_corto OR f_taxi_largo) AS apto_entrenamiento
    FROM banderas
    """)


def informe_calidad(con, por_aeropuerto: bool) -> pd.DataFrame:
    """Porcentaje de cada bandera en las salidas, por año y opcionalmente por aeropuerto."""
    pct = ",\n".join(f"round(100.0 * avg({b}::INT), 2) AS {b}" for b in BANDERAS)
    grupo = "dataset, airport" if por_aeropuerto else "dataset"
    return con.sql(f"""
        SELECT {grupo},
               count(*) AS salidas,
               {pct},
               round(100.0 * avg(apto_entrenamiento::INT), 2)              AS apto_entrenamiento,
               round(100.0 * avg((second(MVT_TIME_UTC_mvt) = 0)::INT), 1)   AS pct_despegue_segundo_0,
               round(100.0 * avg((second(BLOCK_TIME_UTC_mvt) = 0)::INT), 1) AS pct_offblock_segundo_0
        FROM movimientos
        WHERE PHASE_mvt = 'DEP'
        GROUP BY {grupo}
        ORDER BY {grupo}
    """).df()


def escribir_interim(con) -> dict:
    """Escribe dep/arr de 2025 y 2026 en `Data/interim/`, ordenados por aeropuerto y hora.

    En el periodo final solo se escribe 2026 si 2025 ya existe (2025 es el mismo en los dos periodos)."""
    K.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    salidas = {}
    for anio in K.ANIOS:
        for fase in ("dep", "arr"):
            path = K.interim(fase, anio)
            if K.FINAL and anio == 2025 and path.exists():
                continue
            con.sql(f"""
                COPY (
                    SELECT * EXCLUDE (dataset)
                    FROM movimientos
                    WHERE dataset = {anio} AND PHASE_mvt = '{fase.upper()}'
                    ORDER BY airport, MVT_TIME_UTC_mvt
                ) TO '{path.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
            """)
            salidas[path.name] = path
    return salidas


def comprobar_interim(con) -> pd.DataFrame:
    """Filas, IDs únicos, husos y cuadre con los ficheros oficiales. Falla si algo no cuadra."""
    def contar(path):
        return con.sql(f"""
            SELECT count(*) AS filas, count(DISTINCT MVT_ID_mvt) AS ids_distintos,
                   count(*) FILTER (WHERE tz IS NULL) AS sin_huso
            FROM read_parquet('{path.as_posix()}')""").df().iloc[0]
    paths = [K.interim(f, a) for a in K.ANIOS for f in ("dep", "arr")]
    salida = pd.DataFrame({p.name: contar(p) for p in paths}).T
    salida["MB"] = [round(p.stat().st_size / 1e6, 1) for p in paths]
    n_training = con.sql("SELECT count(*) FROM training").fetchone()[0]
    n_ranking = con.sql("SELECT count(*) FROM ranking").fetchone()[0]
    assert (salida["filas"] == salida["ids_distintos"]).all(), "MVT_ID duplicados"
    assert (salida["sin_huso"] == 0).all(), "movimientos sin aeropuerto conocido"
    assert salida.loc["dep_2026.parquet", "filas"] == K.N_FILAS_ENVIO, "dep_2026 no tiene las filas del envío"
    assert salida.loc["dep_2025.parquet", "filas"] + salida.loc["arr_2025.parquet", "filas"] == n_training
    assert salida.loc["dep_2026.parquet", "filas"] + salida.loc["arr_2026.parquet", "filas"] == n_ranking
    return salida
