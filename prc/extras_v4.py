"""Features del experimento v4: cola de salidas, vecinos, ritmo de pista, ocupación del stand y recuperación por nº de vuelo.

Todas se calculan igual para 2025 y 2026 y solo usan información que existe en `ranking.parquet`: horas de despegue,
`AOBT_3` de los vuelos (el off-block del Network Manager, que sobrevive en 2026), llegadas completas y la hora
programada. Ninguna mira el target.

Hoy lo usa `prc/base/tablas.py` para construir la tabla `extras_v4`.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from prc import config as C

# Grupos de features, para poder encenderlos y apagarlos por separado en los experimentos.
COLA = ["n_rodando_apt", "despegues_mientras_apt", "empujes_15_despues", "n_rodando_pista",
        "despegues_mientras_pista", "llegadas_mientras_rodamos"]
VECINOS = ["nm_vecinos_apt_30", "nm_vecinos_pista_30", "nm_dia_apt"]
RITMO = ["dep_apt_30_antes", "dep_apt_30_despues", "dep_pista_30_antes", "dep_pista_30_despues",
         "separacion_media_pista_30"]
STAND = ["llegadas_a_mi_stand", "taxi_min_por_stand", "taxi_max_por_stand"]
NUMERICAS = COLA + VECINOS + RITMO + STAND
CATEGORICAS = ["operador_rec", "apt_operador_rec", "segmento_rec", "estela_rec"]
TODAS = NUMERICAS + CATEGORICAS


def _cola(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Cola de salidas alrededor del vuelo, usando `AOBT_3` como off-block.

    `n_rodando`: aviones que ya han empujado y aún no han despegado cuando empujamos nosotros (la N(t) de la
    literatura de taxi-out). `despegues_mientras`: cuántos despegan mientras rodamos. `empujes_15_despues`: demanda
    que viene detrás. Nulas si el vuelo no tiene `AOBT_3` (1,5 % en 2026).
    """
    d = dep.sort_values("t_off").reset_index(drop=True)
    taxi_nm = d["t_off"] - d["t_aobt"]
    d.loc[~taxi_nm.between(0, 7200), "t_aobt"] = np.nan   # off-block incoherente: como si no lo hubiera
    out = pd.DataFrame(index=d.index)
    for etiqueta, clave in (("apt", ["airport"]), ("pista", ["airport", "RUNWAY_mvt"])):
        n_rodando = np.full(len(d), np.nan)
        despegues_mientras = np.full(len(d), np.nan)
        empujes_despues = np.full(len(d), np.nan)
        for _, g in d.groupby(clave, dropna=True):
            i = g.index.to_numpy()
            a, t = g["t_aobt"].to_numpy(), g["t_off"].to_numpy()
            hay = ~np.isnan(a)
            a_ord, t_ord, t_con_aobt = np.sort(a[hay]), np.sort(t), np.sort(t[hay])
            n_rodando[i[hay]] = (np.searchsorted(a_ord, a[hay], "right")
                                 - np.searchsorted(t_con_aobt, a[hay], "right") - 1)
            despegues_mientras[i[hay]] = (np.searchsorted(t_ord, t[hay], "right")
                                          - np.searchsorted(t_ord, a[hay], "right") - 1)
            empujes_despues[i[hay]] = (np.searchsorted(a_ord, a[hay] + 900, "right")
                                       - np.searchsorted(a_ord, a[hay], "right"))
        out[f"n_rodando_{etiqueta}"] = n_rodando
        out[f"despegues_mientras_{etiqueta}"] = despegues_mientras
        if etiqueta == "apt":
            out["empujes_15_despues"] = empujes_despues
    llegadas = np.full(len(d), np.nan)
    aterrizajes = arr.dropna(subset=["t_land"]).groupby("airport")["t_land"].apply(lambda s: np.sort(s.to_numpy()))
    for apt, g in d.groupby("airport"):
        if apt in aterrizajes.index:
            i, a, t = g.index.to_numpy(), g["t_aobt"].to_numpy(), g["t_off"].to_numpy()
            hay = ~np.isnan(a)
            l = aterrizajes.loc[apt]
            llegadas[i[hay]] = np.searchsorted(l, t[hay], "right") - np.searchsorted(l, a[hay], "right")
    out["llegadas_mientras_rodamos"] = llegadas
    out[C.ID] = d[C.ID].to_numpy()
    return out


def _vecinos(dep_file: Path) -> pd.DataFrame:
    """Taxi-out según `AOBT_3` de los demás vuelos: mediana en ±30 min (aeropuerto y pista) y mediana del día."""
    return duckdb.sql(f"""
        WITH d AS (SELECT MVT_ID_mvt, airport, mes_utc, fecha_local, RUNWAY_mvt, epoch(MVT_TIME_UTC_mvt) AS t,
                          CASE WHEN epoch(MVT_TIME_UTC_mvt - AOBT_3_flt) BETWEEN 0 AND 7200
                               THEN epoch(MVT_TIME_UTC_mvt - AOBT_3_flt) END AS nm
                   FROM '{Path(dep_file).as_posix()}')
        SELECT MVT_ID_mvt,
               median(nm) OVER (PARTITION BY airport, mes_utc ORDER BY t
                    RANGE BETWEEN 1800 PRECEDING AND 1800 FOLLOWING EXCLUDE CURRENT ROW) AS nm_vecinos_apt_30,
               median(nm) OVER (PARTITION BY airport, mes_utc, RUNWAY_mvt ORDER BY t
                    RANGE BETWEEN 1800 PRECEDING AND 1800 FOLLOWING EXCLUDE CURRENT ROW) AS nm_vecinos_pista_30,
               median(nm) OVER (PARTITION BY airport, fecha_local) AS nm_dia_apt
        FROM d""").df()


def _ritmo(dep_file: Path) -> pd.DataFrame:
    """Despegues en los 30 min de antes y de después, por separado, y separación media en la pista."""
    antes = "RANGE BETWEEN 1800 PRECEDING AND CURRENT ROW"
    despues = "RANGE BETWEEN CURRENT ROW AND 1800 FOLLOWING"
    return duckdb.sql(f"""
        WITH d AS (SELECT MVT_ID_mvt, airport, mes_utc, RUNWAY_mvt, epoch(MVT_TIME_UTC_mvt) AS t
                   FROM '{Path(dep_file).as_posix()}')
        SELECT MVT_ID_mvt,
               count(*) OVER (PARTITION BY airport, mes_utc ORDER BY t {antes}) - 1 AS dep_apt_30_antes,
               count(*) OVER (PARTITION BY airport, mes_utc ORDER BY t {despues}) - 1 AS dep_apt_30_despues,
               count(*) OVER (PARTITION BY airport, mes_utc, RUNWAY_mvt ORDER BY t {antes}) - 1 AS dep_pista_30_antes,
               count(*) OVER (PARTITION BY airport, mes_utc, RUNWAY_mvt ORDER BY t {despues}) - 1 AS dep_pista_30_despues,
               1800.0 / nullif(count(*) OVER (PARTITION BY airport, mes_utc, RUNWAY_mvt ORDER BY t {antes}) - 1, 0)
                   AS separacion_media_pista_30
        FROM d""").df()


def _stand(dep_file: Path, arr_file: Path) -> pd.DataFrame:
    """Cotas del taxi-out según quién ocupa el stand, a partir de las llegadas (intactas en 2026).

    `taxi_min_por_stand`: si otro avión aparcó en mi stand antes de mi despegue, dejé el stand antes de esa hora.
    `taxi_max_por_stand`: la última llegada a mi stand antes de mi hora programada suele ser mi propio avión.
    No son cotas exactas (hay stands con varias posiciones), pero separan bien los casos raros.
    """
    return duckdb.sql(f"""
        WITH d AS (SELECT MVT_ID_mvt, airport, STAND_mvt, SCHED_TIME_UTC_mvt AS sched, MVT_TIME_UTC_mvt AS despegue
                   FROM '{Path(dep_file).as_posix()}'),
             a AS (SELECT airport, STAND_mvt, BLOCK_TIME_UTC_mvt AS inblock FROM '{Path(arr_file).as_posix()}'
                   WHERE STAND_mvt IS NOT NULL AND BLOCK_TIME_UTC_mvt IS NOT NULL),
             despues AS (SELECT d.MVT_ID_mvt, count(*) AS llegadas_a_mi_stand,
                                epoch(any_value(d.despegue) - min(a.inblock)) AS taxi_min_por_stand
                         FROM d JOIN a USING (airport, STAND_mvt)
                         WHERE a.inblock > d.sched + INTERVAL 5 MINUTE AND a.inblock < d.despegue GROUP BY 1),
             antes AS (SELECT d.MVT_ID_mvt, epoch(d.despegue - a.inblock) AS taxi_max_por_stand
                       FROM d ASOF LEFT JOIN a ON d.airport = a.airport AND d.STAND_mvt = a.STAND_mvt
                                              AND d.sched >= a.inblock)
        SELECT d.MVT_ID_mvt, coalesce(despues.llegadas_a_mi_stand, 0) AS llegadas_a_mi_stand,
               despues.taxi_min_por_stand,
               CASE WHEN antes.taxi_max_por_stand BETWEEN 0 AND 86400 THEN antes.taxi_max_por_stand END AS taxi_max_por_stand
        FROM d LEFT JOIN despues USING (MVT_ID_mvt) LEFT JOIN antes USING (MVT_ID_mvt)""").df()


def _recuperadas(dep_file: Path, referencia_file: Path) -> pd.DataFrame:
    """Operador, segmento y estela de las salidas sin plan de vuelo, por (aeropuerto, nº de vuelo) de otros días."""
    return duckdb.sql(f"""
        WITH ref AS (SELECT airport, FLIGHT_mvt, mode(AIRCRAFT_OPERATOR_flt) AS op, mode(MARKET_SEGMENT_flt) AS seg,
                            mode(WK_TBL_CAT_flt) AS wk
                     FROM '{Path(referencia_file).as_posix()}'
                     WHERE FLIGHT_ID_mvt IS NOT NULL AND FLIGHT_mvt IS NOT NULL GROUP BY 1, 2)
        SELECT d.MVT_ID_mvt,
               coalesce(d.AIRCRAFT_OPERATOR_flt, ref.op) AS operador_rec,
               d.airport || '|' || coalesce(d.AIRCRAFT_OPERATOR_flt, ref.op) AS apt_operador_rec,
               coalesce(d.MARKET_SEGMENT_flt, ref.seg) AS segmento_rec,
               coalesce(d.WK_TBL_CAT_flt, ref.wk) AS estela_rec
        FROM '{Path(dep_file).as_posix()}' d LEFT JOIN ref USING (airport, FLIGHT_mvt)""").df()


def construir(dep_file: Path, arr_file: Path, referencia_file: Path | None = None) -> pd.DataFrame:
    """Tabla de features extra por `MVT_ID_mvt` para un año. `referencia_file` es de dónde salen los nº de vuelo."""
    dep = duckdb.sql(f"""SELECT MVT_ID_mvt, airport, RUNWAY_mvt, epoch(MVT_TIME_UTC_mvt) AS t_off,
                         epoch(AOBT_3_flt) AS t_aobt FROM '{Path(dep_file).as_posix()}'""").df()
    arr = duckdb.sql(f"SELECT airport, epoch(MVT_TIME_UTC_mvt) AS t_land FROM '{Path(arr_file).as_posix()}'").df()
    return (_cola(dep, arr)
            .merge(_vecinos(dep_file), on=C.ID, validate="one_to_one")
            .merge(_ritmo(dep_file), on=C.ID, validate="one_to_one")
            .merge(_stand(dep_file, arr_file), on=C.ID, validate="one_to_one")
            .merge(_recuperadas(dep_file, referencia_file or C.DEP_2025_FILE), on=C.ID, validate="one_to_one"))
