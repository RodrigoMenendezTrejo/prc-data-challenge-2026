"""El taxi-IN de las llegadas como medida directa de congestión de superficie.

Las llegadas traen `BLOCK_TIME − aterrizaje`: cuánto tarda un avión en recorrer la superficie en sentido contrario.
Es el mismo aeropuerto, las mismas calles de rodaje y la misma hora que la salida que estamos prediciendo, así que
mide la congestión sin depender de ningún dato de la propia salida. Existe igual en 2026, donde las llegadas están
intactas, y no mira el target de ninguna salida.

Lo usó el experimento v9; la tubería de v30 no lo importa.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from prc import config as C

TODAS = ["taxi_in_vecinos_30", "taxi_in_p90_30", "taxi_in_dia", "llegadas_rodando"]
MAX_TAXI_IN_SEC = 7200


def construir(dep_file: Path, arr_file: Path) -> pd.DataFrame:
    """Por cada salida: taxi-in típico de las llegadas de alrededor y cuántas están rodando cuando empujamos."""
    llegadas = duckdb.sql(f"""
        SELECT airport, epoch(MVT_TIME_UTC_mvt) AS t_land, epoch(BLOCK_TIME_UTC_mvt) AS t_block,
               epoch(BLOCK_TIME_UTC_mvt - MVT_TIME_UTC_mvt) AS taxi_in
        FROM '{Path(arr_file).as_posix()}'
        WHERE BLOCK_TIME_UTC_mvt IS NOT NULL AND MVT_TIME_UTC_mvt IS NOT NULL
          AND epoch(BLOCK_TIME_UTC_mvt - MVT_TIME_UTC_mvt) BETWEEN 0 AND {MAX_TAXI_IN_SEC}""").df()
    salidas = duckdb.sql(f"""
        SELECT MVT_ID_mvt, airport, epoch(MVT_TIME_UTC_mvt) AS t_off, epoch(AOBT_3_flt) AS t_aobt
        FROM '{Path(dep_file).as_posix()}'""").df()

    ventana = duckdb.sql("""
        SELECT s.MVT_ID_mvt,
               median(l.taxi_in) AS taxi_in_vecinos_30,
               quantile_cont(l.taxi_in, 0.9) AS taxi_in_p90_30
        FROM salidas s JOIN llegadas l
          ON l.airport = s.airport AND l.t_land BETWEEN s.t_off - 1800 AND s.t_off + 1800
        GROUP BY 1""").df()
    dia = duckdb.sql("""
        SELECT s.MVT_ID_mvt, median(l.taxi_in) AS taxi_in_dia
        FROM salidas s JOIN llegadas l
          ON l.airport = s.airport AND l.t_land BETWEEN s.t_off - 43200 AND s.t_off + 43200
        GROUP BY 1""").df()

    # Llegadas que ya han aterrizado pero aún no han llegado al stand cuando empujamos: la N(t) de las entradas.
    rodando = np.full(len(salidas), np.nan)
    for apt, g in salidas.groupby("airport"):
        li = llegadas[llegadas["airport"] == apt]
        if li.empty:
            continue
        land, block = np.sort(li["t_land"].to_numpy()), np.sort(li["t_block"].to_numpy())
        a = g["t_aobt"].to_numpy()
        hay = ~np.isnan(a)
        rodando[g.index.to_numpy()[hay]] = (np.searchsorted(land, a[hay], "right")
                                            - np.searchsorted(block, a[hay], "right"))
    salidas["llegadas_rodando"] = rodando
    return (salidas[[C.ID, "llegadas_rodando"]]
            .merge(ventana, on=C.ID, how="left", validate="one_to_one")
            .merge(dia, on=C.ID, how="left", validate="one_to_one"))
