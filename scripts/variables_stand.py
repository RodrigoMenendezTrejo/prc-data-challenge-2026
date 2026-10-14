"""Posición de cada stand estimada con ADS-B y distancia al primer punto observado de cada salida.

La posición de cada (aeropuerto, stand) es la mediana del primer punto (adsb_lat0/lon0) de las salidas que se ven
paradas allí antes de moverse, con 2025 y 2026 (sin etiquetas). Se calcula dos veces: puerta (paradas más de 15 min)
y posición laxa (más de 60 s). Para cada una:

- distancia del primer punto observado a esa posición: casi 0 si la traza empieza en el stand, grande si empieza
  a mitad del rodaje;
- número de salidas con que se situó (fiabilidad).

Uso: python scripts/variables_stand.py  -> Data/features/adsb_stand_{2025,2026}.parquet (puerta, > 15 min)
     y Data/features/adsb_stand60_{2025,2026}.parquet (posición laxa, > 60 s)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import periodo  # noqa: E402

FEAT = periodo.FEAT


def salidas(anio):
    src = ROOT / "Data" / "raw" / "training_*.parquet" if anio == 2025 else periodo.RANKING
    import duckdb
    return duckdb.sql(f"""select MVT_ID_mvt, ADEP_mvt airport, STAND_mvt stand from '{src.as_posix()}' where PHASE_mvt = 'DEP'""").df()


def construir(tablas, umbral_s: int, prefijo: str):
    todo = pd.concat(tablas.values())
    ok = todo.adsb_stat_obs.gt(umbral_s) & todo.stand.notna() & todo.adsb_lat0.notna()
    pos = (todo[ok].groupby(["airport", "stand"])
           .agg(st_lat=("adsb_lat0", "median"), st_lon=("adsb_lon0", "median"), stand_pos_n=("adsb_lat0", "size"))
           .reset_index())
    pos = pos[pos.stand_pos_n >= 5]
    print(prefijo, "stands situados:", len(pos), pos.groupby("airport").size().to_dict())
    for anio, t in tablas.items():
        q = t.merge(pos, on=["airport", "stand"], how="left")
        dx = (q.adsb_lon0 - q.st_lon) * 111320 * np.cos(np.radians(q.st_lat))
        dy = (q.adsb_lat0 - q.st_lat) * 110540
        q["adsb_d_stand_m"] = np.hypot(dx, dy)
        q[["MVT_ID_mvt", "adsb_d_stand_m", "stand_pos_n"]].to_parquet(FEAT / f"{prefijo}_{anio}.parquet", index=False)
        print(" ", anio, "con distancia:", round(q.adsb_d_stand_m.notna().mean(), 3), "mediana m:", round(q.adsb_d_stand_m.median(), 1))


def main():
    tablas = {}
    for anio in (2025, 2026):
        a = pd.read_parquet(FEAT / f"adsb_{anio}.parquet", columns=["MVT_ID_mvt", "adsb_lat0", "adsb_lon0", "adsb_stat_obs"])
        tablas[anio] = salidas(anio).merge(a, on="MVT_ID_mvt", how="left")
    construir(tablas, 900, "adsb_stand")     # puerta: aviones vistos aparcados > 15 min
    construir(tablas, 60, "adsb_stand60")    # posición laxa: parados > 60 s


if __name__ == "__main__":
    main()
