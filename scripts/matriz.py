"""Matriz del modelo: las 137 variables base + los bloques nuevos (fuente del off-block, ADS-B, puerta/stand).

`cargar(anio, bloques, sufijo_fuentes)` devuelve (d, X, ref, cats):
- d: tabla combinada del año (target, mes, aeropuerto, aptitud...), X: variables en orden fijo, ref: referencia
  del residuo (NM o P10), cats: categóricas.
- bloques de v30: ["fuentes", "adsb", "adsb_stand"] -> 177 variables. `sufijo_fuentes` = "" (modelo final) o
  "_hist" (partición histórica: codificaciones con los meses de entrenamiento).
- opciones de experimentos (no las usa v30): "sincoord", "pushsched", "adsb_otros".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import periodo  # noqa: E402
from prc.base import matriz as Mo  # noqa: E402

FEAT = periodo.FEAT
BLOQUES_V30 = ["fuentes", "adsb", "adsb_stand"]

FUENTES = ["share_other", "n_other", "prev_acpy", "prev_same_pref", "prev_gap_sched", "prev_adelay",
           "arr_copy_pref_6h", "arr_n_pref_6h", "arr_copy_apt_2h", "fl_rep", "fl_nostd", "fl_len",
           "te_copy_fl", "te_n_fl", "te_copy_pref", "cross_day_local"]


def cargar(anio: int, bloques: list[str], sufijo_fuentes: str):
    r = Mo.receta()
    refs = Mo.leer_referencias()
    d, X, ref = Mo.matriz(anio, refs, r)
    cats = list(r["variables"]["categoricas"])
    if "fuentes" in bloques:
        f = pd.read_parquet(FEAT / f"fuentes_{anio}{sufijo_fuentes}.parquet")
        f["MVT_ID_mvt"] = f["MVT_ID_mvt"].astype("int64")
        q = d[["MVT_ID_mvt"]].merge(f, on="MVT_ID_mvt", how="left", validate="one_to_one")
        stale = q["prev_gap_takeoff"].to_numpy(float) > 86400
        for c in ["prev_acpy", "prev_same_pref", "prev_gap_sched", "prev_adelay"]:
            v = q[c].to_numpy(float).copy()
            v[stale] = np.nan
            q[c] = v
        q.loc[q.arr_n_pref_6h < 0, "arr_n_pref_6h"] = 0
        for c in FUENTES:
            X[c] = q[c].to_numpy(dtype="float32")
    if "adsb" in bloques:
        a = pd.read_parquet(FEAT / f"adsb_{anio}.parquet")
        a["MVT_ID_mvt"] = a["MVT_ID_mvt"].astype("int64")
        cols = [c for c in a.columns if c.startswith("adsb_") and c not in ("adsb_source",)]
        if "sincoord" in bloques:   # sin coordenadas absolutas del primer punto (quedan las distancias a puerta/stand)
            cols = [c for c in cols if c not in ("adsb_lat0", "adsb_lon0")]
        q = d[["MVT_ID_mvt"]].merge(a[["MVT_ID_mvt", *cols]], on="MVT_ID_mvt", how="left", validate="one_to_one")
        for c in cols:
            X[c] = q[c].to_numpy(dtype="float32")
        # diferencias con el rodaje NM/P10: lo observado frente a la referencia del residuo
        for c in ("adsb_first_seen", "adsb_first_move", "adsb_last_still"):
            X[c + "_menos_ref"] = (q[c].to_numpy(float) - ref).astype("float32")
        if "pushsched" in bloques:
            # pushback observado respecto a la hora programada: (despegue − programada) − rodaje observado
            sig = d.despegue_menos_sched_sec.to_numpy(float)
            for c in ("adsb_last_still", "adsb_first_move"):
                X[c.replace("adsb_", "adsb_push_") + "_menos_sched"] = (sig - q[c].to_numpy(float)).astype("float32")
        X = X.copy()
    if "adsb_stand" in bloques:
        # distancia del primer punto ADS-B a la puerta (aparcados > 15 min) y a la posición laxa (parados > 60 s)
        for fichero, sufijo in ((f"adsb_stand_{anio}.parquet", "gate"), (f"adsb_stand60_{anio}.parquet", "stand")):
            a = pd.read_parquet(FEAT / fichero)
            a["MVT_ID_mvt"] = a["MVT_ID_mvt"].astype("int64")
            q = d[["MVT_ID_mvt"]].merge(a, on="MVT_ID_mvt", how="left", validate="one_to_one")
            X[f"adsb_d_{sufijo}_m"] = q["adsb_d_stand_m"].to_numpy(dtype="float32")
            X[f"{sufijo}_pos_n"] = q["stand_pos_n"].to_numpy(dtype="float32")
    if "adsb_otros" in bloques:
        # solo otros aviones: medias de vecinos de pista/aeropuerto, sin conteos (dependen de la cobertura)
        a = pd.read_parquet(FEAT / f"adsb_otros_{anio}.parquet")
        a["MVT_ID_mvt"] = a["MVT_ID_mvt"].astype("int64")
        cols = [c for c in a.columns if c.startswith("oth_") and not c.startswith("oth_n_")]
        q = d[["MVT_ID_mvt"]].merge(a[["MVT_ID_mvt", *cols]], on="MVT_ID_mvt", how="left", validate="one_to_one")
        for c in cols:
            X[c] = q[c].to_numpy(dtype="float32")
        for c in ("oth_queue600_pista_b20", "oth_first_seen_pista_pm30", "oth_last_still_pista_pm30"):
            X[c + "_menos_ref"] = (q[c].to_numpy(float) - ref).astype("float32")
        X = X.copy()
    return d, X, ref, cats
