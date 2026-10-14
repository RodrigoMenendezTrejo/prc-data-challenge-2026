"""Post-proceso de la predicción del cuerpo: calibración de fuentes y proyección LOBT.

1. Regla de Roma: en salidas de LIRF sin FLIGHT_ID y con despegue más de 2 h después de la hora programada, el
   off-block publicado suele ser la hora programada, así que se mezcla `p·(despegue − programada) + (1 − p)·cuerpo`,
   con `p` por banda de retraso (2–3 h, 3–6 h, > 6 h) y prefijo de vuelo, regularizada hacia la de la banda.
2. Consenso: si las cuatro fuentes (NM, horario, EOBT, LOBT) dicen más de 2 h, se mezcla con la menor.
3. Proyección LOBT: se recorta al intervalo (despegue − LOBT) ± 3606 s y al suelo de 60 s.

Origen: experimentos/arquitectura_simple_20261001/sources.py (ajuste y aplicación) y
experimentos/mejora_v21_20261006/common.py (proyección).
"""
import numpy as np
import pandas as pd

from prc import referencias as M

from . import config as K


def banda(df):
    return np.minimum(np.searchsorted([10800, 21600], df.despegue_menos_sched_sec.to_numpy(), side="right"), 2)


def seleccion(df):
    return ((df.airport == "LIRF").to_numpy() & df.sin_flight_id.to_numpy(bool)
            & (df.despegue_menos_sched_sec.to_numpy() > 7200))


def ajustar(df, prior_count=20, frequency=False, max_slope=1.0):
    """Pendientes por banda y prefijo, ajustadas con las filas de `df` (en la V27, todo 2025).

    La referencia es el proxy NM/P10, no la predicción in-sample de un regresor flexible.
    """
    apt = df.apto_entrenamiento.to_numpy(bool)
    p10 = M.ajustar_p10(df.loc[apt & df.combo.notna().to_numpy()], 10)
    en = M.aplicar_p10(df, p10)
    nm = df.taxi_nm_sec.to_numpy(float)
    ref = np.where((nm >= 0) & (nm <= 7200), nm, en.ref_p10_sec.to_numpy(float))
    sel = seleccion(df)
    s = df.despegue_menos_sched_sec.to_numpy(float)
    delta = s - ref
    y = df.target.to_numpy(float)
    a = delta * (y - ref)
    b = delta ** 2
    bands = banda(df)
    if frequency:
        a = (np.abs(y - s) <= 120).astype(float)
        b = np.ones(len(y))
    prefix = df.flight_prefix.astype("string").fillna("__NA__").to_numpy()
    coef = {}
    counts = {}
    for k in range(3):
        mask = sel & (bands == k)
        p = float(np.clip(a[mask].sum() / max(b[mask].sum(), 1), 0, max_slope))
        # Fuerza del prior medida en observaciones típicas de hueco al cuadrado dentro de la banda.
        penalty = prior_count * np.mean(b[mask]) if mask.any() else 1
        coef[str(k)] = {"default": p, "prefix": {}}
        counts[str(k)] = int(mask.sum())
        for key in np.unique(prefix[mask]):
            sub = mask & (prefix == key)
            coef[str(k)]["prefix"][str(key)] = float(np.clip((a[sub].sum() + penalty * p) / (b[sub].sum() + penalty),
                                                             0, max_slope))
    c = (df.source_above_2h.to_numpy() == 4)
    # Suavizado Beta(1,1) del acuerdo con una etiqueta larga; seis filas en todo el año.
    consensus = float((np.sum(y[c] > 7200) + 1) / (c.sum() + 2))
    return {"bands": coef, "prior_count": prior_count, "frequency": frequency, "max_slope": max_slope,
            "counts": counts, "consensus_p": consensus, "consensus_n": int(c.sum())}


def aplicar(df, base, cfg, prefix=True, consensus=True):
    pred = np.asarray(base, float).copy()
    mask = seleccion(df)
    bands = banda(df)
    pf = df.flight_prefix.astype("string").fillna("__NA__").to_numpy()
    for k in range(3):
        use = mask & (bands == k)
        c = cfg["bands"][str(k)]
        p = np.array([c["prefix"].get(str(v), c["default"]) if prefix else c["default"] for v in pf[use]])
        pred[use] = (1 - p) * pred[use] + p * df.despegue_menos_sched_sec.to_numpy(float)[use]
    if consensus:
        use = df.source_above_2h.to_numpy() == 4
        p = cfg["consensus_p"]
        pred[use] = (1 - p) * pred[use] + p * df.source_min.to_numpy(float)[use]
    return np.maximum(pred, K.SUELO_SEC)


def relojes_lobt(anio: int) -> pd.DataFrame:
    return pd.read_parquet(K.interim("dep", anio), columns=[K.ID, "MVT_TIME_UTC_mvt", "LOBT_flt"])


def proyectar(frame, pred):
    """Recorta al intervalo (despegue − LOBT) ± 3606 s; sin LOBT conserva la predicción. Suelo de 60 s."""
    center = (frame.MVT_TIME_UTC_mvt - frame.LOBT_flt).dt.total_seconds().to_numpy(float)
    valid = np.isfinite(center)
    lower = np.maximum(center - K.MARGEN_LOBT_SEC, K.SUELO_SEC)
    upper = center + K.MARGEN_LOBT_SEC
    assert not np.any(valid & (upper < K.SUELO_SEC)), "Intervalo incompatible con el suelo de 60 s"
    result = np.asarray(pred, dtype=float).copy()
    result[valid] = np.clip(result[valid], lower[valid], upper[valid])
    return np.maximum(result, K.SUELO_SEC)
