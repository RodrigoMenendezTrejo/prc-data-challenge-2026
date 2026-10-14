"""Calibración de la cola de Roma con el historial de emparejamiento NM del número de vuelo.

Misma forma que la de V27 (`v27/fuentes.py`): en salidas de LIRF sin FLIGHT_ID y con despegue > 2 h tras la hora
programada, pred = (1 − p)·cuerpo + p·(despegue − programada), con p = pendiente de mínimos cuadrados respecto a la
referencia NM/P10. Cambia la clave de p: (tramo de retraso × grupo), donde el grupo separa los números de vuelo que
casi siempre se emparejan con NM ('habitual', share_other >= 0,9) de los raros, regularizado hacia el tramo.
El consenso de cuatro fuentes y la proyección LOBT se mantienen como en V27.
"""
import numpy as np

SUELO = 60.0


def grupo(share_other, n_other):
    s = np.nan_to_num(np.asarray(share_other, float), nan=0.0)
    n = np.nan_to_num(np.asarray(n_other, float), nan=0.0)
    return np.where((n >= 1) & (s >= 0.9), 1, 0)   # 1 = habitual, 0 = raro/nuevo


def seleccion(df):
    return ((df.airport.astype(str) == "LIRF").to_numpy() & df.sin_flight_id.to_numpy(bool)
            & (df.despegue_menos_sched_sec.to_numpy() > 7200))


def banda(df):
    return np.minimum(np.searchsorted([10800, 21600], df.despegue_menos_sched_sec.to_numpy(), side="right"), 2)


def ajustar(df, ref, share_other, n_other, prior=20, max_slope=1.5):
    sel = seleccion(df)
    s = df.despegue_menos_sched_sec.to_numpy(float)
    y = df.target.to_numpy(float)
    delta = s - ref
    a, b = delta * (y - ref), delta ** 2
    bands, g = banda(df), grupo(share_other, n_other)
    coef = {}
    for k in range(3):
        mk = sel & (bands == k)
        p = float(np.clip(a[mk].sum() / max(b[mk].sum(), 1), 0, max_slope))
        pen = prior * np.mean(b[mk]) if mk.any() else 1
        coef[str(k)] = {"default": p}
        for gg in (0, 1):
            sub = mk & (g == gg)
            coef[str(k)][str(gg)] = float(np.clip((a[sub].sum() + pen * p) / (b[sub].sum() + pen), 0, max_slope))
    c = df.source_above_2h.to_numpy() == 4
    consensus = float((np.sum(y[c] > 7200) + 1) / (c.sum() + 2))
    return {"bands": coef, "consensus_p": consensus}


def aplicar(df, base, cfg, share_other, n_other):
    pred = np.asarray(base, float).copy()
    sel = seleccion(df)
    bands, g = banda(df), grupo(share_other, n_other)
    s = df.despegue_menos_sched_sec.to_numpy(float)
    for k in range(3):
        for gg in (0, 1):
            use = sel & (bands == k) & (g == gg)
            p = cfg["bands"][str(k)][str(gg)]
            pred[use] = (1 - p) * pred[use] + p * s[use]
    use = df.source_above_2h.to_numpy() == 4
    p = cfg["consensus_p"]
    pred[use] = (1 - p) * pred[use] + p * df.source_min.to_numpy(float)[use]
    return np.maximum(pred, SUELO)
