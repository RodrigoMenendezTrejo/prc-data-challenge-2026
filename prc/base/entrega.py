"""Validación del fichero de predicciones contra la plantilla oficial y comparación con otro fichero."""

import numpy as np
import pandas as pd

from . import config as K
from .utiles import sha256


def validar(path) -> dict:
    """Formato que exige el reto: dos columnas, IDs de la plantilla en su orden, float64/int32, >= 60 s."""
    p = pd.read_parquet(path)
    t = pd.read_parquet(K.SUBMITTING_FILE)
    comprobaciones = {
        "columnas": list(p) == [K.ID, K.TARGET],
        "filas": len(p) == K.N_FILAS_ENVIO,
        "ids_unicos": bool(p[K.ID].is_unique),
        "orden_plantilla": bool(p[K.ID].equals(t[K.ID])),
        "tipos": str(p[K.ID].dtype) == "float64" and str(p[K.TARGET].dtype) == "int32",
        "sin_nulos": bool(p.notna().all().all()),
        "finitas": bool(np.isfinite(p[K.TARGET]).all()),
        "suelo_60": bool(p[K.TARGET].ge(K.SUELO_SEC).all()),
    }
    return {"valido": all(comprobaciones.values()), **comprobaciones, "sha256": sha256(path)}


def comparar_con_enviado(path, enviado=K.PREDICCION_V27) -> dict:
    nuevo = pd.read_parquet(path)
    viejo = pd.read_parquet(enviado)
    m = nuevo.merge(viejo, on=K.ID, suffixes=("_nuevo", "_enviado"), validate="one_to_one")
    dif = m[f"{K.TARGET}_nuevo"].to_numpy(float) - m[f"{K.TARGET}_enviado"].to_numpy(float)
    return {"filas": len(m), "predicciones_distintas": int((dif != 0).sum()), "max_abs_diferencia": float(abs(dif).max()),
            "mismo_sha256": sha256(path) == sha256(enviado), "sha256_enviado": sha256(enviado)}
