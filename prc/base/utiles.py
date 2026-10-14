"""Utilidades pequeñas compartidas: JSON, hashes, registro con hora y comparación de tablas."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd


def leer_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def guardar_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def comparar_tablas(nueva: pd.DataFrame, original: pd.DataFrame, clave: str | None = None) -> dict:
    """Compara dos tablas celda a celda, ordenadas por `clave` si se indica.

    Devuelve un resumen: mismas filas y columnas, columnas con diferencias y la mayor diferencia numérica.
    Dos NaN en la misma celda cuentan como iguales.
    """
    if clave is not None:
        nueva = nueva.sort_values(clave).reset_index(drop=True)
        original = original.sort_values(clave).reset_index(drop=True)
    out = {"filas": [len(nueva), len(original)],
           "columnas_iguales": list(nueva.columns) == list(original.columns),
           "solo_en_nueva": sorted(set(nueva) - set(original)),
           "solo_en_original": sorted(set(original) - set(nueva)),
           "columnas_distintas": {}, "identica": False}
    if len(nueva) != len(original):
        return out
    for c in [c for c in original.columns if c in nueva.columns]:
        a, b = nueva[c], original[c]
        if a.equals(b):
            continue
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(a):
            x, y = a.to_numpy(float), b.to_numpy(float)
            iguales = (x == y) | (np.isnan(x) & np.isnan(y))
            if iguales.all():
                if str(a.dtype) != str(b.dtype):
                    out["columnas_distintas"][c] = f"mismos valores, tipo {a.dtype} frente a {b.dtype}"
                continue
            out["columnas_distintas"][c] = f"{int((~iguales).sum())} celdas, máx |dif| {np.nanmax(abs(x - y)):.3g}"
        else:
            sa, sb = a.astype("string"), b.astype("string")
            iguales = (sa == sb).fillna(False) | (sa.isna() & sb.isna())
            if iguales.all():
                if str(a.dtype) != str(b.dtype):
                    out["columnas_distintas"][c] = f"mismos valores, tipo {a.dtype} frente a {b.dtype}"
                continue
            out["columnas_distintas"][c] = f"{int((~iguales).sum())} celdas distintas"
    out["identica"] = (out["columnas_iguales"] and not out["columnas_distintas"]
                       and not out["solo_en_nueva"] and not out["solo_en_original"])
    out["mismos_valores"] = (not out["solo_en_nueva"] and not out["solo_en_original"]
                             and all(v.startswith("mismos valores") for v in out["columnas_distintas"].values()))
    return out
