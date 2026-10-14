"""Matriz de las 137 variables base y referencias del residuo (tabla P10 y vocabulario de categóricas).

- Referencia del residuo: taxi NM (despegue − AOBT_3) si está entre 0 y 7200 s; si no, P10 de su grupo
  (combo -> dique-pista -> pista -> aeropuerto).
- Las referencias se ajustan con 2025 y se guardan en `configuracion/referencias/` (`ajustar_referencias` las
  recalcula; el cuaderno 02 comprueba que coinciden con las guardadas).
- `matriz(anio, refs, receta)` devuelve la tabla del año, las 137 variables en el orden de la receta y la referencia.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prc import referencias as M

from . import config as K
from . import post_proceso
from .utiles import guardar_json, leer_json

ID = K.ID
RECETA = K.CONFIG_DIR / "receta.json"


def receta() -> dict:
    return leer_json(RECETA)


def normal(d) -> np.ndarray:
    """Régimen normal: taxi NM observable entre 0 y 7200 s. Es la ruta de la mezcla."""
    return d.taxi_nm_sec.between(0, K.NM_MAX_SEC).to_numpy(copy=True)


def cargar(anio: int) -> pd.DataFrame:
    """Tabla combinada del año, con las categóricas como `category` (salvo las claves de la tabla P10)."""
    df = pd.read_parquet(K.feature("combinado", anio))
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c not in ["mes_utc", "split", *M.NIVELES_P10.values()]:
            df[c] = df[c].astype("category")
    return df


# --- Tablas ajustadas con el target de 2025 -------------------------------------------------------
@dataclass
class Referencias:
    p10: pd.DataFrame
    categorias: dict
    calibracion: dict


def ajustar_referencias(d: pd.DataFrame, r: dict) -> Referencias:
    """P10 por grupo, niveles de las categóricas y calibración de fuentes, con todo 2025."""
    apt = d.apto_entrenamiento.to_numpy()
    p10 = M.ajustar_p10(d.loc[apt & d.combo.notna().to_numpy()], r["p10_min_bajo"])
    enriquecido = M.aplicar_p10(d, p10)
    categorias = M.ajustar_categorias(enriquecido, r["variables"]["categoricas_referencia"], r["categorias_min_filas"])
    calibracion = post_proceso.ajustar(d, r["calibracion"]["prior_count"], max_slope=r["calibracion"]["max_slope"])
    return Referencias(p10, categorias, calibracion)


def guardar_referencias(refs: Referencias, carpeta: Path):
    carpeta.mkdir(parents=True, exist_ok=True)
    refs.p10.to_parquet(carpeta / "p10.parquet", index=False)
    guardar_json(refs.categorias, carpeta / "categories.json")
    guardar_json(refs.calibracion, carpeta / "calibration.json")


def leer_referencias(carpeta: Path = K.REFERENCIAS_DIR) -> Referencias:
    return Referencias(pd.read_parquet(carpeta / "p10.parquet"), leer_json(carpeta / "categories.json"),
                       leer_json(carpeta / "calibration.json"))


# --- Matriz de variables --------------------------------------------------------------------------
def matriz(anio: int, refs: Referencias, r: dict):
    """Devuelve `d` (tabla combinada), `X` (las 137 variables en el orden de los especialistas) y `ref`.

    Las 92 primeras columnas son las del cuerpo V26, las 109 primeras las del alineado.
    """
    v = r["variables"]
    contexto, cats = v["contexto"], v["categoricas"]
    d = cargar(anio)
    enriquecido = M.aplicar_p10(d, refs.p10)
    cols = [c for c in contexto if c in enriquecido]
    X = M.aplicar_categorias(enriquecido[cols], {k: n for k, n in refs.categorias.items() if k in cols})
    for fichero in [K.feature("meteo", anio), K.feature("trafico", anio)]:
        disponibles = [c for c in pd.read_parquet(fichero).columns if c in contexto and c != ID]
        q = d[[ID]].merge(pd.read_parquet(fichero, columns=[ID, *disponibles]), on=ID, how="left", validate="one_to_one")
        for c in disponibles:
            X[c] = q[c].to_numpy(dtype="float32")
    X = X[contexto].copy()
    q = d[[ID]].merge(pd.read_parquet(K.feature("puerta", anio)), on=ID, how="left", validate="one_to_one")
    for c in v["puerta"]:
        X[c] = q[c].to_numpy(dtype="float32")
    for c in cats:
        X[c] = X[c].astype("string").fillna("__MISSING__").astype(object)
    for c in set(X) - set(cats):
        X[c] = pd.to_numeric(X[c]).astype("float32")
    nm = d.taxi_nm_sec.to_numpy(float)
    ref = np.where((nm >= 0) & (nm <= K.NM_MAX_SEC), nm, enriquecido.ref_p10_sec.to_numpy(float))
    f = d[[ID]].merge(pd.read_parquet(K.feature("alineadas", anio)), on=ID, validate="one_to_one")
    extra = f[v["alineadas"]].astype("float32").copy()
    extra.index = X.index
    X = pd.concat([X, extra], axis=1)
    f = d[[ID]].merge(pd.read_parquet(K.feature("relojes", anio)), on=ID, validate="one_to_one").drop(columns=ID)
    X = pd.concat([X, f], axis=1)
    assert list(X) == v["especialistas"], "El orden de las variables no coincide con la receta"
    return d, X, ref
