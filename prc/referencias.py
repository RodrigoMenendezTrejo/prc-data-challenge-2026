"""Referencias del residuo: tabla P10 por grupo y vocabulario fijo de categóricas (se ajustan con 2025)."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config as C

# Nivel común para las categorías raras en entrenamiento y para las que no se vieron nunca (p. ej. un stand
# nuevo en 2026).
NIVEL_OTROS = "__OTROS__"


# --- Experimentos ----------------------------------------------------------------------------------
def exp_dir(exp_id: str) -> Path:
    return C.MODELS_DIR / exp_id


def guardar_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def leer_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- Categóricas -----------------------------------------------------------------------------------
def ajustar_categorias(df: pd.DataFrame, columnas: list[str], min_filas: int) -> dict[str, list[str]]:
    """Niveles de cada categórica con al menos `min_filas` filas, más el nivel común.

    Se ajusta solo con las filas de entrenamiento, porque contar niveles es mirar muchas filas a la vez.
    """
    categorias = {}
    for col in columnas:
        conteo = df[col].value_counts(dropna=True)
        niveles = sorted(str(v) for v in conteo[conteo >= min_filas].index)
        categorias[col] = niveles + [NIVEL_OTROS]
    return categorias


def aplicar_categorias(df: pd.DataFrame, categorias: dict[str, list[str]]) -> pd.DataFrame:
    """Convierte cada categórica a `pd.Categorical` con exactamente los niveles de entrenamiento.

    Lo raro y lo no visto va a NIVEL_OTROS y los nulos siguen nulos. Sin esto, un nivel nuevo desplaza los
    códigos y LightGBM predice con la categoría equivocada sin avisar.
    """
    out = df.copy()
    for col, niveles in categorias.items():
        s = out[col].astype("string")
        s = s.where(s.isin(niveles) | s.isna(), NIVEL_OTROS)
        out[col] = pd.Categorical(s, categories=niveles)
    return out


# --- Referencia P10 (método ATXOT de EUROCONTROL) --------------------------------------------------
# Cadena de respaldo, de lo más fino a lo más grueso: nivel -> columna clave. Las claves son los cruces que
# calcula `construir_features` en 03, así que la tabla se cruza con un simple `map`.
NIVELES_P10 = {"combo": "combo", "dique": "apt_dique_pista", "pista": "apt_pista", "aeropuerto": "airport"}

# Features que salen de la tabla P10. No se guardan en data/features/: dependen de con qué meses se ajustó
# la tabla, así que viajan con el modelo (p10.parquet / p10_final.parquet) y se añaden al predecir.
FEATURES_P10 = ["ref_p10_sec", "ref_p25_sec", "ref_muestra", "ref_nivel"]


def ajustar_p10(df: pd.DataFrame, min_bajo_p10: int, col_y: str = "target") -> pd.DataFrame:
    """Tabla de referencia: P10, P25 y nº de vuelos de cada grupo, en los cuatro niveles.

    `df` son las filas aptas con stand y pista de los meses de entrenamiento: la tabla usa el target de muchas
    filas, como una codificación de la variable objetivo. Un grupo vale si al menos `min_bajo_p10` vuelos quedan
    en su P10 o por debajo (regla ATXOT).
    """
    tablas = []
    for nivel, clave in NIVELES_P10.items():
        d = df[[clave, col_y]].dropna()
        g = d.groupby(clave)[col_y]
        t = pd.DataFrame({"p10": g.quantile(0.10), "p25": g.quantile(0.25), "n": g.size()})
        bajo_p10 = d[col_y] <= d[clave].map(t["p10"])
        t["n_bajo_p10"] = bajo_p10.groupby(d[clave]).sum()
        t = t[t["n_bajo_p10"] >= min_bajo_p10]
        tablas.append(t.rename_axis("clave").reset_index().assign(nivel=nivel))
    return pd.concat(tablas, ignore_index=True)[["nivel", "clave", "p10", "p25", "n", "n_bajo_p10"]]


def aplicar_p10(df: pd.DataFrame, tabla: pd.DataFrame) -> pd.DataFrame:
    """Añade las FEATURES_P10: cada fila toma la referencia del primer nivel de la cadena que sea válido.

    El nivel `aeropuerto` siempre existe, así que ninguna fila se queda sin referencia.
    """
    p10 = pd.Series(np.nan, index=df.index)
    p25 = pd.Series(np.nan, index=df.index)
    n = pd.Series(np.nan, index=df.index)
    nivel = pd.Series(pd.NA, index=df.index, dtype="string")
    for nombre, clave in NIVELES_P10.items():
        t = tabla[tabla["nivel"] == nombre].set_index("clave")
        claves = df[clave].where(nivel.isna())
        encontrado = claves.map(t["p10"]).notna()
        p10 = p10.mask(encontrado, claves.map(t["p10"]))
        p25 = p25.mask(encontrado, claves.map(t["p25"]))
        n = n.mask(encontrado, claves.map(t["n"]))
        nivel = nivel.mask(encontrado, nombre)
    return df.assign(ref_p10_sec=p10, ref_p25_sec=p25, ref_muestra=n, ref_nivel=nivel)


# --- Regla del off-block programado ----------------------------------------------------------------
# En algunos aeropuertos (en 2025, solo LIRF), si una salida no enlaza con el plan de vuelo, el off-block publicado
# es la hora programada y el target pasa a ser `despegue − hora programada`, que sí conocemos en 2026. A esas filas
# les damos la esperanza de los dos casos:
#     p · (despegue − programada) + (1 − p) · predicción del modelo
# con `p` = fracción de filas de ese aeropuerto y tramo de retraso en las que el off-block era la hora programada.
# Se ajusta con las filas de entrenamiento, también las no aptas (que son las que importan), y va con el modelo.
def en_regla_programada(df: pd.DataFrame, regla: dict | None) -> pd.Series:
    """Filas a las que se aplica la regla: aeropuerto activado, sin plan de vuelo y retraso sobre lo programado > umbral."""
    if not regla or not regla["aeropuertos"]:
        return pd.Series(False, index=df.index)
    return (df["airport"].isin(list(regla["aeropuertos"])) & df["sin_flight_id"].astype(bool)
            & (df["despegue_menos_sched_sec"] > regla["umbral_retraso_sec"]))


def _tramo_retraso(retraso: pd.Series, tramos_sec: list[float]) -> pd.Series:
    """Índice del tramo de retraso (0, 1, …): [t0, t1), [t1, t2), …, [tn, ∞)."""
    return pd.cut(retraso, bins=[*tramos_sec, np.inf], right=False, labels=False)


def ajustar_regla_programada(df: pd.DataFrame, umbral_retraso_sec: float, tramos_sec: list[float],
                             tolerancia_sec: float, p_min: float, min_filas: int, col_y: str = "target") -> dict:
    """Ajusta la regla con las filas con target (aptas o no) de los meses de entrenamiento.

    Candidatas: sin plan de vuelo y con `despegue − programada` > `umbral_retraso_sec`. Una candidata es "off-block
    programado" si su target coincide con `despegue − programada` a ± `tolerancia_sec`. Un aeropuerto se activa si
    tiene al menos `min_filas` candidatas y al menos una fracción `p_min` de ellas lo son. Para los activados se guarda
    `p` por tramo de retraso; un tramo con menos de 10 filas usa la `p` del aeropuerto. `candidatos` guarda el
    diagnóstico de todos los aeropuertos, activados o no.
    """
    cand = df[df[col_y].notna() & df["sin_flight_id"].astype(bool) & (df["despegue_menos_sched_sec"] > umbral_retraso_sec)]
    es_programada = (cand[col_y] - cand["despegue_menos_sched_sec"]).abs() <= tolerancia_sec
    tramo = _tramo_retraso(cand["despegue_menos_sched_sec"], tramos_sec)
    candidatos, activos = {}, {}
    for apt, idx in cand.groupby("airport").groups.items():
        n, p = len(idx), float(es_programada.loc[idx].mean())
        candidatos[str(apt)] = {"n": n, "p": round(p, 4)}
        if n >= min_filas and p >= p_min:
            por_tramo = []
            for t in range(len(tramos_sec)):
                sel = idx[(tramo.loc[idx] == t).to_numpy()]
                por_tramo.append(round(float(es_programada.loc[sel].mean()), 4) if len(sel) >= 10 else round(p, 4))
            activos[str(apt)] = {"n": n, "p": por_tramo}
    return {"umbral_retraso_sec": umbral_retraso_sec, "tramos_sec": list(tramos_sec), "tolerancia_sec": tolerancia_sec,
            "p_min": p_min, "min_filas": min_filas, "aeropuertos": activos, "candidatos": candidatos}


def aplicar_regla_programada(df: pd.DataFrame, pred, regla: dict | None) -> np.ndarray:
    """`p · (despegue − programada) + (1 − p) · pred` en las filas de la regla; `pred` sin tocar en el resto."""
    pred = pd.Series(np.asarray(pred, dtype="float64"), index=df.index)
    en = en_regla_programada(df, regla)
    if not en.any():
        return pred.to_numpy()
    retraso = df["despegue_menos_sched_sec"]
    tramo = _tramo_retraso(retraso, regla["tramos_sec"])
    p = pd.Series(np.nan, index=df.index)
    for apt, info in regla["aeropuertos"].items():
        m = en & (df["airport"] == apt)
        p[m] = tramo[m].astype(int).map(dict(enumerate(info["p"])))
    return pred.where(~en, p * retraso + (1 - p) * pred).to_numpy()


# --- Predicción ------------------------------------------------------------------------------------
def cargar_regla(exp_id: str, final: bool) -> dict | None:
    """La regla del off-block programado que va con el modelo (mismo criterio de sufijo que `cargar_modelo`).

    None si el experimento no la usa. Va aparte de `cargar_modelo` para no romper los cuadernos antiguos.
    """
    d = exp_dir(exp_id)
    sufijo = "_final" if final and (d / "modelo_final.txt").exists() else ""
    path = d / f"regla_programada{sufijo}.json"
    return leer_json(path) if path.exists() else None


def cargar_modelo(exp_id: str, final: bool) -> tuple[lgb.Booster, dict[str, list[str]], pd.DataFrame | None, str]:
    """Modelo + sus categorías + su tabla P10 (None si el experimento no la usa).

    `final=True` usa el re-entrenado con los 12 meses si existe. Los tres ficheros van siempre juntos: el
    modelo final con las categorías y la P10 de los 12 meses, el experimental con los de train.
    """
    d = exp_dir(exp_id)
    sufijo = "_final" if final and (d / "modelo_final.txt").exists() else ""
    modelo = lgb.Booster(model_file=str(d / f"modelo{sufijo}.txt"))
    categorias = leer_json(d / f"categorias{sufijo}.json")
    path_p10 = d / f"p10{sufijo}.parquet"
    p10 = pd.read_parquet(path_p10) if path_p10.exists() else None
    return modelo, categorias, p10, f"modelo{sufijo}"


def predecir(modelo: lgb.Booster, df: pd.DataFrame, features: list[str],
             categorias: dict[str, list[str]], p10: pd.DataFrame | None = None,
             regla: dict | None = None) -> np.ndarray:
    """Predicción del modelo, la regla del off-block programado (si se pasa) y el recorte inferior (nunca superior).

    Con `p10`, añade antes las FEATURES_P10. Sin ella, `df` ya tiene que traerlas si el modelo las usa. Con `regla`,
    `df` necesita `airport`, `sin_flight_id` y `despegue_menos_sched_sec`.
    """
    if p10 is not None:
        df = aplicar_p10(df, p10)
    pred = modelo.predict(aplicar_categorias(df[features], categorias))
    if regla is not None:
        pred = aplicar_regla_programada(df, pred, regla)
    return np.maximum(pred, C.PRED_MIN_SEC)


# --- Métricas --------------------------------------------------------------------------------------
def metricas(y, p) -> dict:
    """RMSE (la métrica del reto), MAE (diagnóstico), sesgo medio (pred − real) y R²."""
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    e = p - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "sesgo": float(np.mean(e)),
        "r2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
    }


def rmse(y, p) -> float:
    return metricas(y, p)["rmse"]


def resumen_por(df: pd.DataFrame, grupo: str, col_y: str = "real", col_p: str = "pred") -> pd.DataFrame:
    """Métricas por grupo y reparto del error: qué % del error cuadrático total aporta cada grupo."""
    d = df.assign(_e2=(df[col_p] - df[col_y]) ** 2, _ae=(df[col_p] - df[col_y]).abs(),
                  _e=df[col_p] - df[col_y])
    g = d.groupby(grupo, observed=True, dropna=False).agg(
        n=("_e2", "size"), sse=("_e2", "sum"), mae=("_ae", "mean"), sesgo=("_e", "mean"))
    g["pct_filas"] = 100 * g["n"] / g["n"].sum()
    g["rmse"] = np.sqrt(g["sse"] / g["n"])
    g["pct_error_cuadratico"] = 100 * g["sse"] / g["sse"].sum()
    g = g[["n", "pct_filas", "rmse", "mae", "sesgo", "pct_error_cuadratico"]]
    return g.sort_values("pct_error_cuadratico", ascending=False).round(1)
