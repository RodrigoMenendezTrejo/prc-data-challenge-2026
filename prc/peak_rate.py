"""Peak service rate al estilo de Dalmau, Attia y Gawinowski (2023), adaptado a los datos del reto.

Experimento descartado (documento 02, sección 6); v30 no usa este módulo.

El artículo (*Modelling the Impact of Adverse Weather on Airport Peak Service Rate with Machine Learning*,
Atmosphere 14, 1476) define la capacidad observada como el percentil 99 de los movimientos por hora, según la
configuración de pistas y el tiempo. Aquí no metemos METAR, así que el P99 solo depende de la operación: es el
techo de movimientos por hora que sostiene cada configuración, no la capacidad con mal tiempo.

Pasos:
1. `construir_ventanas`: ventanas de 1 h cada 15 min (§3.1) con su throughput y las pistas en uso; una pista
   cuenta si se usó en el intervalo y en al menos 2 de los 3 siguientes (§3.1.2).
2. `catalogo` y `emparejar`: configuraciones típicas y emparejamiento por F_beta con beta = 2 (ec. 1-3). Sin las
   configuraciones oficiales del Airport Corner, el catálogo son las observadas con una cuota mínima de ventanas.
3. `ajustar_cuantil`: LightGBM cuantílico con alfa = 0,99 (§3.2, ec. 4); un modelo por tipo de movimiento para los
   diez aeropuertos juntos, como en el artículo.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config as C

# --- Constantes del método -------------------------------------------------------------------------
PASO_MIN = 15                 # cada ventana empieza 15 min después de la anterior (§3.1)
N_PASOS = 4                   # 4 intervalos de 15 min = la ventana de 1 h
HORA_INI, HORA_FIN = 7, 21    # ventanas que empiezan entre las 07:00 y las 21:00 locales (§3.1)
CUOTA_MIN = 0.05              # cuota mínima de ventanas para entrar en el catálogo (§3.1.2)
BETA = 2.0                    # beta del emparejamiento: manda el recall (§3.1.2)
ALFA = 0.99                   # el percentil que define el peak service rate (§3.1)
SIN_PISTAS = "SIN_PISTAS"     # ventana sin pistas de uso sostenido: no tiene configuración

# Dos juegos de predictores:
# - CAPACIDAD, el del artículo: aeropuerto y configuración de pistas (allí también la meteo). Da un techo que no
#   depende de la hora.
# - HORA añade la hora local y el día de la semana. Ya no es capacidad (a las 7 el techo baja por falta de
#   demanda), pero como denominador dice si una hora va cargada para lo que suele ir a esa hora.
# Ninguno usa el mes: enero y julio no están en train y el modelo predeciría a ciegas justo esos meses.
FEATURES_CAPACIDAD = ["airport", "cfg", "n_pistas_dep", "n_pistas_arr"]
FEATURES_HORA = FEATURES_CAPACIDAD + ["hora_local", "minuto_dia_local", "sin_minuto_dia", "cos_minuto_dia",
                                      "dia_semana_local", "fin_de_semana"]
FEATURES = FEATURES_CAPACIDAD          # por defecto, la variante del artículo
CATEGORICAS = ["airport", "cfg"]

PARAMS = {
    "objective": "quantile", "alpha": ALFA, "metric": "quantile",
    "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
    "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
    "lambda_l2": 1.0, "seed": 42, "num_threads": 0, "verbose": -1,
}


# --- 1 y 2 · Ventanas, throughput y pistas en uso ---------------------------------------------------
def _desplaza(a: np.ndarray, k: int) -> np.ndarray:
    """`a` desplazada `k` filas hacia arriba (la fila i ve el valor de i+k), rellenando con ceros.

    La ventana que empieza en i cubre de i a i+3, así que el throughput y el uso sostenido salen de aquí.
    """
    out = np.zeros_like(a)
    if k < a.shape[0]:
        out[:-k] = a[k:]
    return out


def _sostenidas(usada: np.ndarray) -> np.ndarray:
    """Uso sostenido (§3.1.2): usada en el intervalo i y en al menos 2 de los 3 siguientes.

    Sin esto, un solo movimiento en una pista secundaria crearía configuraciones que nunca estuvieron activas.
    """
    siguientes = sum(_desplaza(usada, k).astype("int8") for k in (1, 2, 3))
    return usada & (siguientes >= 2)


def _ventanas_aeropuerto(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ventana de 1 h de un aeropuerto, con el throughput y las pistas en uso de cada tipo."""
    bins = pd.concat([dep["bin"], arr["bin"]])
    grid = pd.date_range(bins.min().floor("D"), bins.max().ceil("D"), freq=f"{PASO_MIN}min")

    salida = {"inicio": grid}
    for nombre, d in (("dep", dep), ("arr", arr)):
        # Throughput entregado: movimientos de los 4 intervalos que forman la ventana.
        c = d.groupby("bin").size().reindex(grid, fill_value=0).to_numpy()
        salida[f"thr_{nombre}"] = c + sum(_desplaza(c, k) for k in range(1, N_PASOS))

        usada = pd.crosstab(d["bin"], d["RUNWAY_mvt"]).reindex(grid, fill_value=0)
        pistas = np.asarray(usada.columns)
        S = _sostenidas(usada.to_numpy() > 0)
        salida[f"pistas_{nombre}"] = ["+".join(pistas[fila]) for fila in S]
        salida[f"n_pistas_{nombre}"] = S.sum(axis=1)
    return pd.DataFrame(salida)


def construir_ventanas(dep_file, arr_file) -> pd.DataFrame:
    """Ventanas de los 10 aeropuertos a partir de los movimientos de `data/interim/`.

    Una fila por aeropuerto y ventana, con la hora de inicio en hora local (la noche se excluye por demanda).
    Se quitan las ventanas vacías, que en 2026 son los meses fuera del ranking.
    """
    cols = ["airport", "RUNWAY_mvt", "mvt_time_local"]
    dep = pd.read_parquet(dep_file, columns=cols).dropna(subset=["RUNWAY_mvt"])
    arr = pd.read_parquet(arr_file, columns=cols).dropna(subset=["RUNWAY_mvt"])
    for d in (dep, arr):
        d["bin"] = d["mvt_time_local"].dt.floor(f"{PASO_MIN}min")

    trozos = [_ventanas_aeropuerto(dep[dep["airport"] == apt], arr[arr["airport"] == apt]).assign(airport=apt)
              for apt in C.AIRPORTS]
    v = pd.concat(trozos, ignore_index=True)
    v = v[(v["thr_dep"] > 0) | (v["thr_arr"] > 0)].reset_index(drop=True)
    return features_tiempo(v)


def features_tiempo(v: pd.DataFrame) -> pd.DataFrame:
    """Features de tiempo de cada ventana y la marca de horario diurno (`en_horario`)."""
    inicio = v["inicio"]
    minuto = inicio.dt.hour * 60 + inicio.dt.minute
    return v.assign(
        fecha_local=inicio.dt.date,
        hora_local=inicio.dt.hour,
        minuto_dia_local=minuto,
        sin_minuto_dia=np.sin(2 * np.pi * minuto / 1440),
        cos_minuto_dia=np.cos(2 * np.pi * minuto / 1440),
        dia_semana_local=inicio.dt.dayofweek,
        fin_de_semana=(inicio.dt.dayofweek >= 5).astype("int8"),
        en_horario=inicio.dt.hour.between(HORA_INI, HORA_FIN),
    )


# --- 3 · Configuraciones de pista -------------------------------------------------------------------
def _vectores(combos, universo: list[str]) -> np.ndarray:
    """Cada combinación 'pistas_arr / pistas_dep' como vector binario (§3.1.2): uso de cada pista en llegadas
    y luego en salidas, para compararlas como clasificación multietiqueta.
    """
    idx = {p: i for i, p in enumerate(universo)}
    M = np.zeros((len(combos), 2 * len(universo)), dtype="int8")
    for fila, combo in enumerate(combos):
        llegadas, salidas = combo.split(" / ")
        for p in filter(None, llegadas.split("+")):
            M[fila, idx[p]] = 1
        for p in filter(None, salidas.split("+")):
            M[fila, len(universo) + idx[p]] = 1
    return M


def f_beta(obs: np.ndarray, cfg: np.ndarray, beta: float = BETA) -> np.ndarray:
    """F_beta entre las pistas observadas y cada configuración candidata (ec. 1-3 del artículo).

    Con beta = 2 manda el recall: la configuración asignada tiene que cubrir las pistas usadas, aunque sobre alguna.
    """
    casadas = obs @ cfg.T
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = casadas / cfg.sum(axis=1)[None, :]
        recall = casadas / obs.sum(axis=1)[:, None]
        f = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    return np.nan_to_num(f)


def catalogo(v: pd.DataFrame, cuota_min: float = CUOTA_MIN) -> pd.DataFrame:
    """Configuraciones típicas de cada aeropuerto: las que superan `cuota_min` de sus ventanas diurnas.

    Las raras se descartan porque su P99 saldría de cuatro observaciones (§3.1.2). Se numeran por frecuencia:
    la 0 de cada aeropuerto es la más usada.
    """
    d = v[v["en_horario"]]
    combo = d["pistas_arr"] + " / " + d["pistas_dep"]
    filas = []
    for apt, g in combo.groupby(d["airport"]):
        frec = g.value_counts(normalize=True)
        frec = frec[frec.index != " / "]
        tipicas = frec[frec >= cuota_min]
        if tipicas.empty:                       # que ningún aeropuerto se quede sin catálogo
            tipicas = frec.head(1)
        for k, (combinacion, cuota) in enumerate(tipicas.items()):
            filas.append({"airport": apt, "k": k, "cfg": f"{apt}|{k}", "combinacion": combinacion,
                          "cuota": round(float(cuota), 4)})
    return pd.DataFrame(filas)


def emparejar(v: pd.DataFrame, cat: pd.DataFrame, beta: float = BETA) -> pd.DataFrame:
    """Asigna a cada ventana la configuración del catálogo más parecida, con su F_beta.

    Las ventanas sin pistas de uso sostenido (madrugada, días flojos) quedan como `SIN_PISTAS`, con F_beta 0,
    y no entran en el ajuste.
    """
    combo = (v["pistas_arr"] + " / " + v["pistas_dep"]).to_numpy()
    cfg = np.full(len(v), SIN_PISTAS, dtype=object)
    score = np.zeros(len(v))

    for apt, g in cat.groupby("airport"):
        filas = np.flatnonzero((v["airport"] == apt).to_numpy())
        universo = sorted({p for c in np.r_[combo[filas], g["combinacion"].to_numpy()]
                           for lado in c.split(" / ") for p in lado.split("+") if p})
        f = f_beta(_vectores(combo[filas], universo), _vectores(g["combinacion"].to_numpy(), universo), beta)
        mejor, maximo = f.argmax(axis=1), f.max(axis=1)
        tiene = maximo > 0
        cfg[filas[tiene]] = g["cfg"].to_numpy()[mejor[tiene]]
        score[filas] = maximo
    return v.assign(cfg=cfg, cfg_f2=score.round(3))


# --- 4 · Peak service rate ---------------------------------------------------------------------------
def pinball(y, p, alfa: float = ALFA) -> float:
    """Error pinball medio (ecuación 4): la pérdida que se minimiza al aprender un percentil."""
    e = np.asarray(y, dtype="float64") - np.asarray(p, dtype="float64")
    return float(np.mean(np.maximum(alfa * e, (alfa - 1) * e)))


def p99_empirico(v: pd.DataFrame, tipo: str, claves=("airport", "cfg"), min_filas: int = 30) -> pd.DataFrame:
    """P99 del throughput por grupos, sin modelo: el control con el que comparamos el regresor.

    Es el método que el artículo descarta (§3.2) por los grupos con pocas observaciones.
    """
    g = v.groupby(list(claves), observed=True)[f"thr_{tipo}"]
    t = pd.DataFrame({"n": g.size(), "peak": g.quantile(ALFA), "mediana": g.median()})
    return t[t["n"] >= min_filas].reset_index()


def preparar(v: pd.DataFrame, cat: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Matriz de predictores con `airport` y `cfg` como categóricas de niveles fijos (los del catálogo)."""
    X = v[list(features or FEATURES)].copy()
    X["airport"] = pd.Categorical(X["airport"], categories=sorted(C.AIRPORTS))
    X["cfg"] = pd.Categorical(X["cfg"], categories=sorted(cat["cfg"]))
    return X


def ajustar_cuantil(train: pd.DataFrame, val: pd.DataFrame, cat: pd.DataFrame, tipo: str,
                    features: list[str] | None = None, params: dict | None = None,
                    rondas: int = 2000, parada: int = 100) -> lgb.Booster:
    """Regresor cuantílico (alfa = 0,99) del throughput de `tipo` ('dep' o 'arr').

    Un solo modelo para los 10 aeropuertos (§3.2): aprende mejor con todas las ventanas y puede opinar sobre
    configuraciones poco vistas en un aeropuerto.
    """
    y = f"thr_{tipo}"
    d_tr = lgb.Dataset(preparar(train, cat, features), label=train[y], categorical_feature=CATEGORICAS)
    d_va = lgb.Dataset(preparar(val, cat, features), label=val[y], categorical_feature=CATEGORICAS,
                       reference=d_tr)
    return lgb.train({**PARAMS, **(params or {})}, d_tr, num_boost_round=rondas, valid_sets=[d_va],
                     valid_names=["val"], callbacks=[lgb.early_stopping(parada, verbose=False)])


def predecir(modelo: lgb.Booster, v: pd.DataFrame, cat: pd.DataFrame) -> np.ndarray:
    """Peak service rate predicho (mínimo 1 movimiento por hora). Vale para las dos variantes porque toma las
    variables del propio modelo (`feature_name()`).
    """
    return np.maximum(modelo.predict(preparar(v, cat, modelo.feature_name())), 1.0)


def shapley(modelo: lgb.Booster, v: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Valores de Shapley por observación (§5.2), en movimientos por hora.

    `pred_contrib` de LightGBM es TreeSHAP; la última columna es el valor base y se descarta.
    """
    features = modelo.feature_name()
    contrib = modelo.predict(preparar(v, cat, features), pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=features, index=v.index)


# --- Features por movimiento -------------------------------------------------------------------------
# Columnas que la tabla de ventanas tiene que traer ya calculadas para poder cruzarlas con los movimientos:
# `cap_*` es el peak rate de la variante capacidad y `techo_*` el de la variante con hora.
PEAK_VENTANA = ["cap_dep", "cap_arr", "techo_dep", "techo_arr"]


def features_por_movimiento(mov_file, ventanas: pd.DataFrame) -> pd.DataFrame:
    """Cruza cada movimiento con la ventana centrada en él y devuelve su saturación.

    Usamos la ventana que empieza 30 min antes del movimiento; la del artículo mira solo hacia adelante y
    perdería la cola que una salida ya tenía delante. Cada movimiento recibe las saturaciones de salidas y de
    llegadas, porque una salida también espera por las llegadas:

    - `pk_sat_*` = throughput / capacidad: 1 es que esa hora entregó todo lo que da la configuración.
    - `pk_sat_hora_*` = throughput / techo de esa hora: 1 es que fue de las más cargadas para esa hora del día.
    """
    mov = pd.read_parquet(mov_file, columns=[C.ID, "airport", "mvt_time_local"])
    desfase = pd.Timedelta(minutes=(N_PASOS // 2) * PASO_MIN)
    mov = mov.assign(inicio=mov["mvt_time_local"].dt.floor(f"{PASO_MIN}min") - desfase)

    cols = ["airport", "inicio", "cfg", "cfg_f2", "en_horario", "thr_dep", "thr_arr", *PEAK_VENTANA]
    out = mov.merge(ventanas[cols], on=["airport", "inicio"], how="left", validate="many_to_one")
    for tipo in ("dep", "arr"):
        out[f"sat_{tipo}"] = out[f"thr_{tipo}"] / out[f"cap_{tipo}"]
        out[f"sat_hora_{tipo}"] = out[f"thr_{tipo}"] / out[f"techo_{tipo}"]

    columnas = ["cfg", "cfg_f2", "en_horario", "thr_dep", "thr_arr", "cap_dep", "cap_arr",
                "sat_dep", "sat_arr", "techo_dep", "techo_arr", "sat_hora_dep", "sat_hora_arr"]
    return out[[C.ID, *columnas]].rename(columns={c: f"pk_{c}" for c in columnas})
