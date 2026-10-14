"""Tablas de variables base por vuelo, de `Data/interim/` a `Data/features/base/` (las de la entrega V27).

Cada función reproduce paso a paso el código que generó la entrega oficial (el origen está en cada sección).
Ninguna variable usa el target ni el off-block real, así que existen igual en 2025 y en 2026.

Se construyen en este orden, porque unas leen las anteriores:

    base -> extras_v4 -> metar -> combinado -> meteo -> trafico -> puerta -> alineadas -> relojes
"""
import gc

import duckdb
import numpy as np
import pandas as pd

from prc import config as C
from prc import extras_v4 as X4
from prc import metar as MT

from . import config as K
from .utiles import guardar_json, log

ID = K.ID


def _sin_barra(con):
    """Apaga la barra de progreso de DuckDB (en Jupyter sin ipywidgets no se puede y se deja como está)."""
    try:
        con.execute("SET enable_progress_bar=false")
    except duckdb.InvalidInputException:
        pass


_sin_barra(duckdb)   # conexión por defecto que usa prc.extras_v4


def _segundos(serie):
    """Instantes como segundos epoch en float, con NaN donde falta el dato."""
    v = serie.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / 1e9
    v[serie.isna().to_numpy()] = np.nan
    return v


def _duckdb(threads=None, memory_limit=None):
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    _sin_barra(con)
    if threads:
        con.execute(f"SET threads={threads}")
    if memory_limit:
        con.execute(f"SET memory_limit='{memory_limit}'")
    K.TMP_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{K.TMP_DIR.as_posix()}'")
    return con


# =================================================================================================
# 1. Base: ventanas de tráfico, cruces, hora local y relojes del plan
#    Origen: notebooks/03_features_entrenamiento.ipynb (§2a y §2b) -> data/features/train_2025|ranking_2026
# =================================================================================================
VENTANAS_MIN = (15, 30, 60)       # despegues del aeropuerto y de la propia pista
VENTANA_ENTORNO_MIN = 30          # aterrizajes, entradas en calzos y pistas activas
MESES_TEST = ["2025-01", "2025-07"]
MESES_VAL = ["2025-11", "2025-12"]


def _marco(minutos: int) -> str:
    return f"RANGE BETWEEN {minutos * 60} PRECEDING AND {minutos * 60} FOLLOWING"


_APT = "PARTITION BY airport, mes_utc ORDER BY t"
_PISTA = "PARTITION BY airport, mes_utc, RUNWAY_mvt ORDER BY t"
_E = VENTANA_ENTORNO_MIN
SQL_COLS_AEROPUERTO = ",\n".join(
    [f"count(*) FILTER (WHERE tipo = 'DEP') OVER ({_APT} {_marco(v)}) AS dep_apt_{v}" for v in VENTANAS_MIN]
    + [f"count(*) FILTER (WHERE tipo = 'ARR') OVER ({_APT} {_marco(_E)}) AS arr_apt_{_E}",
       f"count(*) FILTER (WHERE tipo = 'INB') OVER ({_APT} {_marco(_E)}) AS inblock_apt_{_E}",
       f"count(DISTINCT RUNWAY_mvt) OVER ({_APT} {_marco(_E)}) AS n_pistas_activas"])
SQL_COLS_PISTA = ",\n".join(
    [f"count(*) OVER ({_PISTA} {_marco(v)}) AS dep_pista_{v}" for v in VENTANAS_MIN]
    + [f"t - lag(t) OVER ({_PISTA}, MVT_ID_mvt) AS hueco_anterior_s",
       f"lag(WK_TBL_CAT_flt) OVER ({_PISTA}, MVT_ID_mvt) AS estela_anterior"])

CATEGORICAS_BASE = ["airport", "hora_local_cat", "tipo_avion", "estela", "operador", "segmento", "tipo_vuelo",
                    "regla_vuelo", "destino", "apt_stand", "apt_pista", "combo", "apt_dique_pista",
                    "apt_operador", "estela_anterior"]
NUMERICAS_VENTANA = ["dep_apt_15", "dep_apt_30", "dep_apt_60", "arr_apt_30", "inblock_apt_30",
                     "n_pistas_activas", "dep_pista_15", "dep_pista_30", "dep_pista_60", "hueco_anterior_s"]
NUMERICAS_BASE = ["hora_local", "minuto_dia_local", "dia_semana_local", "fin_de_semana", "sin_minuto_dia",
                  "cos_minuto_dia", "taxi_nm_sec", "tiene_aobt3", "despegue_menos_sched_sec",
                  "despegue_menos_eobt1_sec", "despegue_menos_lobt_sec", "eobt1_menos_sched_sec",
                  "duracion_plan_sec", "desvio", *NUMERICAS_VENTANA]
APOYO = [ID, "target", "mes_utc", "apto_entrenamiento", "sin_flight_id"]


def ventanas(con, dep_file, arr_file) -> pd.DataFrame:
    """Recuentos de tráfico alrededor de cada despegue (ventanas centradas) y hueco con el anterior en pista."""
    dep, arr = dep_file.as_posix(), arr_file.as_posix()
    return con.sql(f"""
        WITH eventos AS (
            SELECT MVT_ID_mvt, airport, mes_utc, RUNWAY_mvt, 'DEP' AS tipo, epoch(MVT_TIME_UTC_mvt) AS t
            FROM read_parquet('{dep}')
            UNION ALL
            SELECT NULL, airport, mes_utc, NULL, 'ARR', epoch(MVT_TIME_UTC_mvt) FROM read_parquet('{arr}')
            UNION ALL
            SELECT NULL, airport, mes_utc, NULL, 'INB', epoch(BLOCK_TIME_UTC_mvt) FROM read_parquet('{arr}')
            WHERE BLOCK_TIME_UTC_mvt IS NOT NULL
        ),
        aeropuerto AS (SELECT MVT_ID_mvt, tipo, {SQL_COLS_AEROPUERTO} FROM eventos),
        pista AS (
            SELECT MVT_ID_mvt, {SQL_COLS_PISTA}
            FROM (SELECT *, epoch(MVT_TIME_UTC_mvt) AS t FROM read_parquet('{dep}') WHERE RUNWAY_mvt IS NOT NULL)
        )
        SELECT a.* EXCLUDE (tipo), p.* EXCLUDE (MVT_ID_mvt)
        FROM aeropuerto a LEFT JOIN pista p USING (MVT_ID_mvt)
        WHERE a.tipo = 'DEP'
    """).df()


def _diferencia(a, b):
    return (a - b).dt.total_seconds()


def construir_base(df: pd.DataFrame) -> pd.DataFrame:
    """Variables directas por fila: categóricas, cruces, hora local, relojes NM y del plan, ventanas."""
    f = pd.DataFrame(index=df.index)
    f[ID] = df[ID]
    f["target"] = df[K.TARGET].astype("float64")
    f["mes_utc"] = df["mes_utc"]
    f["apto_entrenamiento"] = df["apto_entrenamiento"]
    f["sin_flight_id"] = df["f_sin_flight_id"]
    # Directas
    f["airport"] = df["airport"]
    f["hora_local_cat"] = df["hora_local"].astype("str")
    f["tipo_avion"] = df["AIRCRAFT_TYPE_mvt"].fillna(df["AIRCRAFT_TYPE_flt"])
    f["estela"] = df["WK_TBL_CAT_flt"]
    f["operador"] = df["AIRCRAFT_OPERATOR_flt"]
    f["segmento"] = df["MARKET_SEGMENT_flt"]
    f["tipo_vuelo"] = df["FLIGHT_TYPE_flt"]
    f["regla_vuelo"] = df["FLIGHT_RULE_mvt"]
    f["destino"] = df["ADES_mvt"]
    # Cruces: nulos si falta cualquier parte
    f["apt_stand"] = df["airport"] + "|" + df["STAND_mvt"]
    f["apt_pista"] = df["airport"] + "|" + df["RUNWAY_mvt"]
    f["combo"] = df["airport"] + "|" + df["STAND_mvt"] + "|" + df["RUNWAY_mvt"]
    f["apt_dique_pista"] = df["airport"] + "|" + df["dique"] + "|" + df["RUNWAY_mvt"]
    f["apt_operador"] = df["airport"] + "|" + df["AIRCRAFT_OPERATOR_flt"]
    # Tiempo local
    minuto_dia = df["mvt_time_local"].dt.hour * 60 + df["mvt_time_local"].dt.minute
    f["hora_local"] = df["hora_local"]
    f["minuto_dia_local"] = minuto_dia
    f["dia_semana_local"] = df["dia_semana_local"]
    f["fin_de_semana"] = (df["dia_semana_local"] >= 6).astype("int8")
    f["sin_minuto_dia"] = np.sin(2 * np.pi * minuto_dia / 1440)
    f["cos_minuto_dia"] = np.cos(2 * np.pi * minuto_dia / 1440)
    # Off-block según el Network Manager
    f["taxi_nm_sec"] = _diferencia(df["MVT_TIME_UTC_mvt"], df["AOBT_3_flt"])
    f["tiene_aobt3"] = df["AOBT_3_flt"].notna().astype("int8")
    # Plan y retraso
    f["despegue_menos_sched_sec"] = _diferencia(df["MVT_TIME_UTC_mvt"], df["SCHED_TIME_UTC_mvt"])
    f["despegue_menos_eobt1_sec"] = _diferencia(df["MVT_TIME_UTC_mvt"], df["EOBT_1_flt"])
    f["despegue_menos_lobt_sec"] = _diferencia(df["MVT_TIME_UTC_mvt"], df["LOBT_flt"])
    f["eobt1_menos_sched_sec"] = _diferencia(df["EOBT_1_flt"], df["SCHED_TIME_UTC_mvt"])
    f["duracion_plan_sec"] = _diferencia(df["ARVT_1_flt"], df["EOBT_1_flt"])
    conocido = df["ADES_FILED_flt"].notna() & df["ADES_flt"].notna()
    f["desvio"] = (df["ADES_FILED_flt"] != df["ADES_flt"]).astype("float64").where(conocido)
    # Ventanas: float64 en los dos años, los recuentos por pista traen nulos
    for col in NUMERICAS_VENTANA:
        f[col] = df[col].astype("float64")
    f["estela_anterior"] = df["estela_anterior"]
    return f[APOYO + CATEGORICAS_BASE + NUMERICAS_BASE]


def base(anio: int) -> pd.DataFrame:
    con = _duckdb()
    dep = pd.read_parquet(K.interim("dep", anio))
    dep = dep.merge(ventanas(con, K.interim("dep", anio), K.interim("arr", anio)), on=ID, how="left",
                    validate="one_to_one")
    assert dep["dep_apt_15"].notna().all(), "salidas sin ventana"
    f = construir_base(dep)
    if anio == 2025:
        # División histórica por mes UTC; viaja en el fichero de 2025 como en el original.
        f["split"] = np.select([f["mes_utc"].isin(MESES_TEST), f["mes_utc"].isin(MESES_VAL)], ["test", "val"],
                               "train")
    con.close()
    return f


# =================================================================================================
# 2. Extras v4: cola de salidas, vecinos NM, ritmo de pista, stand y operador recuperado
#    Origen: prc/extras_v4.py vía notebooks/07_modelo_v4.ipynb -> data/features/extras_v4_<año>
# =================================================================================================
TABLA_RECUPERACION = K.CONFIG_DIR / "recuperacion_vuelos.parquet"
_RECUPERABLES = {"op": "AIRCRAFT_OPERATOR_flt", "seg": "MARKET_SEGMENT_flt", "wk": "WK_TBL_CAT_flt"}


def tabla_recuperacion(ref_file) -> pd.DataFrame:
    """Operador, segmento y estela más frecuentes por (aeropuerto, nº de vuelo) en las salidas con plan de 2025.

    Cambia respecto al original: `prc/extras_v4.py` usaba `mode()` de DuckDB, que desempata al azar (34, 243 y
    189 grupos empatados; 40 filas de 2025 y 3 de 2026 dependían de ello). Aquí se desempata por orden
    alfabético. La tabla congelada de la entrega está en `configuracion/`.
    """
    ref = pd.read_parquet(ref_file, columns=["airport", "FLIGHT_mvt", "FLIGHT_ID_mvt", *_RECUPERABLES.values()])
    ref = ref[ref.FLIGHT_ID_mvt.notna() & ref.FLIGHT_mvt.notna()]
    tabla = ref[["airport", "FLIGHT_mvt"]].drop_duplicates()
    for nombre, col in _RECUPERABLES.items():
        c = ref.dropna(subset=[col]).groupby(["airport", "FLIGHT_mvt", col]).size().rename("n").reset_index()
        c = c.sort_values(["airport", "FLIGHT_mvt", "n", col], ascending=[True, True, False, True])
        tabla = tabla.merge(c.drop_duplicates(["airport", "FLIGHT_mvt"])[["airport", "FLIGHT_mvt", col]]
                            .rename(columns={col: nombre}), on=["airport", "FLIGHT_mvt"], how="left")
    return tabla.sort_values(["airport", "FLIGHT_mvt"]).reset_index(drop=True)


def recuperadas(dep_file, tabla: pd.DataFrame) -> pd.DataFrame:
    """Rellena operador, segmento y estela de las salidas sin plan de vuelo con la tabla de recuperación."""
    con = duckdb.connect()
    con.register("ref", tabla)
    out = con.sql(f"""
        SELECT d.MVT_ID_mvt,
               coalesce(d.AIRCRAFT_OPERATOR_flt, ref.op) AS operador_rec,
               d.airport || '|' || coalesce(d.AIRCRAFT_OPERATOR_flt, ref.op) AS apt_operador_rec,
               coalesce(d.MARKET_SEGMENT_flt, ref.seg) AS segmento_rec,
               coalesce(d.WK_TBL_CAT_flt, ref.wk) AS estela_rec
        FROM '{dep_file.as_posix()}' d LEFT JOIN ref USING (airport, FLIGHT_mvt)""").df()
    con.close()
    return out


def extras_v4(anio: int) -> pd.DataFrame:
    """Mismas variables que `prc.extras_v4.construir`, con la recuperación por nº de vuelo determinista."""
    if TABLA_RECUPERACION.exists():
        tabla = pd.read_parquet(TABLA_RECUPERACION)
    else:
        tabla = tabla_recuperacion(K.interim("dep", 2025))
        tabla.to_parquet(TABLA_RECUPERACION, index=False)
    dep_file, arr_file = K.interim("dep", anio), K.interim("arr", anio)
    dep = duckdb.sql(f"""SELECT MVT_ID_mvt, airport, RUNWAY_mvt, epoch(MVT_TIME_UTC_mvt) AS t_off,
                         epoch(AOBT_3_flt) AS t_aobt FROM '{dep_file.as_posix()}'""").df()
    arr = duckdb.sql(f"SELECT airport, epoch(MVT_TIME_UTC_mvt) AS t_land FROM '{arr_file.as_posix()}'").df()
    return (X4._cola(dep, arr)
            .merge(X4._vecinos(dep_file), on=ID, validate="one_to_one")
            .merge(X4._ritmo(dep_file), on=ID, validate="one_to_one")
            .merge(X4._stand(dep_file, arr_file), on=ID, validate="one_to_one")
            .merge(recuperadas(dep_file, tabla), on=ID, validate="one_to_one"))


# =================================================================================================
# 3. METAR v1 (LVP, helada, dinámica) cruzado con el despegue
#    Origen: prc/metar.py vía notebooks/11_modelo_metar.ipynb -> data/external/metar_obs_<año>, metar_<año>
# =================================================================================================
def metar_obs(anio: int) -> pd.DataFrame:
    return MT.anadir_dinamica(MT.anadir_banderas(MT.cargar(K.METAR_DIRS[anio])))


def metar(anio: int, obs: pd.DataFrame) -> pd.DataFrame:
    return MT.features_por_movimiento(K.interim("dep", anio), obs)


# =================================================================================================
# 4. Tabla combinada: base + extras v4 + METAR v1 + relojes derivados del fichero limpio
#    Origen: experimentos/arquitectura_simple_20261001/lab.py:prepare -> cache/combined_<año>
# =================================================================================================
def combinado(anio: int) -> pd.DataFrame:
    con = _duckdb()
    fb, fx, fm, fd = (K.feature("base", anio).as_posix(), K.feature("extras_v4", anio).as_posix(),
                      K.feature("metar", anio).as_posix(), K.interim("dep", anio).as_posix())
    frame = con.sql(f"""SELECT f.*, x.* EXCLUDE(MVT_ID_mvt), m.* EXCLUDE(MVT_ID_mvt),
      d.MVT_TIME_UTC_mvt AS movement_time,d.fecha_local AS date,
      CAST(substr(f.mes_utc,6,2) AS INT) AS "month",
      regexp_extract(d.FLIGHT_mvt,'^([A-Za-z]+)',1) flight_prefix,
      d.airport||'|'||d.FLIGHT_mvt AS flight_key,
      try_cast(regexp_extract(d.FLIGHT_mvt,'([0-9]+)',1) AS INT) flight_number,
      epoch(d.MVT_TIME_UTC_mvt-d.IOBT_flt) despegue_menos_iobt,
      epoch(d.ARVT_3_flt-d.ARVT_1_flt) arr_delay_nm,
      d.f_tipo_avion_distinto::INT aircraft_mismatch,d.f_adep_distinto::INT airport_mismatch,
      second(d.MVT_TIME_UTC_mvt) takeoff_second,second(d.SCHED_TIME_UTC_mvt) sched_second,
      second(d.AOBT_3_flt) nm_second,hour(d.SCHED_TIME_UTC_mvt) sched_hour,
      date_diff('day',d.AOBT_3_flt,d.MVT_TIME_UTC_mvt) nm_day_gap,
      sin(2*pi()*dayofyear(d.MVT_TIME_UTC_mvt)/365.25) year_sin,
      cos(2*pi()*dayofyear(d.MVT_TIME_UTC_mvt)/365.25) year_cos
      FROM '{fb}' f
      JOIN '{fx}' x USING(MVT_ID_mvt)
      JOIN '{fm}' m USING(MVT_ID_mvt)
      JOIN '{fd}' d USING(MVT_ID_mvt)
      ORDER BY f.MVT_ID_mvt""").df()
    con.close()
    nm = frame.taxi_nm_sec
    frame["nm_missing"] = nm.isna().astype("int8")
    frame["nm_abnormal"] = (nm.notna() & ~nm.between(0, 7200)).astype("int8")
    for col, otro in [("nm_minus_lobt", "despegue_menos_lobt_sec"), ("nm_minus_eobt", "despegue_menos_eobt1_sec"),
                      ("nm_minus_sched", "despegue_menos_sched_sec")]:
        frame[col] = frame[otro] - nm
    frame["lobt_minus_sched"] = frame.despegue_menos_sched_sec - frame.despegue_menos_lobt_sec
    relojes = frame[["taxi_nm_sec", "despegue_menos_sched_sec", "despegue_menos_eobt1_sec", "despegue_menos_lobt_sec"]]
    frame["source_min"] = relojes.min(axis=1)
    frame["source_max"] = relojes.max(axis=1)
    frame["source_spread"] = frame.source_max - frame.source_min
    frame["source_above_2h"] = (relojes > 7200).sum(axis=1).astype("int8")
    assert frame[ID].is_unique
    return frame


# =================================================================================================
# 5. METAR v2: parser corregido (nieve, engelante, techo) y viento sobre la pista
#    Origen: experimentos/criba_peak_metar_20261002/weather.py -> cache/weather_obs|weather_features_<año>
# =================================================================================================
import re  # noqa: E402  (solo lo usa el parser)


def parsear_metar(text):
    b = MT._cuerpo(text)
    tokens = b.split()
    phenomena = []
    for token in tokens:
        t = token.lstrip("+-")
        if re.fullmatch(r"(?:VC)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS)*", t) and t:
            phenomena.append(t)
    # Grupos completos: buscar "SG" dentro de "TSGR" confunde granizo con nieve.
    codes = set()
    descriptors = set()
    for t in phenomena:
        if t.startswith("VC"):
            t = t[2:]
        if t[:2] in {"MI", "PR", "BC", "DR", "BL", "SH", "TS", "FZ"}:
            descriptors.add(t[:2])
            t = t[2:]
        codes.update(t[i:i + 2] for i in range(0, len(t), 2))
    w = re.search(r"\b(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?(KT|MPS)\b", b)
    direction = np.nan
    speed = np.nan
    gust = np.nan
    variable = np.nan
    if w:
        variable = float(w[1] == "VRB")
        direction = np.nan if variable else float(w[1])
        factor = 1 if w[4] == "KT" else 1.94384449
        speed = float(w[2]) * factor
        gust = float(w[3]) * factor if w[3] else speed
    clouds = re.findall(r"\b(FEW|SCT|BKN|OVC)(\d{3}|///)(CB|TCU)?\b", b)
    ceiling = [int(h) * 100 for cover, h, _ in clouds if cover in ("BKN", "OVC") and h != "///"]
    vv = re.search(r"\bVV(\d{3}|///)", b)
    if vv and vv[1] != "///":
        ceiling.append(int(vv[1]) * 100)
    unknown = bool(re.search(r"(?:BKN|OVC|VV)///", b))
    clear = bool(re.search(r"\b(CAVOK|NSC|NCD|SKC|CLR)\b", b))
    ceil = min(ceiling) if ceiling else (np.nan if unknown or not (clouds or clear) else 20000.)
    temp, dew = MT._temp_rocio(b)
    vis = MT._visibilidad_m(b)
    if vis == 9999:
        vis = 10000.
    out = {"visibility_m": vis, "ceiling_ft": ceil, "ceiling_unknown": float(unknown), "cavok": float("CAVOK" in tokens),
           "cover_oktas": max([{"FEW": 2, "SCT": 4, "BKN": 7, "OVC": 8}[c] for c, _, _ in clouds],
                              default=0 if clear else np.nan),
           "cb": float(any(c == "CB" for _, _, c in clouds)), "snow": float(bool(codes & {"SN", "SG", "IC", "PL"})),
           "freezing": float("FZ" in descriptors),
           "precip": float(bool(codes & {"RA", "DZ", "SN", "SG", "IC", "PL", "GR", "GS", "UP"})),
           "obscuration": float(bool(codes & {"BR", "FG", "FU", "VA", "DU", "SA", "HZ", "PY"})),
           "ts": float("TS" in descriptors), "other": float(bool(codes & {"PO", "SQ", "FC", "SS", "DS"})),
           "wind_kt": speed, "gust_kt": gust, "wind_dir": direction, "wind_variable": variable,
           "temp_c": temp, "dew_spread": temp - dew, "rvr_m": MT._rvr_min_m(b)}
    return {"wx_" + k: v for k, v in out.items()}


def comprobar_parser_metar():
    """Casos de prueba del parser v2 (los mismos del original)."""
    a = parsear_metar("20010G20KT 3000 -SHSN BKN010CB M02/M03 Q1000 TEMPO TSRA BKN001")
    assert a["wx_snow"] == 1 and a["wx_cb"] == 1 and a["wx_ceiling_ft"] == 1000 and a["wx_ts"] == 0
    assert parsear_metar("VRB03MPS 9999 RASN SCT020")["wx_snow"] == 1
    assert abs(parsear_metar("VRB03MPS CAVOK 02/01")["wx_wind_kt"] - 5.83153347) < 1e-6
    assert np.isnan(parsear_metar("00000KT 0200 VV/// 00/00")["wx_ceiling_ft"])
    assert parsear_metar("18005KT CAVOK 10/05")["wx_ceiling_ft"] == 20000
    assert parsear_metar("+TSGRRA BKN///CB 26/18")["wx_snow"] == 0
    assert parsear_metar("+TSGRRA BKN///CB 26/18")["wx_ts"] == 1
    assert parsear_metar("TSSN OVC003 M02/M03")["wx_snow"] == 1
    assert parsear_metar("TSSN OVC003 M02/M03")["wx_other"] == 0


def meteo_obs(anio: int) -> pd.DataFrame:
    """Una fila por (aeropuerto, parte), deduplicada; ignora las carpetas de derivados."""
    rows = {}
    for f in sorted(K.METAR_DIRS[anio].rglob("*.txt")):
        if any(x in str(f).upper() for x in ["AFECTADOS", "RELEVANTES", "CONDICIONES"]):
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = MT.LINEA.match(line.strip())
            if m and m[4].strip() != "NIL":
                rows[(m[3], m[1])] = m[4]
    keys = list(rows)
    d = pd.DataFrame([parsear_metar(rows[k]) for k in keys])
    d["airport"] = [k[0] for k in keys]
    d["valid"] = pd.to_datetime([k[1] for k in keys], format="%Y%m%d%H%M", utc=True)
    return d.sort_values("valid").reset_index(drop=True)


def meteo(anio: int, obs: pd.DataFrame) -> pd.DataFrame:
    """Último parte anterior al despegue (máx. 2 h), con viento de cara y cruzado sobre la pista."""
    d = pd.read_parquet(K.interim("dep", anio), columns=[ID, "airport", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
    d = d.rename(columns={"MVT_TIME_UTC_mvt": "time"}).sort_values("time")
    z = pd.merge_asof(d, obs.sort_values("valid"), left_on="time", right_on="valid", by="airport",
                      direction="backward", tolerance=pd.Timedelta("2h"))
    z["wx_missing"] = z.valid.isna().astype("float32")
    z["wx_age_min"] = (z.time - z.valid).dt.total_seconds() / 60
    heading = pd.to_numeric(z.RUNWAY_mvt.astype("string").str.extract(r"^(\d{2})", expand=False), errors="coerce") * 10
    angle = np.deg2rad(z.wx_wind_dir - heading)
    z["wx_headwind_kt"] = z.wx_wind_kt * np.cos(angle)
    z["wx_crosswind_kt"] = abs(z.wx_wind_kt * np.sin(angle))
    z["wx_gust_excess_kt"] = z.wx_gust_kt - z.wx_wind_kt
    cols = [c for c in z if c.startswith("wx_")]
    z[cols] = z[cols].astype("float32")
    return z[[ID, *cols]]


# =================================================================================================
# 6. Contexto de tráfico: exceso de taxi NM de los vecinos sobre la geometría y ocupación de la superficie
#    Origen: experimentos/mejora_v21_20261004/03_features_contexto.ipynb (2025) y
#            20_traslado_contexto_2026.ipynb (2026, geometría congelada en 2025) -> cache/traffic_*
# =================================================================================================
COLUMNAS_TRAFICO = ["ctx_nm_geometry", "ctx_geometry_support_log", "ctx_excess_apt_15", "ctx_excess_apt_45",
                    "ctx_excess_pista_15", "ctx_excess_pista_45", "ctx_excess_prefix_60", "ctx_excess_stand_180",
                    "ctx_excess_p90_pista_45", "ctx_excess_std_pista_45", "ctx_nm_missing_fraction_apt_45",
                    "ctx_count_stand_180", "ctx_count_prefix_60", "ctx_surface_apt_at_takeoff",
                    "ctx_pushes_apt_prev30", "ctx_surface_pista_at_takeoff", "ctx_pushes_pista_prev30"]


def trafico(anio: int) -> pd.DataFrame:
    """La geometría (mediana del taxi NM por combo/dique/pista/aeropuerto) se ajusta siempre con 2025.

    En 2026 se usa la de 2025 congelada, para no reajustar la referencia con dos meses. No mira el target.
    """
    con = _duckdb(threads=3, memory_limit="3GB")
    path = K.feature("combinado", anio).as_posix()
    geometria = K.feature("combinado", 2025).as_posix()
    con.execute("CREATE TABLE peer_data AS " + f"""
    WITH d AS (
      SELECT MVT_ID_mvt, airport, apt_pista, apt_stand, combo, apt_dique_pista, flight_prefix,
             epoch(movement_time) t,
             CASE WHEN taxi_nm_sec BETWEEN 0 AND 7200 THEN taxi_nm_sec END nm
      FROM '{path}'
    ), geometry AS (
      SELECT combo,apt_dique_pista,apt_pista,airport,
        CASE WHEN taxi_nm_sec BETWEEN 0 AND 7200 THEN taxi_nm_sec END nm
      FROM '{geometria}'
    ), c AS (SELECT combo,median(nm) med,count(nm) n FROM geometry GROUP BY combo),
    k AS (SELECT apt_dique_pista,median(nm) med,count(nm) n FROM geometry GROUP BY apt_dique_pista),
    p AS (SELECT apt_pista,median(nm) med,count(nm) n FROM geometry GROUP BY apt_pista),
    a AS (SELECT airport,median(nm) med FROM geometry GROUP BY airport),
    b AS (
      SELECT d.*,
        CASE WHEN c.n>=20 THEN c.med WHEN k.n>=100 THEN k.med WHEN p.n>=100 THEN p.med ELSE a.med END ctx_nm_geometry,
        ln(1+coalesce(c.n,0)) ctx_geometry_support_log
      FROM d LEFT JOIN c USING(combo) LEFT JOIN k USING(apt_dique_pista) LEFT JOIN p USING(apt_pista) LEFT JOIN a USING(airport)
    )
    SELECT *, nm-ctx_nm_geometry excess FROM b
    """)
    windows = {
        "apt_15": "PARTITION BY airport ORDER BY t RANGE BETWEEN 900 PRECEDING AND 900 FOLLOWING EXCLUDE CURRENT ROW",
        "apt_45": "PARTITION BY airport ORDER BY t RANGE BETWEEN 2700 PRECEDING AND 2700 FOLLOWING EXCLUDE CURRENT ROW",
        "pista_15": "PARTITION BY airport,apt_pista ORDER BY t RANGE BETWEEN 900 PRECEDING AND 900 FOLLOWING EXCLUDE CURRENT ROW",
        "pista_45": "PARTITION BY airport,apt_pista ORDER BY t RANGE BETWEEN 2700 PRECEDING AND 2700 FOLLOWING EXCLUDE CURRENT ROW",
        "prefix_60": "PARTITION BY airport,flight_prefix ORDER BY t RANGE BETWEEN 3600 PRECEDING AND 3600 FOLLOWING EXCLUDE CURRENT ROW",
        "stand_180": "PARTITION BY airport,apt_stand ORDER BY t RANGE BETWEEN 10800 PRECEDING AND 10800 FOLLOWING EXCLUDE CURRENT ROW"}
    expressions = [ID, "ctx_nm_geometry", "ctx_geometry_support_log"]
    for name, window in windows.items():
        expressions.append(f"median(excess) OVER ({window}) AS ctx_excess_{name}")
    expressions += [
        f"quantile_cont(excess,.9) OVER ({windows['pista_45']}) AS ctx_excess_p90_pista_45",
        f"stddev_pop(excess) OVER ({windows['pista_45']}) AS ctx_excess_std_pista_45",
        f"avg(CASE WHEN nm IS NULL THEN 1.0 ELSE 0.0 END) OVER ({windows['apt_45']}) AS ctx_nm_missing_fraction_apt_45",
        f"count(nm) OVER ({windows['stand_180']}) AS ctx_count_stand_180",
        f"count(nm) OVER ({windows['prefix_60']}) AS ctx_count_prefix_60"]
    features = con.sql("SELECT " + ", ".join(expressions) + " FROM peer_data").df()
    raw = con.sql("SELECT MVT_ID_mvt,airport,apt_pista,t,nm FROM peer_data ORDER BY MVT_ID_mvt").df()
    con.close()
    surface = pd.DataFrame({ID: raw[ID].to_numpy()})
    for label, keys in [("apt", ["airport"]), ("pista", ["airport", "apt_pista"])]:
        values = np.full(len(raw), np.nan, dtype="float32")
        pushes = np.full(len(raw), np.nan, dtype="float32")
        for _, g in raw.groupby(keys, dropna=True, observed=True):
            i = g.index.to_numpy()
            t = g.t.to_numpy()
            valid = g.nm.notna().to_numpy()
            off = np.sort(t[valid] - g.nm.to_numpy()[valid])
            end = np.sort(t[valid])
            values[i] = np.searchsorted(off, t, side="right") - np.searchsorted(end, t, side="right")
            # Excluir el propio pushback cuando cae en la ventana.
            own = valid & g.nm.between(0, 1800, inclusive="left").to_numpy()
            pushes[i] = np.searchsorted(off, t, side="right") - np.searchsorted(off, t - 1800, side="right") - own.astype(int)
        surface[f"ctx_surface_{label}_at_takeoff"] = values
        surface[f"ctx_pushes_{label}_prev30"] = pushes
    features = features.merge(surface, on=ID, validate="one_to_one")
    cols = [c for c in features if c != ID]
    features[cols] = features[cols].astype("float32")
    assert features[ID].is_unique and not np.isinf(features[cols].to_numpy()).any()
    del raw, surface
    gc.collect()
    return features


# =================================================================================================
# 7. Retención en puerta: vuelos que esperan en puerta tras su EOBT y demora reciente
#    Origen: experimentos/mejora_v21_20261006/research.py:gate_features -> cache/gate_<año>
# =================================================================================================
COLUMNAS_PUERTA = ["gate_pending_apt", "gate_pending_pista", "gate_recent_delay_mean_sec"]


def puerta(anio: int) -> pd.DataFrame:
    raw = pd.read_parquet(K.interim("dep", anio),
                          columns=[ID, "airport", "RUNWAY_mvt", "EOBT_1_flt", "AOBT_3_flt", "MVT_TIME_UTC_mvt"])
    raw = raw.sort_values(ID).reset_index(drop=True)
    push, expected, takeoff = [_segundos(raw[c]) for c in ["AOBT_3_flt", "EOBT_1_flt", "MVT_TIME_UTC_mvt"]]
    reliable = np.isfinite(push) & np.isfinite(takeoff) & (takeoff - push >= 0) & (takeoff - push <= 7200)
    pending_valid = reliable & np.isfinite(expected) & (push >= expected) & (push - expected <= 21600)
    f = raw[[ID]].copy()
    for label, keys in [("apt", ["airport"]), ("pista", ["airport", "RUNWAY_mvt"])]:
        values = np.full(len(raw), np.nan, dtype="float32")
        for _, g in raw.groupby(keys, observed=True, dropna=True):
            ix = g.index.to_numpy()
            valid = ix[pending_valid[ix]]
            starts, ends = np.sort(expected[valid]), np.sort(push[valid])
            v = np.searchsorted(starts, push[ix], side="right") - np.searchsorted(ends, push[ix], side="right")
            v = v.astype(float)
            v[~reliable[ix]] = np.nan
            values[ix] = v
        f[f"gate_pending_{label}"] = values
    delay = np.full(len(raw), np.nan, dtype="float32")
    for _, g in raw.groupby("airport", observed=True, dropna=True):
        ix = g.index.to_numpy()
        valid = ix[pending_valid[ix]]
        order = np.argsort(push[valid])
        valid = valid[order]
        t = push[valid]
        duration = push[valid] - expected[valid]
        csum = np.concatenate([[0], np.cumsum(duration)])
        # Estrictamente anterior: excluye el vuelo focal y todos los pushes simultáneos.
        right = np.searchsorted(t, push[ix], side="left")
        left = np.searchsorted(t, push[ix] - 1800, side="left")
        count = right - left
        avg = np.divide(csum[right] - csum[left], count, out=np.full(len(ix), np.nan), where=count > 0)
        avg[~reliable[ix]] = np.nan
        delay[ix] = avg
    f["gate_recent_delay_mean_sec"] = delay
    assert f[ID].is_unique and not np.isinf(f[COLUMNAS_PUERTA].to_numpy()).any()
    return f


# =================================================================================================
# 8. Variables alineadas: meteo en el push (AOBT_3) y su cambio hasta el despegue, llegadas recientes
#    Origen: experimentos/mejora_v26_normal_20261006/experiment.py:new_features -> cache/aligned_<año>
# =================================================================================================
PUSH_WEATHER = ["wx_visibility_m", "wx_ceiling_ft", "wx_snow", "wx_freezing", "wx_temp_c", "wx_dew_spread",
                "wx_wind_kt", "wx_gust_excess_kt", "wx_headwind_kt", "wx_crosswind_kt", "wx_missing", "wx_age_min"]
COLUMNAS_ALINEADAS = ["push_" + c for c in PUSH_WEATHER] + ["taxi_wx_visibility_change", "taxi_wx_ceiling_change",
                                                            "taxi_wx_temp_change", "arrivals_push_prev15",
                                                            "arrivals_takeoff_prev15"]


def alineadas(anio: int) -> pd.DataFrame:
    raw = pd.read_parquet(K.interim("dep", anio), columns=[ID, "airport", "RUNWAY_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt"])
    nm = (raw.MVT_TIME_UTC_mvt - raw.AOBT_3_flt).dt.total_seconds()
    good = nm.between(0, 7200)
    query = raw.loc[good].sort_values("AOBT_3_flt")
    obs = pd.read_parquet(K.feature("meteo_obs", anio)).sort_values("valid")
    z = pd.merge_asof(query, obs, left_on="AOBT_3_flt", right_on="valid", by="airport", direction="backward",
                      tolerance=pd.Timedelta("2h"))
    z["wx_missing"] = z.valid.isna().astype(float)
    z["wx_age_min"] = (z.AOBT_3_flt - z.valid).dt.total_seconds() / 60
    heading = pd.to_numeric(z.RUNWAY_mvt.astype("string").str.extract(r"^(\d{2})", expand=False), errors="coerce") * 10
    angle = np.deg2rad(z.wx_wind_dir - heading)
    z["wx_headwind_kt"] = z.wx_wind_kt * np.cos(angle)
    z["wx_crosswind_kt"] = abs(z.wx_wind_kt * np.sin(angle))
    z["wx_gust_excess_kt"] = z.wx_gust_kt - z.wx_wind_kt
    f = z[[ID, *PUSH_WEATHER]].rename(columns={c: "push_" + c for c in PUSH_WEATHER})
    takeoff = pd.read_parquet(K.feature("meteo", anio), columns=[ID, "wx_visibility_m", "wx_ceiling_ft", "wx_temp_c"])
    f = f.merge(takeoff, on=ID, validate="one_to_one")
    for suffix in ["visibility", "ceiling", "temp"]:
        c = {"visibility": "wx_visibility_m", "ceiling": "wx_ceiling_ft", "temp": "wx_temp_c"}[suffix]
        f[f"taxi_wx_{suffix}_change"] = f[c] - f["push_" + c]
    f = f.drop(columns=["wx_visibility_m", "wx_ceiling_ft", "wx_temp_c"])
    arr = pd.read_parquet(K.interim("arr", anio), columns=["airport", "MVT_TIME_UTC_mvt"])
    values = {c: np.full(len(raw), np.nan, dtype="float32") for c in ["arrivals_push_prev15", "arrivals_takeoff_prev15"]}
    for apt, g in raw.groupby("airport", observed=True):
        times = np.sort(_segundos(arr.loc[arr.airport.eq(apt), "MVT_TIME_UTC_mvt"]))
        ix = g.index.to_numpy()
        valid = good.iloc[ix].to_numpy()
        for c, clock in [("arrivals_push_prev15", "AOBT_3_flt"), ("arrivals_takeoff_prev15", "MVT_TIME_UTC_mvt")]:
            q = _segundos(g[clock])
            v = np.searchsorted(times, q, side="right") - np.searchsorted(times, q - 900, side="right")
            v = v.astype(float)
            v[~valid] = np.nan
            values[c][ix] = v
    f = raw[[ID]].merge(f, on=ID, how="left", validate="one_to_one")
    for c, v in values.items():
        f[c] = v
    f[COLUMNAS_ALINEADAS] = f[COLUMNAS_ALINEADAS].astype("float32")
    assert len(f) == len(raw) and f[ID].is_unique and not np.isinf(f[COLUMNAS_ALINEADAS].to_numpy()).any()
    return f


# =================================================================================================
# 9. Coherencia de relojes: segundos, minuto mod 5, duraciones y desacuerdos entre fuentes
#    Origen: experimentos/mejora_estructural_v26_20261006/correction.py:extra -> cache/clock_extra_<año>
# =================================================================================================
def relojes(anio: int) -> pd.DataFrame:
    columns = [ID, "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt", "IOBT_flt",
               "ARVT_1_flt", "ARVT_3_flt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt", "ADEP_mvt", "ADEP_flt",
               "ADES_mvt", "ADES_flt", "CALLSIGN_flt", "FLIGHT_mvt"]
    r = pd.read_parquet(K.interim("dep", anio), columns=columns)
    f = r[[ID]].copy()
    for name, c in [("mvt", "MVT_TIME_UTC_mvt"), ("sched", "SCHED_TIME_UTC_mvt"), ("nm", "AOBT_3_flt"),
                    ("lobt", "LOBT_flt"), ("eobt", "EOBT_1_flt"), ("iobt", "IOBT_flt")]:
        f["clock_" + name + "_second"] = r[c].dt.second.astype("float32")
        f["clock_" + name + "_minute5"] = (r[c].dt.minute % 5).astype("float32")
    for name, left, right in [("flight_plan", "ARVT_1_flt", "EOBT_1_flt"), ("flight_actual", "ARVT_3_flt", "AOBT_3_flt"),
                              ("arrival_to_takeoff", "ARVT_3_flt", "MVT_TIME_UTC_mvt"), ("iobt_nm", "AOBT_3_flt", "IOBT_flt")]:
        f["clock_" + name] = (r[left] - r[right]).dt.total_seconds().astype("float32")
    for name, a, b in [("aircraft", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"), ("adep", "ADEP_mvt", "ADEP_flt"),
                       ("ades", "ADES_mvt", "ADES_flt"), ("callsign", "FLIGHT_mvt", "CALLSIGN_flt")]:
        f["clock_" + name + "_mismatch"] = (r[a].fillna("") != r[b].fillna("")).astype("float32")
    lookup = dict(sched="SCHED_TIME_UTC_mvt", nm="AOBT_3_flt", eobt="EOBT_1_flt", iobt="IOBT_flt")
    for a, b in [("sched", "nm"), ("eobt", "nm"), ("iobt", "nm"), ("sched", "eobt")]:
        delta = (r[lookup[a]] - r[lookup[b]]).dt.total_seconds()
        f[f"clock_{a}_{b}_agreement60"] = (abs(delta) <= 60).where(delta.notna()).astype("float32")
        f[f"clock_{a}_{b}_abs"] = abs(delta).astype("float32")
    return f


# =================================================================================================
# Orquestación
# =================================================================================================
PASOS = ["base", "extras_v4", "metar_obs", "metar", "combinado", "meteo_obs", "meteo", "trafico", "puerta",
         "alineadas", "relojes"]


def construir(anio: int, paso: str, recalcular: bool = False) -> pd.DataFrame:
    """Construye un paso para un año y lo guarda en `Data/features/base/<paso>_<año>.parquet`.

    Si el fichero existe y `recalcular` es False, lo lee. Los pasos dependen de los anteriores (ver PASOS);
    `trafico` de 2026 necesita además `combinado_2025`.
    """
    destino = K.feature(paso, anio)
    if destino.exists() and not recalcular:
        return pd.read_parquet(destino)
    if paso == "metar":
        resultado = metar(anio, pd.read_parquet(K.feature("metar_obs", anio)))
    elif paso == "meteo":
        resultado = meteo(anio, pd.read_parquet(K.feature("meteo_obs", anio)))
    else:
        resultado = {"base": base, "extras_v4": extras_v4, "metar_obs": metar_obs, "combinado": combinado,
                     "meteo_obs": meteo_obs, "trafico": trafico, "puerta": puerta, "alineadas": alineadas,
                     "relojes": relojes}[paso](anio)
    destino.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_parquet(destino, index=False)
    log("Guardado", destino.name, f"{len(resultado):,} filas x {resultado.shape[1]} columnas")
    return resultado
