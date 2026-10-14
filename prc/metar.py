"""METAR: parser, banderas de LVP por aeropuerto y features por movimiento.

Los partes vienen de la descarga del IEM de Iowa State (`Data/metar/<año>/`), un fichero de texto por aeropuerto y
mes, con una línea por observación:

    202501010020 METAR EDDF 010020Z AUTO 20010KT 9999 OVC008 00/M01 Q1027 NOSIG=

Tres detalles que condicionan el módulo:

1. Cada fichero son 31 días desde el día 1, no el mes natural (febrero se mete en marzo, junio en julio...),
   así que hay que deduplicar por `(aeropuerto, sello)`.
2. EHAM emite 8.976 partes `NIL` (vacíos) en una serie paralela a la real. Se filtran.
3. La API no da el RVR como campo, solo dentro del texto, y los criterios de LVP de nueve de los diez
   aeropuertos están en RVR, así que hay que parsear el texto crudo.

Todo se ancla a la hora de despegue, que existe en 2026. Ninguna variable mira el off-block ni el target.
"""

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from prc import config as C

# --- Umbrales de LVP, uno por aeropuerto -----------------------------------------------------------
# Del documento operativo del reto. Cada aeropuerto tiene el suyo (Múnich activa a 1.000 m y Roma a 400), y una
# regla común sesgaría unos frente a otros. `rvr` en metros, `techo` en pies; basta con que se cumpla uno.
LVP_UMBRALES = {
    "EDDF": {"rvr": 600, "techo": 200},    # RVR <= 600 m o techo < 200 ft
    "EDDM": {"rvr": 1000, "techo": 300},   # el más permisivo de los diez
    "EGLL": {"rvr": 600, "techo": None},   # solo RVR
    "EHAM": {"rvr": 550, "techo": 200},    # fase B; las cuatro fases van aparte en `lvp_fase`
    "LEBL": {"rvr": 650, "techo": 250},    # simplificación: el real depende de la pista de llegada (600/800/650)
    "LEMD": {"rvr": 700, "techo": 300},
    "LFPG": {"rvr": 600, "techo": 200},    # simplificación: el real se activa por doblete de pistas
    "LIRF": {"rvr": 400, "techo": None},   # el más exigente
    "LSZH": {"rvr": 550, "techo": 200},
    "LTFM": {"rvr": 550, "techo": 200},    # el documento no da umbral numérico; se usa el de CAT II/III
}

# Fases de EHAM, que es el único que las publica escalonadas.
FASES_EHAM = [("A", 1500, 300), ("B", 550, 200), ("C", 350, None), ("D", 200, None)]

LINEA = re.compile(r"^(\d{12})\s+(METAR|SPECI)\s+(?:COR\s+|RTD\s+)?([A-Z]{4})\s+\d{6}Z\s+(.*?)=?\s*$")

NUMERICAS = ["visibilidad_m", "techo_ft", "rvr_min_m", "temp_c", "rocio_c", "viento_kt", "racha_kt",
             "antiguedad_parte_min", "minutos_en_lvp", "delta_visibilidad_30min", "delta_techo_30min"]
BANDERAS = ["lvp", "lvp_fase", "lvp_recien_activado", "helada", "nieve_engelante", "niebla",
            "tormenta", "precip_fuerte"]
TODAS = NUMERICAS + BANDERAS


# --- Parser ----------------------------------------------------------------------------------------
def _cuerpo(texto: str) -> str:
    """El cuerpo del parte, sin el bloque de tendencia: TEMPO/BECMG describen el futuro, no el presente."""
    return re.split(r"\b(?:TEMPO|BECMG|NOSIG|TREND)\b", texto)[0]


def _techo_ft(b: str) -> float:
    """Techo = capa BKN/OVC más baja, o visibilidad vertical. CAVOK/NSC/NCD/SKC/CLR = sin techo."""
    if re.search(r"\b(CAVOK|NSC|NCD|SKC|CLR)\b", b):
        return np.inf
    capas = [int(h) * 100 for _, h in re.findall(r"\b(BKN|OVC)(\d{3})\b", b)]
    vv = re.search(r"\bVV(\d{3})\b", b)
    if vv:
        capas.append(int(vv.group(1)) * 100)
    return min(capas) if capas else np.inf


def _visibilidad_m(b: str) -> float:
    """Visibilidad prevaleciente. CAVOK = 10 km. Se quitan antes viento, dirección variable y RVR."""
    if "CAVOK" in b:
        return 10000.0
    b = re.sub(r"\b(\d{3}|VRB)\d{2}(G\d{2,3})?(KT|MPS)\b", " ", b)
    b = re.sub(r"\b\d{3}V\d{3}\b", " ", b)
    b = re.sub(r"\bR\d{2}[LCR]?/\S+", " ", b)
    g = re.findall(r"(?<![\d/])\b(\d{4})\b(?![/\d])", b)
    return float(min(int(x) for x in g)) if g else np.nan


def _rvr_min_m(b: str) -> float:
    """RVR mínimo entre las pistas que lo reportan. Solo aparece en un 2 % de los partes."""
    v = [int(x) for _, x in re.findall(r"\bR(\d{2}[LCR]?)/[MP]?(\d{4})", b)]
    return float(min(v)) if v else np.nan


def _viento(b: str):
    m = re.search(r"\b(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT\b", b)
    if not m:
        return np.nan, np.nan, np.nan
    dir_ = np.nan if m.group(1) == "VRB" else float(m.group(1))
    return dir_, float(m.group(2)), float(m.group(3)) if m.group(3) else np.nan


def _temp_rocio(b: str):
    m = re.search(r"\b(M?\d{2})/(M?\d{2})\b", b)
    if not m:
        return np.nan, np.nan
    conv = lambda s: -float(s[1:]) if s.startswith("M") else float(s)
    return conv(m.group(1)), conv(m.group(2))


def parsear(texto: str) -> dict:
    """Un parte a campos. `texto` es el cuerpo completo, con tendencia incluida."""
    b = _cuerpo(texto)
    direccion, viento, racha = _viento(b)
    temp, rocio = _temp_rocio(b)
    return {
        "visibilidad_m": _visibilidad_m(b), "techo_ft": _techo_ft(b), "rvr_min_m": _rvr_min_m(b),
        "viento_dir": direccion, "viento_kt": viento, "racha_kt": racha,
        "temp_c": temp, "rocio_c": rocio,
        # El engelante y la nieve son los que marcan el deshielo; la niebla, la baja visibilidad.
        "nieve_engelante": bool(re.search(r"\b[-+]?(SN|SG|PL|GS|GR)\b|\bFZ(RA|DZ|FG)\b", b)),
        "niebla": bool(re.search(r"\b(BR|FG|BCFG|MIFG|FZFG|PRFG)\b", b)),
        "tormenta": bool(re.search(r"\bTS", b)),
        "precip_fuerte": bool(re.search(r"\+(RA|SN|SH|TS|DZ|GR)", b)),
    }


def cargar(carpeta_metar: str | Path) -> pd.DataFrame:
    """Lee todos los ficheros de `METARS 2025/`, deduplica y parsea. Una fila por (aeropuerto, instante)."""
    carpeta = Path(carpeta_metar)
    filas = {}
    for f in glob.glob(str(carpeta / "**" / "*.txt"), recursive=True):
        # Las carpetas de derivados ("AFECTADOS POR METEO", ficheros "RELEVANTES") repiten partes de los crudos
        # con cabeceras por medio, así que solo leemos los ficheros crudos.
        if "AFECTADOS" in f.upper() or "RELEVANTES" in f.upper() or "CONDICIONES" in f.upper():
            continue
        for linea in open(f, encoding="utf-8", errors="replace"):
            m = LINEA.match(linea.strip())
            if m and m.group(4).strip() != "NIL":        # EHAM emite miles de partes vacíos
                filas[(m.group(3), m.group(1))] = m.group(4)
    if not filas:
        raise FileNotFoundError(f"no se han encontrado partes en {carpeta}")
    df = pd.DataFrame([{"airport": a, "valid": pd.to_datetime(t, format="%Y%m%d%H%M", utc=True),
                        **parsear(cuerpo)} for (a, t), cuerpo in filas.items()])
    return df.sort_values(["airport", "valid"]).reset_index(drop=True)


# --- Banderas --------------------------------------------------------------------------------------
def _cumple(rvr, techo, umbral_rvr, umbral_techo) -> np.ndarray:
    """Un criterio de LVP: RVR por debajo del umbral, o techo por debajo. El RVR ausente no activa."""
    cond = np.zeros(len(rvr), dtype=bool)
    if umbral_rvr is not None:
        cond |= (rvr <= umbral_rvr).to_numpy(na_value=False)
    if umbral_techo is not None:
        cond |= (techo < umbral_techo).to_numpy(na_value=False)
    return cond


def anadir_banderas(met: pd.DataFrame) -> pd.DataFrame:
    """`lvp` con el umbral propio de cada aeropuerto, `lvp_fase` (solo EHAM) y `helada`."""
    m = met.copy()
    m["lvp"] = False
    m["lvp_fase"] = 0
    for apt, u in LVP_UMBRALES.items():
        sel = m["airport"] == apt
        if not sel.any():
            continue
        sub = m.loc[sel]
        m.loc[sel, "lvp"] = _cumple(sub["rvr_min_m"], sub["techo_ft"], u["rvr"], u["techo"])
        if apt == "EHAM":                                 # el único con fases publicadas
            fase = np.zeros(len(sub), dtype=int)
            for i, (_, r_u, t_u) in enumerate(FASES_EHAM, start=1):
                fase = np.where(_cumple(sub["rvr_min_m"], sub["techo_ft"], r_u, t_u), i, fase)
            m.loc[sel, "lvp_fase"] = fase
    # Deshielo: fenómeno engelante explícito, o temperatura baja con humedad alta (rocío pegado a la temperatura).
    m["helada"] = (m["nieve_engelante"]
                   | ((m["temp_c"] <= 3) & ((m["temp_c"] - m["rocio_c"]).abs() <= 2)))
    return m


def anadir_dinamica(met: pd.DataFrame) -> pd.DataFrame:
    """Cambios respecto al parte anterior: un cambio de régimen pesa más que el nivel absoluto.

    `minutos_en_lvp` acumula el tiempo en LVP: el LVP reduce la cadencia de la pista, así que la cola depende
    de cuánto tiempo lleva activo, no solo de que lo esté.
    """
    m = met.sort_values(["airport", "valid"]).copy()
    g = m.groupby("airport", sort=False)
    m["delta_visibilidad_30min"] = m["visibilidad_m"] - g["visibilidad_m"].shift()
    m["delta_techo_30min"] = m["techo_ft"].replace(np.inf, 20000) - g["techo_ft"].shift().replace(np.inf, 20000)
    m["lvp_recien_activado"] = m["lvp"] & ~g["lvp"].shift(fill_value=False)

    minutos = np.zeros(len(m))
    acumulado = {}
    prev_t = {}
    for i, (apt, t, on) in enumerate(zip(m["airport"], m["valid"], m["lvp"])):
        if not on:
            acumulado[apt] = 0.0
        else:
            paso = (t - prev_t[apt]).total_seconds() / 60 if apt in prev_t else 30.0
            acumulado[apt] = acumulado.get(apt, 0.0) + min(paso, 120.0)
        prev_t[apt] = t
        minutos[i] = acumulado[apt]
    m["minutos_en_lvp"] = minutos
    return m


# --- Cruce por movimiento --------------------------------------------------------------------------
def features_por_movimiento(dep_file, met: pd.DataFrame) -> pd.DataFrame:
    """Cruza cada salida con el último parte anterior a su despegue (ASOF hacia atrás).

    Se ancla al despegue porque en 2026 no hay off-block. Como el parte tiene una antigüedad mediana de 15 min y
    el taxi dura otros 15, el parte describe más o menos el momento del off-block, que es el que importa.
    """
    dep = pd.read_parquet(dep_file, columns=[C.ID, "airport", "MVT_TIME_UTC_mvt"])
    dep = dep.rename(columns={"MVT_TIME_UTC_mvt": "despegue"}).sort_values("despegue")
    cols = [c for c in met.columns if c not in ("airport", "valid")]
    out = pd.merge_asof(dep, met.sort_values("valid")[["airport", "valid"] + cols],
                        left_on="despegue", right_on="valid", by="airport", direction="backward")
    out["antiguedad_parte_min"] = (out["despegue"] - out["valid"]).dt.total_seconds() / 60
    # El techo infinito se codifica alto y finito: LightGBM no trata el infinito de forma útil.
    out["techo_ft"] = out["techo_ft"].replace(np.inf, 20000)
    for c in ("lvp", "lvp_recien_activado", "helada", "niebla", "tormenta", "precip_fuerte", "nieve_engelante"):
        if c in out:
            out[c] = out[c].fillna(False).astype("int8")
    return out[[C.ID] + [c for c in TODAS if c in out]]


def construir(dep_file, carpeta_metar, destino=None) -> pd.DataFrame:
    """Todo el camino: parsear, banderas, dinámica y cruce por movimiento."""
    met = anadir_dinamica(anadir_banderas(cargar(carpeta_metar)))
    fe = features_por_movimiento(dep_file, met)
    if destino is not None:
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        fe.to_parquet(destino, index=False)
    return fe
