"""Rutas y constantes de las variables base (las 137 variables heredadas que usa el modelo).

Periodo a predecir con la variable de entorno PRC_PERIODO (igual que `prc/config.py` y `scripts/periodo.py`):
- "ranking" (por defecto): 2026 = enero + julio de `ranking.parquet` (344.841 salidas).
- "final": 2026 = enero, febrero, junio y julio de `final_ranking.parquet` (670.790 salidas). Las tablas de 2026 se
  escriben aparte (`Data/interim_final`, `Data/features_final/base`); las de 2025 son las mismas en los dos periodos.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
FINAL = os.environ.get("PRC_PERIODO", "ranking") == "final"

# --- Entradas (solo lectura) ----------------------------------------------------------------------
DATA_DIR = ROOT / "Data"
RAW_DIR = DATA_DIR / "raw"
SUBMITTING_FILE = DATA_DIR / "raw_final" / "final_submitting.parquet" if FINAL else RAW_DIR / "submitting.parquet"
METAR_DIRS = {2025: DATA_DIR / "metar" / "2025", 2026: DATA_DIR / "metar" / "2026"}
CONFIG_DIR = ROOT / "configuracion"                 # receta (lista y orden de variables) y tabla de recuperación
REFERENCIAS_DIR = CONFIG_DIR / "referencias"        # tabla P10, vocabulario de categóricas y calibración de V27
PREDICCION_V27 = DATA_DIR / "comparacion" / "prediccion_v27.parquet"    # solo para comparar
PARTICIONES_V27 = DATA_DIR / "comparacion" / "v27_particiones"            # solo para los experimentos

# --- Salidas --------------------------------------------------------------------------------------
INTERIM_DIR = DATA_DIR / ("interim_final" if FINAL else "interim")                 # salidas y llegadas limpias
FEATURES_DIR = DATA_DIR / ("features_final" if FINAL else "features") / "base"    # tablas de variables base
TMP_DIR = DATA_DIR / "tmp"                                                         # temporal de DuckDB


def _dir(base: Path, anio: int) -> Path:
    """2025 es igual en los dos periodos: en el final se lee de la carpeta normal."""
    if FINAL and anio == 2025:
        return Path(str(base).replace("interim_final", "interim").replace("features_final", "features"))
    return base


def interim(fase: str, anio: int) -> Path:
    """`fase` es 'dep' o 'arr'."""
    return _dir(INTERIM_DIR, anio) / f"{fase}_{anio}.parquet"


def feature(nombre: str, anio: int) -> Path:
    return _dir(FEATURES_DIR, anio) / f"{nombre}_{anio}.parquet"


# --- Reto -----------------------------------------------------------------------------------------
ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
N_FILAS_ENVIO = 670_790 if FINAL else 344_841
TEAM_NAME = "knowledgeable-alligator"
AEROPUERTOS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
ANIOS = (2025, 2026)                        # 2025 = entrenamiento, 2026 = vuelos a predecir

NM_MAX_SEC = 7200          # régimen normal: 0 <= despegue − AOBT_3 <= 7200 s
MARGEN_LOBT_SEC = 3606.0   # proyección final al intervalo (despegue − LOBT) ± 3606 s
SUELO_SEC = 60             # ninguna predicción por debajo de un minuto
