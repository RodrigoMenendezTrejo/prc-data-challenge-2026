"""Configuración compartida por todos los cuadernos: rutas, aeropuertos y constantes del reto."""

import os
from pathlib import Path

# --- Rutas -----------------------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PRC_DATA_DIR", PROJECT_DIR / "Data"))

RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
INTERIM_DIR = DATA_DIR / "interim"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

DOCS_DIR = PROJECT_DIR / "Documentacion"
FIGURES_DIR = DOCS_DIR / "figuras"

CREDENTIALS_FILE = PROJECT_DIR / "credentials.json"

# --- Ficheros originales ---------------------------------------------------------------------------
# Periodo a predecir (variable de entorno PRC_PERIODO): "ranking" = ene + jul 2026 (fase inicial) o
# "final" = ene, feb, jun y jul 2026 (final_ranking.parquet / final_submitting.parquet, fase final).
PERIODO = os.environ.get("PRC_PERIODO", "ranking")
FINAL = PERIODO == "final"
TRAINING_GLOB = (RAW_DIR / "training_*.parquet").as_posix()
RANKING_FILE = (DATA_DIR / "raw_final" / "final_ranking.parquet").as_posix() if FINAL else (RAW_DIR / "ranking.parquet").as_posix()
SUBMITTING_FILE = (DATA_DIR / "raw_final" / "final_submitting.parquet").as_posix() if FINAL else (RAW_DIR / "submitting.parquet").as_posix()

# --- Ficheros intermedios entre cuadernos ----------------------------------------------------------
DEP_2025_FILE = INTERIM_DIR / "dep_2025.parquet"
ARR_2025_FILE = INTERIM_DIR / "arr_2025.parquet"
DEP_2026_FILE = INTERIM_DIR / "dep_2026.parquet"
ARR_2026_FILE = INTERIM_DIR / "arr_2026.parquet"
CALIDAD_FILE = INTERIM_DIR / "calidad_datos.csv"

FEATURES_TRAIN_FILE = FEATURES_DIR / "train_2025.parquet"
FEATURES_RANKING_FILE = FEATURES_DIR / "ranking_2026.parquet"

EXPERIMENTOS_FILE = MODELS_DIR / "experimentos.csv"
REGISTRO_ENVIOS_FILE = SUBMISSIONS_DIR / "registro_envios.csv"

# --- Datos externos --------------------------------------------------------------------------------
# Indicador ATXOT publicado por la Performance Review Unit (PRU) de EUROCONTROL: taxi-out de referencia y
# adicional por aeropuerto y mes, en totales. Solo se usa para contrastar, nunca como feature.
PRU_ATXOT_URL = "https://www.eurocontrol.int/performance/data/download/xls/Taxi-Out_Additional_Time.xlsx"
PRU_ATXOT_FILE = EXTERNAL_DIR / "Taxi-Out_Additional_Time.xlsx"

# --- MinIO / envío ---------------------------------------------------------------------------------
MC_ALIAS = "dc26"
MC_ENDPOINT = "https://s3.opensky-network.org"
DATA_BUCKET = "prc-2026-datasets"
TEAM_NAME = "knowledgeable-alligator"
TEAM_BUCKET = f"prc-2026-{TEAM_NAME}"

# --- Reto ------------------------------------------------------------------------------------------
ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
N_RANKING_DEP = 670_790 if FINAL else 344_841

# Suelo de todas las predicciones. No hay techo: el RMSE lo deciden los taxi-out largos.
PRED_MIN_SEC = 60.0

# Valores que en los datos originales significan "desconocido" y se tratan como nulo.
TOKENS_NULOS = ("", "NA", "ZZZZ", "UNKNOWN")

# ICAO -> (IATA, nombre, huso IANA). El huso es IANA y no un desfase fijo: Estambul no cambia de hora.
AIRPORTS = {
    "EDDF": ("FRA", "Frankfurt Main", "Europe/Berlin"),
    "EDDM": ("MUC", "Munich", "Europe/Berlin"),
    "EGLL": ("LHR", "London Heathrow", "Europe/London"),
    "EHAM": ("AMS", "Amsterdam Schiphol", "Europe/Amsterdam"),
    "LEBL": ("BCN", "Barcelona El Prat", "Europe/Madrid"),
    "LEMD": ("MAD", "Madrid Barajas", "Europe/Madrid"),
    "LFPG": ("CDG", "Paris Charles de Gaulle", "Europe/Paris"),
    "LIRF": ("FCO", "Rome Fiumicino", "Europe/Rome"),
    "LSZH": ("ZRH", "Zurich", "Europe/Zurich"),
    "LTFM": ("IST", "Istanbul", "Europe/Istanbul"),
}

# Aeropuerto observado de un movimiento: el de salida si es DEP, el de llegada si es ARR.
AIRPORT_SQL = "CASE WHEN PHASE_mvt = 'DEP' THEN ADEP_mvt ELSE ADES_mvt END"

# Dique de un stand: sus letras iniciales ('A36' -> 'A'), o su primer carácter si empieza por número
# ('308' -> '3'). Agrupa stands vecinos, incluidos los que no existían en 2025. Nulo si no hay stand.
DIQUE_SQL = ("CASE WHEN regexp_matches(STAND_mvt, '^[A-Za-z]') "
             "THEN regexp_extract(STAND_mvt, '^[A-Za-z]+') ELSE substr(STAND_mvt, 1, 1) END")
