"""Acceso a datos: conexión DuckDB con vistas sobre los parquet originales, escritura del envío y bucket."""

import json
import time
from urllib.parse import urlsplit

import duckdb
import pandas as pd

from prc import config as C

_CON: duckdb.DuckDBPyConnection | None = None


def connect() -> duckdb.DuckDBPyConnection:
    """Conexión nueva con las vistas `training`, `ranking` y `submitting`.

    La sesión se fija en UTC: sin eso, `strftime` y las restas entre timestamps usan el huso del
    sistema operativo y el mismo código cuenta filas distintas en máquinas distintas.
    """
    con = duckdb.connect()
    con.sql("SET TimeZone = 'UTC'")
    con.sql("SET enable_progress_bar = false")
    con.sql(f"CREATE VIEW training   AS SELECT * FROM read_parquet('{C.TRAINING_GLOB}')")
    con.sql(f"CREATE VIEW ranking    AS SELECT * FROM read_parquet('{C.RANKING_FILE}')")
    con.sql(f"CREATE VIEW submitting AS SELECT * FROM read_parquet('{C.SUBMITTING_FILE}')")
    return con


def sql(query: str) -> pd.DataFrame:
    """Consulta puntual contra las tres vistas, reutilizando una conexión compartida."""
    global _CON
    if _CON is None:
        _CON = connect()
    return _CON.sql(query).df()


def load_training(phase: str | None = None, months: list[str] | None = None) -> pd.DataFrame:
    """Movimientos de 2025. `phase` es 'DEP' o 'ARR'; `months` como ['2025-01'] (mes UTC)."""
    clauses = []
    if phase:
        clauses.append(f"PHASE_mvt = '{phase}'")
    if months:
        lista = ", ".join(f"'{m}'" for m in months)
        clauses.append(f"strftime(MVT_TIME_UTC_mvt AT TIME ZONE 'UTC', '%Y-%m') IN ({lista})")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return sql(f"SELECT * FROM training {where}")


def load_ranking(phase: str | None = None) -> pd.DataFrame:
    """Movimientos de ene + jul 2026. Las ARR conservan sus tiempos; las DEP no tienen target."""
    where = f"WHERE PHASE_mvt = '{phase}'" if phase else ""
    return sql(f"SELECT * FROM ranking {where}")


def load_submitting() -> pd.DataFrame:
    """La plantilla de envío: 344.841 MVT_ID de salidas con el target vacío."""
    return sql("SELECT * FROM submitting")


def submission_path(version: int):
    """Nombre exigido por el reto: `<team-name>_v<entero incremental>.parquet`."""
    return C.SUBMISSIONS_DIR / f"{C.TEAM_NAME}_v{int(version)}.parquet"


def write_submission(preds: pd.DataFrame, version: int):
    """Valida las predicciones contra la plantilla y escribe el fichero de envío.

    `preds` necesita `MVT_ID_mvt` y `TAXITIME_SEC_mvt`. Devuelve la ruta escrita. Falla, sin escribir
    nada, si sobra o falta cualquier fila, porque el script de ranking rechaza el envío entero.
    """
    columnas = [C.ID, C.TARGET]
    if list(preds.columns) != columnas:
        raise ValueError(f"se esperaban exactamente {columnas}, llegó {list(preds.columns)}")
    if preds[C.ID].duplicated().any():
        raise ValueError(f"{int(preds[C.ID].duplicated().sum())} MVT_ID duplicados")
    if preds[C.TARGET].isna().any():
        raise ValueError(f"{int(preds[C.TARGET].isna().sum())} predicciones nulas")

    template = load_submitting()[[C.ID]]
    # La plantilla trae el ID como DOUBLE; en interim es entero. Se iguala antes de cruzar.
    preds = preds.assign(**{C.ID: preds[C.ID].astype("float64")})
    joined = template.merge(preds, on=C.ID, how="left")
    missing = int(joined[C.TARGET].isna().sum())
    if missing:
        raise ValueError(f"{missing} filas de la plantilla no tienen predicción")
    if len(joined) != len(preds):
        raise ValueError(f"el nº de filas cambió: {len(preds)} predicciones frente a {len(template)} en la plantilla")

    # Tipos de la plantilla: MVT_ID DOUBLE y TAXITIME INTEGER.
    joined[C.ID] = joined[C.ID].astype("float64")
    joined[C.TARGET] = joined[C.TARGET].round().astype("int32")

    C.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = submission_path(version)
    joined.to_parquet(path, index=False)
    return path


# --- Bucket del equipo -----------------------------------------------------------------------------
def cliente_bucket():
    """Cliente MinIO contra el servidor S3 del reto (`C.MC_ENDPOINT`) con las claves de `credentials.json`.

    Del json solo se usan `accessKey` y `secretKey`: su campo `url` es la consola web de MinIO desde la que se
    exportaron las claves, no el servidor S3, y el cliente la rechaza.
    """
    from minio import Minio  # solo hace falta para hablar con el bucket

    credenciales = json.loads(C.CREDENTIALS_FILE.read_text(encoding="utf-8"))
    servidor = urlsplit(C.MC_ENDPOINT)
    return Minio(servidor.netloc, access_key=credenciales["accessKey"], secret_key=credenciales["secretKey"],
                 secure=servidor.scheme == "https")


def ficheros_bucket(cliente) -> set[str]:
    """Nombres de los ficheros que hay en el bucket del equipo."""
    return {o.object_name for o in cliente.list_objects(C.TEAM_BUCKET, recursive=True)}


def resultado_envio(cliente, fichero: str) -> dict | None:
    """El `<fichero>_result.json` que deja el script de ranking junto a cada envío, o None si aún no está.

    Trae `status` ("Succeeded" si se puntuó), `used_pairs` (filas casadas con la verdad) y `score`: el RMSE del
    envío sobre enero y julio de 2026.
    """
    from minio.error import S3Error

    try:
        respuesta = cliente.get_object(C.TEAM_BUCKET, f"{fichero}_result.json")
    except S3Error as e:
        if e.code == "NoSuchKey":
            return None
        raise
    try:
        return json.loads(respuesta.read().decode("utf-8"))
    finally:
        respuesta.close()
        respuesta.release_conn()


def esperar_resultado(cliente, fichero: str, timeout_s: float, cada_s: float = 10) -> dict | None:
    """Consulta `resultado_envio` cada `cada_s` segundos hasta que aparece o pasa `timeout_s`."""
    fin = time.monotonic() + timeout_s
    while True:
        resultado = resultado_envio(cliente, fichero)
        if resultado is not None or time.monotonic() >= fin:
            return resultado
        time.sleep(cada_s)
