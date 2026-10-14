"""Descarga de puntos ADS-B alrededor de los 10 aeropuertos desde el ClickHouse público de ADSB.exposed.

Solo se envían a la API la fecha, la ventana horaria y un recuadro geográfico por aeropuerto: ningún dato del reto.
Se guardan los puntos en tierra o por debajo de ALT_MAX pies dentro del recuadro, solo de adsb.lol (licencia abierta
ODbL; ADSB.exposed mezcla también airplanes.live, que no se usa). adsb.lol está en ADSB.exposed hasta abril de 2026;
los meses posteriores se sacan de su archivo de GitHub (`scripts/descarga_adsb_github.py`).

Uso:
    python scripts/descarga_adsb_exposed.py 2026-01-01 2026-01-31            # todos los días del rango
    python scripts/descarga_adsb_exposed.py 2025-01-01 2025-12-31 --cada 3   # un día de cada 3
    python scripts/descarga_adsb_exposed.py ... --aeropuertos EDDF,EDDM --hilos 3

Salida: Data/adsb/raw/<APT>/<YYYY-MM-DD>.parquet (un fichero por aeropuerto y día; se salta si ya existe).
Con --cada > 1 conviene --bloque 1 (los días no son consecutivos).
"""
import argparse
import datetime as dt
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Data" / "adsb" / "raw"
URL = "https://kvzqttvc2n.eu-west-1.aws.clickhouse-staging.com/?user=website"
ALT_MAX = 5000

# Punto de referencia del aeropuerto y semiamplitud del recuadro (grados de latitud, longitud).
AEROPUERTOS = {
    "EDDF": (50.0333, 8.5706, 0.045, 0.080),
    "EDDM": (48.3538, 11.7861, 0.040, 0.080),
    "EGLL": (51.4706, -0.4619, 0.030, 0.075),
    "EHAM": (52.3200, 4.7550, 0.065, 0.095),
    "LEBL": (41.2971, 2.0785, 0.035, 0.060),
    "LEMD": (40.4900, -3.5650, 0.060, 0.060),
    "LFPG": (49.0097, 2.5479, 0.035, 0.090),
    "LIRF": (41.8003, 12.2389, 0.050, 0.070),
    "LSZH": (47.4647, 8.5492, 0.035, 0.060),
    "LTFM": (41.2700, 28.7400, 0.055, 0.090),
}

COLS = ("time, icao, r, t, aircraft_flight, lat, lon, altitude, ground_speed, track_degrees, "
        "vertical_rate, source, data_source")


def consulta(apt: str, inicio: dt.datetime, fin: dt.datetime) -> str:
    lat, lon, dlat, dlon = AEROPUERTOS[apt]
    return (f"SELECT {COLS} FROM planes_mercator "
            f"WHERE lat BETWEEN {lat - dlat:.4f} AND {lat + dlat:.4f} "
            f"AND lon BETWEEN {lon - dlon:.4f} AND {lon + dlon:.4f} "
            f"AND time >= '{inicio:%Y-%m-%d %H:%M:%S}' AND time < '{fin:%Y-%m-%d %H:%M:%S}' "
            f"AND altitude <= {ALT_MAX} "
            f"AND data_source = 'adsb.lol' "
            f"FORMAT Parquet")


def pedir(sql: str, intentos=5) -> pa.Table | None:
    for k in range(intentos):
        try:
            r = requests.post(URL, data=sql.encode(), timeout=600)
            if r.status_code == 200:
                if not r.content:
                    return None
                return pq.read_table(io.BytesIO(r.content))
            msg = r.text[:300]
            # Resultado demasiado grande o tiempo excedido: el llamador divide la ventana.
            if any(s in msg for s in ("TOO_MANY_ROWS", "TOO_MANY_BYTES", "TIMEOUT_EXCEEDED", "MEMORY_LIMIT", "Limit for result")):
                raise OverflowError(msg)
            print(f"  error {r.status_code}: {msg}", file=sys.stderr)
        except OverflowError:
            raise
        except Exception as e:  # red
            print(f"  excepción {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        time.sleep(5 * (k + 1))
    raise RuntimeError("Fallo persistente de la API")


def descargar_ventana(apt, inicio, fin, nivel=0) -> list[pa.Table]:
    try:
        t = pedir(consulta(apt, inicio, fin))
        return [t] if t is not None and t.num_rows else []
    except OverflowError:
        if nivel > 4:
            raise
        medio = inicio + (fin - inicio) / 2
        return descargar_ventana(apt, inicio, medio, nivel + 1) + descargar_ventana(apt, medio, fin, nivel + 1)


def descargar_bloque(apt: str, dias: list) -> str:
    """Una consulta para varios días seguidos (el coste es por consulta, no por volumen); un fichero por día."""
    destinos = {d: OUT / apt / f"{d:%Y-%m-%d}.parquet" for d in dias}
    pendientes = [d for d, f in destinos.items() if not f.exists()]
    if not pendientes:
        return f"{apt} {dias[0]}..{dias[-1]} ya existe"
    (OUT / apt).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    inicio = dt.datetime.combine(min(pendientes), dt.time())
    fin = dt.datetime.combine(max(pendientes), dt.time()) + dt.timedelta(days=1)
    tablas = descargar_ventana(apt, inicio, fin)
    if tablas:
        tabla = pa.concat_tables(tablas, promote_options="default")
        fechas = pc.strftime(tabla["time"], format="%Y-%m-%d")
    total = 0
    for d in pendientes:
        if tablas:
            sub = tabla.filter(pc.equal(fechas, f"{d:%Y-%m-%d}"))
        else:
            sub = pa.table({"time": pa.array([], pa.timestamp("ms", tz="UTC"))})
        total += sub.num_rows
        tmp = destinos[d].with_suffix(".tmp")
        pq.write_table(sub, tmp, compression="zstd")
        tmp.replace(destinos[d])
    return f"{apt} {pendientes[0]}..{pendientes[-1]} {total:,} filas {time.time() - t0:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("desde")
    ap.add_argument("hasta")
    ap.add_argument("--cada", type=int, default=1)
    ap.add_argument("--desfase", type=int, default=0, help="primer día dentro del ciclo --cada")
    ap.add_argument("--aeropuertos", default=",".join(AEROPUERTOS))
    ap.add_argument("--hilos", type=int, default=3)
    ap.add_argument("--bloque", type=int, default=3, help="días por consulta")
    a = ap.parse_args()
    d0, d1 = dt.date.fromisoformat(a.desde), dt.date.fromisoformat(a.hasta)
    dias = [d0 + dt.timedelta(days=k) for k in range((d1 - d0).days + 1)][a.desfase::a.cada]
    bloques = [dias[k:k + a.bloque] for k in range(0, len(dias), a.bloque)]
    tareas = [(apt, b) for b in bloques for apt in a.aeropuertos.split(",")]
    print(f"{len(tareas)} consultas ({len(dias)} días x aeropuertos)", flush=True)
    with ThreadPoolExecutor(a.hilos) as ex:
        futs = {ex.submit(descargar_bloque, apt, b): (apt, b[0]) for apt, b in tareas}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print(f"FALLO {futs[f]}: {e}", flush=True)


if __name__ == "__main__":
    main()
