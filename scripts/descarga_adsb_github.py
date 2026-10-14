"""Extracción de adsb.lol (ODbL) desde su archivo oficial en GitHub (adsblol/globe_history_YYYY).

Cada día es un tar partido en trozos (.tar.aa, .tar.ab, ...) de ~4 GB en total. Se descarga en streaming, se
recorre el tar sin guardarlo en disco y de cada traza diaria por aeronave (traces/xx/trace_full_<icao>.json) se
conservan solo los puntos dentro del recuadro de alguno de los 10 aeropuertos y en tierra o por debajo de
ALT_MAX pies. Mismo formato que la extracción de ADSB.exposed (altitud 0 = en tierra), con data_source='adsb.lol'.

Uso: python scripts/descarga_adsb_github.py 2026-07-01 2026-07-31 [--procesos 6]
Salida: Data/adsb/raw_lol/<APT>/<YYYY-MM-DD>.parquet
"""
import argparse
import os
import datetime as dt
import gzip
import json
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from descarga_adsb_exposed import AEROPUERTOS, ALT_MAX  # noqa: E402

OUT = ROOT / "Data" / "adsb" / "raw_lol"   # con --variante distinta de prod: raw_lol_<variante>
API = "https://api.github.com/repos/adsblol/globe_history_{y}/releases/tags/{tag}"


class Encadenado:
    """Objeto tipo fichero que lee secuencialmente varias respuestas HTTP (los trozos del tar)."""

    def __init__(self, urls):
        self.urls = list(urls)
        self.r = None
        self.it = None
        self.buf = b""

    def _siguiente(self):
        if self.r is not None:
            self.r.close()
            self.r = None
        if not self.urls:
            self.it = None
            return False
        url = self.urls.pop(0)
        for k in range(5):
            try:
                self.r = requests.get(url, stream=True, timeout=120)
                self.r.raise_for_status()
                self.it = self.r.iter_content(chunk_size=4 << 20)
                return True
            except Exception as e:  # noqa: BLE001
                print("  reintento descarga", e, file=sys.stderr)
                time.sleep(5 * (k + 1))
        raise RuntimeError("descarga fallida")

    def read(self, n=-1):
        while (n < 0 or len(self.buf) < n):
            if self.it is None and not self._siguiente():
                break
            try:
                self.buf += next(self.it)
            except StopIteration:
                self.it = None
                if not self._siguiente():
                    break
        if n < 0:
            out, self.buf = self.buf, b""
        else:
            out, self.buf = self.buf[:n], self.buf[n:]
        return out


# prod (archivo principal), staging o mlatonly. Se pasa por variable de entorno para que la hereden los procesos hijos.
VARIANTE = os.environ.get("ADSB_LOL_VARIANTE", "prod")
if VARIANTE != "prod":
    OUT = ROOT / "Data" / "adsb" / f"raw_lol_{VARIANTE}"


def urls_dia(dia: dt.date) -> list[str]:
    tag = f"v{dia:%Y.%m.%d}-planes-readsb-{VARIANTE}-0"
    r = requests.get(API.format(y=dia.year, tag=tag), timeout=60)
    r.raise_for_status()
    assets = sorted(r.json()["assets"], key=lambda a: a["name"])
    return [a["browser_download_url"] for a in assets if ".tar" in a["name"]]


CAJAS = {apt: (lat - dlat, lat + dlat, lon - dlon, lon + dlon) for apt, (lat, lon, dlat, dlon) in AEROPUERTOS.items()}


def procesar_dia(dia: dt.date) -> str:
    destinos = {apt: OUT / apt / f"{dia:%Y-%m-%d}.parquet" for apt in AEROPUERTOS}
    if all(f.exists() for f in destinos.values()):
        return f"{dia} ya existe"
    t0 = time.time()
    filas = {apt: [] for apt in AEROPUERTOS}
    n_trazas = 0
    stream = Encadenado(urls_dia(dia))
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for m in tar:
            if not m.isfile() or "trace_full_" not in m.name:
                continue
            raw = tar.extractfile(m).read()
            try:
                raw = gzip.decompress(raw)
            except OSError:
                pass
            try:
                j = json.loads(raw)
            except ValueError:
                continue
            tr = j.get("trace")
            if not tr:
                continue
            n_trazas += 1
            lat = np.fromiter((p[1] for p in tr), float, len(tr))
            lon = np.fromiter((p[2] for p in tr), float, len(tr))
            for apt, (la0, la1, lo0, lo1) in CAJAS.items():
                dentro = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
                if not dentro.any():
                    continue
                base = float(j["timestamp"])
                for k in np.flatnonzero(dentro):
                    p = tr[k]
                    alt = 0 if p[3] == "ground" else (int(p[3]) if p[3] is not None else None)
                    if alt is None or alt > ALT_MAX:
                        continue
                    det = p[8] if len(p) > 8 and isinstance(p[8], dict) else None
                    filas[apt].append((int(round((base + p[0]) * 1000)), j.get("icao", m.name[-11:-5]), j.get("r") or "",
                                       j.get("t") or "", (det or {}).get("flight", "") or "", p[1], p[2], alt,
                                       p[4] if p[4] is not None else np.nan, p[5] if p[5] is not None else np.nan,
                                       p[7] if len(p) > 7 and p[7] is not None else 0,
                                       p[9] if len(p) > 9 and isinstance(p[9], str) else ""))
    total = 0
    for apt, rows in filas.items():
        cols = list(zip(*rows)) if rows else [[] for _ in range(12)]
        tabla = pa.table({
            "time": pa.array(cols[0], pa.int64()).cast(pa.timestamp("ms", tz="UTC")),
            "icao": pa.array(cols[1], pa.string()), "r": pa.array(cols[2], pa.string()),
            "t": pa.array(cols[3], pa.string()), "aircraft_flight": pa.array(cols[4], pa.string()),
            "lat": pa.array(cols[5], pa.float64()), "lon": pa.array(cols[6], pa.float64()),
            "altitude": pa.array(cols[7], pa.int32()), "ground_speed": pa.array(cols[8], pa.float32()),
            "track_degrees": pa.array(cols[9], pa.float32()), "vertical_rate": pa.array(cols[10], pa.int32()),
            "source": pa.array(cols[11], pa.string()),
            "data_source": pa.array(["adsb.lol"] * len(rows), pa.string()),
        })
        destinos[apt].parent.mkdir(parents=True, exist_ok=True)
        tmp = destinos[apt].with_suffix(".tmp")
        pq.write_table(tabla, tmp, compression="zstd")
        tmp.replace(destinos[apt])
        total += len(rows)
    return f"{dia} {n_trazas} trazas, {total:,} puntos, {time.time() - t0:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("desde")
    ap.add_argument("hasta")
    ap.add_argument("--procesos", type=int, default=6)
    ap.add_argument("--variante", default="prod", choices=["prod", "staging", "mlatonly"])
    a = ap.parse_args()
    assert a.variante == VARIANTE, "usa ADSB_LOL_VARIANTE=<variante> en el entorno (los procesos hijos la heredan)"
    d0, d1 = dt.date.fromisoformat(a.desde), dt.date.fromisoformat(a.hasta)
    dias = [d0 + dt.timedelta(days=k) for k in range((d1 - d0).days + 1)]
    with ProcessPoolExecutor(a.procesos) as ex:
        futs = {ex.submit(procesar_dia, d): d for d in dias}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"FALLO {futs[f]}: {e}", flush=True)


if __name__ == "__main__":
    main()
