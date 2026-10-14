"""Descarga de los METAR de los diez aeropuertos desde el IEM (Iowa State) para un año del reto.

Una petición por aeropuerto, con hasta tres reintentos. El CSV se guarda tal cual llega en
Data/metar/iem/<ST>_<año>.csv, y los partes se escriben en el formato que lee prc/metar.py:
    Data/metar/<año>/<ST>/<ST>_<AAAA-MM>.txt
    202602010020 METAR EDDF 010020Z AUTO VRB01KT 9999 OVC011 03/01 Q1009 NOSIG=

Periodos: 2025 = 31/12/2024 – 1/1/2026 (el último parte anterior a cada vuelo, con margen en los bordes);
2026 = 1/1 – 2/8/2026 (los cuatro meses del conjunto final; incluye enero y julio de ranking.parquet).

Uso:
    python scripts/descarga_metar.py [2025|2026]   (por defecto 2026)
"""
import csv
import io
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SALIDA_CSV = ROOT / "Data" / "metar" / "iem"

AEROPUERTOS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={st}&data=metar"
       "&year1={y1}&month1={m1}&day1={d1}&year2={y2}&month2={m2}&day2={d2}&tz=Etc/UTC&format=onlycomma"
       "&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4")
# (inicio, fin exclusivo) de cada año
PERIODOS = {2025: ((2024, 12, 31), (2026, 1, 2)), 2026: ((2026, 1, 1), (2026, 8, 3))}
CABECERA = "station,valid,metar"
REINTENTOS = 3


def descargar(st: str, anio: int) -> str:
    """CSV de un aeropuerto como texto. Reintenta si falla la red o si la respuesta no es el CSV esperado."""
    for intento in range(REINTENTOS + 1):
        try:
            (y1, m1, d1), (y2, m2, d2) = PERIODOS[anio]
            r = requests.get(URL.format(st=st, y1=y1, m1=m1, d1=d1, y2=y2, m2=m2, d2=d2), timeout=300)
            r.raise_for_status()
            texto = r.content.decode("utf-8", errors="replace")
            if texto.splitlines()[:1] != [CABECERA]:
                raise ValueError(f"respuesta inesperada: {texto[:200]!r}")
            return texto
        except Exception as e:
            print(f"  {st} intento {intento + 1} fallido: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            if intento < REINTENTOS:
                time.sleep(10 * (intento + 1))
    raise RuntimeError(f"{st}: sin respuesta válida tras {REINTENTOS} reintentos")


def main(anio: int):
    SALIDA_TXT = ROOT / "Data" / "metar" / str(anio)
    SALIDA_TXT.mkdir(parents=True, exist_ok=True)
    SALIDA_CSV.mkdir(parents=True, exist_ok=True)
    fallos = []
    for i, st in enumerate(AEROPUERTOS):
        if i > 0:
            time.sleep(2)                       # pausa entre estaciones
        try:
            texto = descargar(st, anio)
        except RuntimeError as e:
            print(f"{st}: ERROR {e}", file=sys.stderr)
            fallos.append(st)
            continue
        (SALIDA_CSV / f"{st}_{anio}.csv").write_text(texto, encoding="utf-8", newline="")

        filas = list(csv.DictReader(io.StringIO(texto)))
        por_mes = {}
        primera = ultima = None
        for f in filas:
            parte = (f["metar"] or "").strip()
            if parte in ("", "M"):              # vacíos o sin dato
                continue
            t = datetime.fromisoformat(f["valid"])
            cuerpo = parte if parte.startswith(("METAR ", "SPECI ")) else "METAR " + parte
            por_mes.setdefault(f"{t:%Y-%m}", []).append(f"{t:%Y%m%d%H%M} {cuerpo}=")
            primera = primera or f["valid"]
            ultima = f["valid"]

        carpeta = SALIDA_TXT / st
        carpeta.mkdir(parents=True, exist_ok=True)
        for mes, lineas in sorted(por_mes.items()):
            (carpeta / f"{st}_{mes}.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")

        escritas = sum(len(v) for v in por_mes.values())
        print(f"{st}: {len(filas)} filas CSV, {escritas} partes escritos, primero {primera}, ultimo {ultima}")

    if fallos:
        print("Fallaron:", ", ".join(fallos), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
