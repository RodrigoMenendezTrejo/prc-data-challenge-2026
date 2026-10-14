"""Tablas de variables base (las 137 variables heredadas que usa el modelo) para 2025 y 2026.

Ejecuta la tubería del paquete `prc/base`:
movimientos limpios (interim) -> base, extras_v4, METAR, combinado, meteo, tráfico, puerta, alineadas y relojes.

Sin --rehacer reutiliza las tablas que existan (las que usó v30) y solo construye las que falten. Con --rehacer las
vuelve a calcular todas; para eso hacen falta los METAR en `Data/metar/2025/` y `Data/metar/2026/`
(`python scripts/descarga_metar.py 2025` y `... 2026`). Las referencias guardadas (tabla P10, categorías) y la tabla
de recuperación de vuelos viven en `configuracion/`.

Uso: python scripts/variables_base.py [--rehacer]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc.base import config as K  # noqa: E402
from prc.base import procesamiento as P  # noqa: E402
from prc.base import tablas as F  # noqa: E402


def main(rehacer: bool):
    if rehacer:
        faltan = [K.METAR_DIRS[a] for a in K.ANIOS if not any(K.METAR_DIRS[a].rglob("*.txt"))]
        assert not faltan, f"faltan los METAR en {faltan}: python scripts/descarga_metar.py <año>"
    interim = [K.interim(f, a) for a in K.ANIOS for f in ("dep", "arr")]
    if rehacer or not all(p.exists() for p in interim):
        con = P.conectar()
        P.construir_movimientos(con)
        P.escribir_interim(con)
        P.informe_calidad(con, por_aeropuerto=True).to_csv(K.INTERIM_DIR / "calidad_datos.csv", index=False)
        print(P.comprobar_interim(con).to_string())
    else:
        print("interim: se reutilizan", ", ".join(p.name for p in interim))
    for paso in F.PASOS:
        for anio in K.ANIOS:
            t0 = time.time()
            reutilizada = K.feature(paso, anio).exists() and not rehacer
            tabla = F.construir(anio, paso, recalcular=rehacer)
            print(f"{paso}_{anio}: {len(tabla):,} filas x {tabla.shape[1]} columnas "
                  f"({'reutilizada' if reutilizada else f'calculada en {time.time() - t0:.0f} s'})", flush=True)


if __name__ == "__main__":
    main("--rehacer" in sys.argv)
