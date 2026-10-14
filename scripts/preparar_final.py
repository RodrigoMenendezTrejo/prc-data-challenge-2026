"""Datos y variables del conjunto final (ene, feb, jun y jul de 2026; 670.790 salidas), sin cambiar el diseño.

Se ejecuta con PRC_PERIODO=final: «2026» pasa a ser final_ranking.parquet y todo se escribe aparte
(Data/interim_final, Data/features_final, Data/adsb/por_dia_final). Las tablas base de 2025 son las mismas en los dos
periodos y se leen de Data/interim y Data/features/base. No entrena, no predice y no sube nada.

Pasos (cada uno se salta si su salida existe, salvo --rehacer):
  base      interim y las once tablas de variables base de 2026. Necesita Data/metar/2026 (scripts/descarga_metar.py).
  fuentes   variables de fuente del off-block de 2025 y 2026 (share_other/n_other cambian con los meses nuevos).
  adsb      enlace y medidas ADS-B de 2026 (adsb_2025 se copia: las salidas de 2025 no cambian). Necesita las
            trazas de febrero (ADSB.exposed) y junio (archivo de GitHub de adsb.lol).
  stand     posiciones de puertas/stands con 2025 + los cuatro meses, y distancias.
  comprobar matriz de 2026 con los bloques de v30: columnas iguales a 2025, filas = plantilla, cobertura por mes.

Uso: PRC_PERIODO=final python scripts/preparar_final.py [pasos...] [--rehacer]
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import periodo  # noqa: E402

assert periodo.FINAL, "ejecutar con PRC_PERIODO=final"
import matriz as MZ  # noqa: E402
from prc.base import config as K  # noqa: E402

PASOS = ["base", "fuentes", "adsb", "stand", "comprobar"]


def paso_base(rehacer):
    from prc.base import procesamiento as P
    from prc.base import tablas as F
    if rehacer or not K.interim("dep", 2026).exists():
        con = P.conectar()
        P.construir_movimientos(con)
        P.escribir_interim(con)
        P.informe_calidad(con, por_aeropuerto=True).to_csv(K.INTERIM_DIR / "calidad_datos.csv", index=False)
        print(P.comprobar_interim(con).to_string())
    for paso in F.PASOS:
        t0 = time.time()
        tabla = F.construir(2026, paso, recalcular=rehacer)
        print(f"base {paso}: {len(tabla):,} filas, {tabla.shape[1]} columnas, {time.time() - t0:.0f} s", flush=True)


def paso_fuentes(rehacer):
    import variables_fuente as VF
    if rehacer or not (periodo.FEAT / "fuentes_2026.parquet").exists():
        VF.construir(list(range(1, 13)))


def paso_adsb(rehacer):
    import variables_adsb as VA
    periodo.FEAT.mkdir(parents=True, exist_ok=True)
    destino = periodo.FEAT / "adsb_2025.parquet"
    if not destino.exists():
        shutil.copy2(ROOT / "Data" / "features" / "adsb_2025.parquet", destino)
    if rehacer or not (periodo.FEAT / "adsb_2026.parquet").exists():
        VA.main(2026)


def paso_stand(rehacer):
    import variables_stand as VS
    if rehacer or not (periodo.FEAT / "adsb_stand_2026.parquet").exists():
        VS.main()


def paso_comprobar(_):
    d25, X25, _, cats = MZ.cargar(2025, MZ.BLOQUES_V30, "")
    d26, X26, ref26, _ = MZ.cargar(2026, MZ.BLOQUES_V30, "")
    plantilla = pd.read_parquet(K.SUBMITTING_FILE)
    assert list(X25.columns) == list(X26.columns), "columnas distintas entre 2025 y 2026"
    assert len(d26) == len(plantilla) == K.N_FILAS_ENVIO, (len(d26), len(plantilla))
    assert set(d26.MVT_ID_mvt.astype("int64")) == set(plantilla[K.ID].astype("int64")), "IDs distintos de la plantilla"
    mes = d26.month.to_numpy()
    filas = []
    for m in sorted(set(mes)):
        s = mes == m
        nm = d26.taxi_nm_sec.to_numpy()[s]
        filas.append({"mes": int(m), "salidas": int(s.sum()),
                      "nm_0_7200_%": round(100 * float(np.mean((nm >= 0) & (nm <= 7200))), 1),
                      "metar_%": round(100 * float(1 - np.isnan(X26.wx_temp_c.to_numpy()[s]).mean()), 1),
                      "adsb_%": round(100 * float(1 - np.isnan(X26.adsb_first_seen.to_numpy()[s]).mean()), 1),
                      "puerta_%": round(100 * float(1 - np.isnan(X26.adsb_d_gate_m.to_numpy()[s]).mean()), 1),
                      "fuentes_%": round(100 * float(1 - np.isnan(X26.share_other.to_numpy()[s]).mean()), 1),
                      "ref_media_s": round(float(np.mean(ref26[s])), 1)})
    t = pd.DataFrame(filas)
    print(t.to_string(index=False))
    nulos = X26.drop(columns=cats).isna().mean().sort_values(ascending=False)
    print("columnas con más nulos en 2026:", nulos.head(12).round(3).to_dict())
    t.to_csv(periodo.FEAT / "cobertura_final.csv", index=False)
    print(f"OK: {len(d26):,} salidas, {X26.shape[1]} variables, mismas columnas que 2025")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rehacer = "--rehacer" in sys.argv
    for p in args or PASOS:
        t0 = time.time()
        print(f"=== {p}", flush=True)
        globals()[f"paso_{p}"](rehacer)
        print(f"=== {p} terminado en {time.time() - t0:.0f} s", flush=True)
