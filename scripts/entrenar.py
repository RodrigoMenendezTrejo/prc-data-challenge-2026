"""Entrena el modelo con todo 2025 y predice 2026 (344.841 salidas; 670.790 con PRC_PERIODO=final).

Pasos: matriz (137 variables base + bloques nuevos, `scripts/matriz.py`) -> [it_clf > 0] clasificador CatBoost de copia de la hora
programada (filas aptas) -> regresor CatBoost residual sobre NM/P10 (filas aptas que no copian; sin clasificador,
todas las aptas), con el número de árboles fijado en la validación -> mezcla p·(despegue − programada) + (1 − p)·rodaje
-> calibración de la cola de Roma (con o sin el grupo de emparejamiento) -> proyección LOBT ± 3606 s -> suelo de
60 s -> redondeo y orden de la plantilla oficial.

v30: python scripts/entrenar.py v30 fuentes,adsb,adsb_stand 24000 1 3300 '{"learning_rate": 0.03}'
--modelos=<carpeta>: no entrena; carga clf.cbm y model.cbm de esa carpeta (p. ej. Submissions/v30) y solo predice.
Entrenar en GPU no es reproducible bit a bit; con los modelos guardados, el fichero sí lo es.
No sube nada al bucket: el envío va aparte (scripts/enviar.py o el cuaderno 07).

Uso: python scripts/entrenar.py <nombre> <bloques> <iteraciones> [cola_nueva=1] [it_clf=0] ['{params json}']
     p.ej. python scripts/entrenar.py v28 fuentes,adsb 12000 1 3300   (it_clf > 0: mezcla con clasificador de copia)
Salida: Submissions/<nombre>/{knowledgeable-alligator_<nombre>.parquet, etapas.parquet, informe.json, model.cbm}
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import cola_roma as cola  # noqa: E402
import matriz as MZ  # noqa: E402
from prc import referencias as M  # noqa: E402
from prc.base import config as K  # noqa: E402
from prc.base import entrega as EN  # noqa: E402
from prc.base import post_proceso as FU  # noqa: E402


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(nombre, bloques, iteraciones, cola_nueva=True, it_clf=0, extra=None, modelos=None):
    """it_clf > 0 activa la mezcla explícita: clasificador de copia de la hora programada con it_clf árboles y
    regresor entrenado solo con filas que no copian."""
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    t0 = time.time()
    out = ROOT / "Submissions" / nombre
    out.mkdir(parents=True, exist_ok=True)
    d25, X25, ref25, cats = MZ.cargar(2025, bloques, "")
    d26, X26, ref26, _ = MZ.cargar(2026, bloques, "")
    assert list(X25.columns) == list(X26.columns)
    for X in (X25, X26):
        for c in cats:
            X[c] = X[c].astype(str)
    apto = d25.apto_entrenamiento.to_numpy(bool)
    y = d25.target.to_numpy(float) - ref25
    params = dict(iterations=int(iteraciones), depth=10, learning_rate=0.06, loss_function="RMSE", l2_leaf_reg=10,
                  random_seed=42, task_type="GPU", devices="0", gpu_ram_part=0.8, max_ctr_complexity=1,
                  border_count=128, allow_writing_files=False, verbose=1000)
    params.update(extra or {})
    copia = np.zeros(len(d25), bool)
    if it_clf:
        raw = pd.read_parquet(K.interim("dep", 2025), columns=["MVT_ID_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
        q = d25[["MVT_ID_mvt"]].merge(raw, on="MVT_ID_mvt", validate="one_to_one")
        copia = ((q.BLOCK_TIME_UTC_mvt - q.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 30).to_numpy()
        clf = CatBoostClassifier(iterations=int(it_clf), depth=8, learning_rate=0.06, loss_function="Logloss",
                                 l2_leaf_reg=10, random_seed=42, task_type="GPU", devices="0", gpu_ram_part=0.8,
                                 max_ctr_complexity=1, border_count=128, allow_writing_files=False, verbose=1000)
        if modelos:
            clf.load_model(str(ROOT / modelos / "clf.cbm"))
        else:
            clf.fit(Pool(X25.loc[apto], copia[apto].astype(int), cat_features=cats))
            clf.save_model(str(out / "clf.cbm"))
        p_copia = clf.predict_proba(Pool(X26, cat_features=cats))[:, 1]
    m = CatBoostRegressor(**params)
    filas = apto & ~copia
    if modelos:
        m.load_model(str(ROOT / modelos / "model.cbm"))
    else:
        m.fit(Pool(X25.loc[filas], y[filas], cat_features=cats))
        m.save_model(str(out / "model.cbm"))
    rodaje = np.maximum(m.predict(Pool(X26, cat_features=cats)) + ref26, 0)
    if it_clf:
        sig26 = np.maximum(d26.despegue_menos_sched_sec.to_numpy(float), 0)
        cuerpo = p_copia * sig26 + (1 - p_copia) * rodaje
    else:
        cuerpo = rodaje
    # cola: ajustada con todo 2025
    if cola_nueva:
        f25 = pd.read_parquet(MZ.FEAT / "fuentes_2025.parquet", columns=["MVT_ID_mvt", "share_other", "n_other"])
        f26 = pd.read_parquet(MZ.FEAT / "fuentes_2026.parquet", columns=["MVT_ID_mvt", "share_other", "n_other"])
        for f in (f25, f26):
            f["MVT_ID_mvt"] = f.MVT_ID_mvt.astype("int64")
        g25 = d25[["MVT_ID_mvt"]].merge(f25, on="MVT_ID_mvt", how="left")
        g26 = d26[["MVT_ID_mvt"]].merge(f26, on="MVT_ID_mvt", how="left")
        p10 = M.ajustar_p10(d25.loc[apto & d25.combo.notna().to_numpy()], 10)
        nm = d25.taxi_nm_sec.to_numpy(float)
        refc = np.where((nm >= 0) & (nm <= 7200), nm, M.aplicar_p10(d25, p10).ref_p10_sec.to_numpy(float))
        cal = cola.ajustar(d25, refc, g25.share_other, g25.n_other)
        calibrado = cola.aplicar(d26, cuerpo, cal, g26.share_other, g26.n_other)
    else:
        cal = FU.ajustar(d25, 20, max_slope=1.5)
        calibrado = FU.aplicar(d26, cuerpo, cal)
    relojes = d26[["MVT_ID_mvt"]].merge(FU.relojes_lobt(2026), on="MVT_ID_mvt", validate="one_to_one")
    pred = FU.proyectar(relojes, calibrado)
    etapas = pd.DataFrame({"MVT_ID_mvt": d26.MVT_ID_mvt, "airport": d26.airport.astype(str), "month": d26.month,
                           "rodaje": rodaje, "p_copia": p_copia if it_clf else 0.0, "cuerpo": cuerpo, "calibrado": calibrado, "pred": pred})
    etapas.to_parquet(out / "etapas.parquet")
    plantilla = pd.read_parquet(K.SUBMITTING_FILE)[[K.ID]]
    sub = plantilla.merge(pd.DataFrame({K.ID: etapas.MVT_ID_mvt.astype("float64"),
                                        K.TARGET: np.round(etapas.pred.to_numpy()).astype("int32")}),
                          on=K.ID, how="left", validate="one_to_one")
    fichero = out / f"{K.TEAM_NAME}_{nombre}.parquet"
    sub.to_parquet(fichero, index=False)
    v = EN.validar(fichero)
    comp = EN.comparar_con_enviado(fichero, K.PREDICCION_V27)
    v27 = pd.read_parquet(K.PREDICCION_V27)
    q = sub.merge(v27, on=K.ID, suffixes=("", "_v27"))
    dif = q[K.TARGET].astype(float) - q[f"{K.TARGET}_v27"].astype(float)
    informe = {"nombre": nombre, "bloques": bloques, "iteraciones": int(iteraciones), "cola_nueva": bool(cola_nueva),
               "it_clf": int(it_clf), "params_regresor": params, "modelos_cargados": modelos,
               "validacion": v, "comparacion_v27": comp, "cambio_medio_vs_v27": float(dif.mean()),
               "cambio_sd_vs_v27": float(dif.std()), "pred_media": float(sub[K.TARGET].mean()),
               "segundos": time.time() - t0, "sha256": sha256(fichero)}
    (out / "informe.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(informe, indent=1, default=str))


if __name__ == "__main__":
    modelos = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--modelos=")), None)
    sys.argv = [a for a in sys.argv if not a.startswith("--modelos=")]
    main(sys.argv[1], [b for b in sys.argv[2].split(",") if b not in ("base", "")], int(sys.argv[3]),
         bool(int(sys.argv[4])) if len(sys.argv) > 4 else True, int(sys.argv[5]) if len(sys.argv) > 5 else 0,
         json.loads(sys.argv[6]) if len(sys.argv) > 6 else None, modelos)
