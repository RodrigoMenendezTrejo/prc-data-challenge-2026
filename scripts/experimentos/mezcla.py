"""Mezcla explícita: clasificador de copia de la hora programada + regresor del rodaje no copiado.

pred = p(x)·(despegue − programada) + (1 − p(x))·rodaje(x)

- Clasificador (CatBoost GPU, Logloss): etiqueta |BLOCK − SCHED| <= 30 s, filas aptas de los meses de ajuste.
- Regresor (CatBoost GPU, RMSE): residuo sobre NM/P10 en filas aptas que NO copian.
Misma partición, calibración de la cola y proyección LOBT que `experimento.py`; se compara con V27 y con el
modelo único guardado en Data/experimentos/<base>.

Uso: python scripts/experimentos/mezcla.py fuentes   (bloques de variables como en experimento.py)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "experimentos"))
import experimento as E  # noqa: E402
from experimento import FU, K  # noqa: E402


def main(bloques, extra_reg=None, tag=""):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    t0 = time.time()
    nombre = "mezcla_" + "+".join(bloques or ["base"]) + (f"_{tag}" if tag else "")
    d, X, ref, cats = E.cargar(2025, [b for b in bloques if b != "refadsb"], "_hist")
    if "refadsb" in bloques:
        ref = E.ref_adsb(X, ref)
    for c in cats:
        X[c] = X[c].astype(str)
    mes = d.month.to_numpy()
    apto = d.apto_entrenamiento.to_numpy(bool)
    raw = pd.read_parquet(K.interim("dep", 2025), columns=["MVT_ID_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    q = d[["MVT_ID_mvt"]].merge(raw, on="MVT_ID_mvt", validate="one_to_one")
    copia = ((q.BLOCK_TIME_UTC_mvt - q.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 30).to_numpy()
    tr = np.isin(mes, E.TRAIN_M) & apto
    va = np.isin(mes, E.VAL_M) & apto
    te = np.isin(mes, E.TEST_M)
    sig = d.despegue_menos_sched_sec.to_numpy(float)
    comun = dict(task_type="GPU", devices="0", gpu_ram_part=0.8, max_ctr_complexity=1, border_count=128,
                 random_seed=42, od_type="Iter", od_wait=300, allow_writing_files=False, verbose=1000)
    clf = CatBoostClassifier(iterations=6000, depth=8, learning_rate=0.06, loss_function="Logloss", l2_leaf_reg=10, **comun)
    clf.fit(Pool(X.loc[tr], copia[tr].astype(int), cat_features=cats),
            eval_set=Pool(X.loc[va], copia[va].astype(int), cat_features=cats), use_best_model=True)
    p = clf.predict_proba(Pool(X, cat_features=cats))[:, 1]
    y = d.target.to_numpy(float) - ref
    preg = dict(iterations=12000, depth=10, learning_rate=0.06, loss_function="RMSE", l2_leaf_reg=10)
    preg.update(extra_reg or {})
    reg = CatBoostRegressor(**preg, **comun)
    reg.fit(Pool(X.loc[tr & ~copia], y[tr & ~copia], cat_features=cats),
            eval_set=Pool(X.loc[va & ~copia], y[va & ~copia], cat_features=cats), use_best_model=True)
    rodaje = np.maximum(reg.predict(Pool(X, cat_features=cats)) + ref, 0)
    cuerpo = p * np.maximum(sig, 0) + (1 - p) * rodaje
    dtr = d.loc[np.isin(mes, E.TRAIN_M)]
    cal = FU.ajustar(dtr, 20, max_slope=1.5)
    relojes = d[["MVT_ID_mvt"]].merge(FU.relojes_lobt(2025), on="MVT_ID_mvt", validate="one_to_one")
    pred = FU.proyectar(relojes, FU.aplicar(d, cuerpo, cal))
    from sklearn.metrics import roc_auc_score
    res = {"nombre": nombre, "segundos": time.time() - t0, "it_clf": int(clf.tree_count_), "it_reg": int(reg.tree_count_),
           "auc_copia_test": float(roc_auc_score(copia[te & apto], p[te & apto])),
           "val": E.evaluar(d, pred, np.isin(mes, E.VAL_M)), "test": E.evaluar(d, pred, te)}
    out = E.EXP / nombre
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt, "month": mes, "p_copia": p, "rodaje": rodaje, "cuerpo": cuerpo, "pred": pred}
                 ).loc[te | np.isin(mes, E.VAL_M)].to_parquet(out / "pred.parquet")
    (out / "metricas.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    tag = extra.pop("tag", "")
    main([b for b in (sys.argv[1] if len(sys.argv) > 1 else "base").split(",") if b != "base"], extra, tag)
