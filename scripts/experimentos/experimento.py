"""Experimentos de modelo único sobre la partición histórica de V27.

Partición (la de las predicciones históricas de V27): entrenamiento feb–jun + ago–oct de 2025, validación (parada
temprana) nov–dic y test ene + jul de 2025, para comparar cada candidato con V27 sobre los mismos vuelos.

Todos los candidatos comparten: objetivo residual sobre la referencia NM/P10, filas aptas para entrenar,
calibración de fuentes de Roma ajustada solo con los meses de entrenamiento, proyección LOBT y suelo de 60 s.

Uso:
    python scripts/experimentos/experimento.py base            # 137 variables de V27, un CatBoost
    python scripts/experimentos/experimento.py fuentes         # + variables de fuente del off-block
    python scripts/experimentos/experimento.py fuentes,adsb    # + ADS-B
Salida: Data/experimentos/<nombre>/{pred_test.parquet, pred_val.parquet, metricas.json}
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
from prc.base import config as K  # noqa: E402
from prc.base import post_proceso as FU  # noqa: E402
from matriz import FEAT, FUENTES, cargar  # noqa: E402,F401  (los demás experimentos los importan de aquí)

EXP = ROOT / "Data" / "experimentos"
TRAIN_M = [2, 3, 4, 5, 6, 8, 9, 10]
VAL_M = [11, 12]
TEST_M = [1, 7]

def ref_adsb(X, ref):
    """Referencia del residuo = último punto quieto ADS-B cuando la traza empieza en la puerta (<= 25 m) y el avión
    se ve parado > 60 s antes de moverse; en el resto, la referencia NM/P10 de siempre."""
    ls = X["adsb_last_still"].to_numpy(float)
    ok = ((X["adsb_d_gate_m"].to_numpy(float) <= 25) & (X["adsb_stat_obs"].to_numpy(float) > 60)
          & (ls > 60) & (ls < 7200))
    out = np.asarray(ref, float).copy()
    out[ok] = ls[ok]
    print(f"referencia ADS-B en {ok.mean():.3%} de las filas", flush=True)
    return out


def entrenar_catboost(X, y, cats, tr, va, params):
    from catboost import CatBoostRegressor, Pool
    m = CatBoostRegressor(**params)
    m.fit(Pool(X.loc[tr], y[tr], cat_features=cats), eval_set=Pool(X.loc[va], y[va], cat_features=cats),
          use_best_model=True)
    return m


def entrenar_lgb(X, y, cats, tr, va, params):
    import lightgbm as lgb
    Xl = X.copy()
    for c in cats:
        Xl[c] = Xl[c].astype("category")
    p = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, cat_smooth=30, cat_l2=30,
             max_cat_threshold=64, max_bin=255, num_threads=24, verbosity=-1, seed=42)
    p.update({k: v for k, v in params.items() if k not in ("learner",)})
    dtr = lgb.Dataset(Xl.loc[tr], y[tr], categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Xl.loc[va], y[va], reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=6000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(500)])
    return m, Xl


def evaluar(d, pred, mask, v27=None):
    y = d.target.to_numpy(float)[mask]
    p = pred[mask]
    nm = d.taxi_nm_sec.between(0, 7200).to_numpy()[mask]
    apto = d.apto_entrenamiento.to_numpy(bool)[mask]
    out = {"n": int(mask.sum()), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
           "rmse_nm": float(np.sqrt(np.mean((y[nm] - p[nm]) ** 2))),
           "rmse_nm_apto": float(np.sqrt(np.mean((y[nm & apto] - p[nm & apto]) ** 2)))}
    apt = d.airport.astype(str).to_numpy()[mask]
    out["por_aeropuerto_nm_apto"] = {a: float(np.sqrt(np.mean((y[(apt == a) & nm & apto] - p[(apt == a) & nm & apto]) ** 2)))
                                     for a in sorted(set(apt))}
    return out


def main(nombre: str, bloques: list[str], params_extra: dict | None = None):
    t0 = time.time()
    part = (params_extra or {}).pop("particion", "historical") if params_extra else "historical"
    d, X, ref, cats = cargar(2025, bloques, "_hist" if part == "historical" else "")
    mes = d.month.to_numpy()
    apto = d.apto_entrenamiento.to_numpy(bool)
    if part == "historical":
        trm, vam, tem = np.isin(mes, TRAIN_M), np.isin(mes, VAL_M), np.isin(mes, TEST_M)
    else:  # particiones semanales de V27: mismos IDs de validación y test
        ids = {f: set(pd.read_parquet(K.PARTICIONES_V27 / f"{part}_{f}.parquet", columns=["MVT_ID_mvt"]).MVT_ID_mvt)
               for f in ("val", "test")}
        vam = d.MVT_ID_mvt.isin(ids["val"]).to_numpy()
        tem = d.MVT_ID_mvt.isin(ids["test"]).to_numpy()
        trm = ~(vam | tem)
    tr = trm & apto
    va = vam & apto
    te = tem
    y = d.target.to_numpy(float) - ref
    for c in cats:
        X[c] = X[c].astype(str)
    learner = (params_extra or {}).get("learner", "cat")
    params = dict(iterations=12000, depth=10, learning_rate=0.06, loss_function="RMSE", l2_leaf_reg=10,
                  random_seed=42, task_type="GPU", devices="0", gpu_ram_part=0.8, max_ctr_complexity=1,
                  border_count=128, od_type="Iter", od_wait=300, allow_writing_files=False, verbose=1000)
    print(f"{nombre}: {X.shape[1]} variables, train {tr.sum():,} val {va.sum():,} test {te.sum():,}", flush=True)
    if learner == "lgb":
        m, Xl = entrenar_lgb(X, y, cats, tr, va, params_extra or {})
        cuerpo = np.maximum(m.predict(Xl, num_iteration=m.best_iteration) + ref, 0)
        n_it = int(m.best_iteration)
        imp = pd.Series(m.feature_importance("gain"), index=X.columns)
        imp = imp / imp.sum() * 100
    else:
        params.update(params_extra or {})
        m = entrenar_catboost(X, y, cats, tr, va, params)
        from catboost import Pool
        cuerpo = np.maximum(m.predict(Pool(X, cat_features=cats)) + ref, 0)
        n_it = int(m.tree_count_)
        imp = pd.Series(m.get_feature_importance(), index=X.columns)
    # post-proceso con calibración ajustada solo en meses de entrenamiento
    dtr = d.loc[trm]
    cal = FU.ajustar(dtr, 20, max_slope=1.5)
    relojes = d[["MVT_ID_mvt"]].merge(FU.relojes_lobt(2025), on="MVT_ID_mvt", validate="one_to_one")
    pred = FU.proyectar(relojes, FU.aplicar(d, cuerpo, cal))
    res = {"nombre": nombre, "bloques": bloques, "variables": list(X.columns), "iteraciones": n_it,
           "segundos": time.time() - t0,
           "particion": part, "val": evaluar(d, pred, vam), "test": evaluar(d, pred, te)}
    # V27 en los mismos vuelos
    v27 = pd.read_parquet(K.PARTICIONES_V27 / f"{part}_test.parquet")[["MVT_ID_mvt", "pred"]]
    q = d.loc[te, ["MVT_ID_mvt"]].merge(v27, on="MVT_ID_mvt", how="left")
    pv = np.full(len(d), np.nan)
    pv[te] = q.pred.to_numpy()
    res["test_v27"] = evaluar(d, pv, te)
    imp = imp.sort_values(ascending=False)
    res["importancia_top30"] = imp.head(30).round(3).to_dict()
    out = EXP / nombre
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt, "month": mes, "cuerpo": cuerpo, "pred": pred}).loc[te | vam].to_parquet(out / "pred.parquet")
    (out / "metricas.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    m.save_model(str(out / ("model.txt" if learner == "lgb" else "model.cbm")))
    print(json.dumps({k: res[k] for k in ("val", "test", "test_v27")}, indent=1)[:3000])
    print("importancia", res["importancia_top30"])
    print(f"tiempo {time.time() - t0:.0f} s")


if __name__ == "__main__":
    nombre = sys.argv[1]
    bloques = [b for b in nombre.split(",") if b != "base"]
    extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    etiqueta = nombre.replace(",", "+") + ("_" + extra.pop("tag") if extra and "tag" in extra else "")         + ("_" + extra["learner"] if extra and "learner" in extra else "")         + ("_" + extra["particion"] if extra and "particion" in extra else "")
    main(etiqueta, bloques, extra)
