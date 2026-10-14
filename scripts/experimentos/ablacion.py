"""Ablación de las 177 variables de v30 sobre la partición histórica (feb–jun + ago–oct / nov–dic / ene + jul de 2025).

Receta de v30 (clasificador de copia + regresor del rodaje no copiado), con lr 0,06 para que cada prueba tarde
~8 min en lugar de ~13 (en v30 lr 0,03 y lr 0,06 dieron val 184,4 / 184,7). Cada configuración quita un conjunto de
variables a los dos modelos, reentrena con parada temprana en nov–dic y evalúa con la cola nueva de Roma (ajustada
con los meses de entrenamiento), proyección LOBT ± 3606 s y suelo de 60 s.

Configuraciones:
  completo            las 177 variables (control); <config>_s<n> = la misma con semilla n (ruido)
  sin_<grupo>         quita un grupo (ver GRUPOS)
  top<K>              conserva las K variables más importantes del control (máximo de la importancia normalizada
                      del clasificador y del regresor)
  lista:<nombre>      conserva las variables de Data/ablacion/listas/<nombre>.json
  <config>+signo      además corrige el signo del desplazamiento dto en los tiempos ADS-B (ver corregir_signo)

Salida: Data/ablacion/<config>/{metricas.json, pred.parquet, importancias.csv}; si metricas.json existe, se salta.
Uso: python scripts/experimentos/ablacion.py completo sin_trafico top60 ...   |   ... todo
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "experimentos"))
import cola_roma as cola  # noqa: E402
import experimento as E  # noqa: E402
from experimento import FU, K  # noqa: E402
from prc import referencias as M  # noqa: E402

OUT = ROOT / "Data" / "ablacion"
BLOQUES = ["fuentes", "adsb", "adsb_stand"]

GRUPOS = {
    "trafico": ["dep_apt_15", "dep_apt_60", "dep_pista_15", "dep_pista_60", "hueco_anterior_s", "n_rodando_apt",
                "despegues_mientras_apt", "empujes_15_despues", "n_rodando_pista", "despegues_mientras_pista",
                "llegadas_mientras_rodamos", "nm_vecinos_apt_30", "nm_vecinos_pista_30", "nm_dia_apt",
                "dep_apt_30_antes", "dep_pista_30_despues", "arrivals_push_prev15", "arrivals_takeoff_prev15"],
    "contexto": lambda c: c.startswith("ctx_") or c.startswith("gate_pending") or c == "gate_recent_delay_mean_sec",
    "calendario": ["sin_minuto_dia", "cos_minuto_dia", "minuto_dia_local", "sched_hour", "year_sin", "year_cos"],
    "id_vuelo": ["flight_prefix", "flight_key", "flight_number"],
    "fuentes_offblock": ["source_min", "source_max", "source_spread", "source_above_2h"],
    "metar_despegue": lambda c: c.startswith("wx_"),
    "metar_push": lambda c: c.startswith("push_wx_") or c.startswith("taxi_wx_"),
    "reloj_digitos": lambda c: c.startswith("clock_") and (c.endswith("_second") or c.endswith("_minute5")),
    "reloj_coherencia": lambda c: c.startswith("clock_") and not (c.endswith("_second") or c.endswith("_minute5")),
    "fuentes": list(E.FUENTES),
    "adsb": lambda c: c.startswith("adsb_") and not c.endswith("_menos_ref") and c not in ("adsb_d_gate_m", "adsb_d_stand_m"),
    "adsb_menos_ref": lambda c: c.endswith("_menos_ref"),
    "adsb_stand": ["adsb_d_gate_m", "gate_pos_n", "adsb_d_stand_m", "stand_pos_n"],
    # del núcleo (el resto, aeropuerto/hora/segmento/estela/pista y los tiempos NM base, no se quita)
    "cat_alta": ["tipo_avion", "destino", "apt_stand", "combo", "apt_dique_pista", "operador_rec", "apt_operador_rec"],
    "nm_derivadas": ["eobt1_menos_sched_sec", "ref_p25_sec", "taxi_min_por_stand", "taxi_max_por_stand", "nm_minus_lobt",
                     "nm_minus_eobt", "nm_minus_sched", "lobt_minus_sched", "despegue_menos_iobt", "arr_delay_nm"],
}
# combinaciones informativas (valor total de lo nuevo)
COMBOS = {"adsb_todo": ["adsb", "adsb_menos_ref", "adsb_stand"],
          "nuevas_todo": ["fuentes", "adsb", "adsb_menos_ref", "adsb_stand"]}


def miembros(grupo, cols):
    g = GRUPOS[grupo]
    return [c for c in cols if (g(c) if callable(g) else c in g)]


ADSB_T = ("adsb_first_seen", "adsb_first_move", "adsb_last_still")


def corregir_signo(X):
    """variables_adsb.py guarda (despegue ADS-B − instante) + dto, con dto = despegue ADS-B − MVT, cuando respecto
    al despegue del reto debería ser − dto. Se corrige restando 2·dto, también en las diferencias con la referencia."""
    X = X.copy()
    dto = X["adsb_dto"].to_numpy(float)
    for c in ADSB_T:
        for col in (c, c + "_menos_ref"):
            if col in X:
                X[col] = (X[col].to_numpy(float) - 2 * dto).astype("float32")
    return X


def semilla_de(cfg):
    """`<config>_s<n>` = misma configuración con semilla n (42 por defecto)."""
    m = re.search(r"_s(\d+)$", cfg.split("+")[0])
    return int(m.group(1)) if m else 42


def config_a_columnas(cfg, cols):
    cfg = re.sub(r"_s\d+$", "", cfg.split("+")[0])
    if cfg.startswith("completo"):
        return list(cols)
    if cfg.startswith("sin_"):
        nombre = cfg[4:]
        quitar = set()
        for g in COMBOS.get(nombre, [nombre]):
            quitar |= set(miembros(g, cols))
        assert quitar, cfg
        return [c for c in cols if c not in quitar]
    if cfg.startswith("top"):
        imp = pd.read_csv(OUT / "completo" / "importancias.csv", index_col=0)
        rank = imp.max(axis=1).sort_values(ascending=False)
        keep = set(rank.index[: int(cfg[3:])])
        return [c for c in cols if c in keep]
    if cfg.startswith("lista:"):
        keep = set(json.loads((OUT / "listas" / f"{cfg[6:]}.json").read_text(encoding="utf-8")))
        assert keep <= set(cols), keep - set(cols)
        return [c for c in cols if c in keep]
    raise ValueError(cfg)


class Datos:
    def __init__(self):
        d, X, ref, cats = E.cargar(2025, BLOQUES, "_hist")
        for c in cats:
            X[c] = X[c].astype(str)
        self.d, self.X, self.ref, self.cats = d, X.copy(), ref, cats
        mes = d.month.to_numpy()
        self.mes = mes
        apto = d.apto_entrenamiento.to_numpy(bool)
        raw = pd.read_parquet(K.interim("dep", 2025), columns=["MVT_ID_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
        q = d[["MVT_ID_mvt"]].merge(raw, on="MVT_ID_mvt", validate="one_to_one")
        self.copia = ((q.BLOCK_TIME_UTC_mvt - q.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 30).to_numpy()
        self.trm, self.vam, self.tem = np.isin(mes, E.TRAIN_M), np.isin(mes, E.VAL_M), np.isin(mes, E.TEST_M)
        self.tr, self.va = self.trm & apto, self.vam & apto
        self.y = d.target.to_numpy(float) - ref
        self.sig = np.maximum(d.despegue_menos_sched_sec.to_numpy(float), 0)
        # cola nueva ajustada solo con los meses de entrenamiento (como en v30, que la ajusta con todo 2025)
        f = pd.read_parquet(E.FEAT / "fuentes_2025_hist.parquet", columns=["MVT_ID_mvt", "share_other", "n_other"])
        f["MVT_ID_mvt"] = f.MVT_ID_mvt.astype("int64")
        g = d[["MVT_ID_mvt"]].merge(f, on="MVT_ID_mvt", how="left", validate="one_to_one")
        self.so, self.no = g.share_other.to_numpy(), g.n_other.to_numpy()
        dtr = d.loc[self.trm]
        a_ = dtr.apto_entrenamiento.to_numpy(bool)
        p10 = M.ajustar_p10(dtr.loc[a_ & dtr.combo.notna().to_numpy()], 10)
        nm = d.taxi_nm_sec.to_numpy(float)
        refc = np.where((nm >= 0) & (nm <= 7200), nm, M.aplicar_p10(d, p10).ref_p10_sec.to_numpy(float))
        self.cal = cola.ajustar(dtr, refc[self.trm], self.so[self.trm], self.no[self.trm])
        self.relojes = d[["MVT_ID_mvt"]].merge(FU.relojes_lobt(2025), on="MVT_ID_mvt", validate="one_to_one")


def metricas(D, pred):
    y = D.d.target.to_numpy(float)
    apt = D.d.airport.astype(str).to_numpy()
    out = {}
    for fase, mask in (("val", D.vam), ("test", D.tem)):
        e = (y - pred)[mask]
        out[fase] = float(np.sqrt(np.mean(e ** 2)))
        out[f"{fase}_mes"] = {int(m): float(np.sqrt(np.mean((y - pred)[mask & (D.mes == m)] ** 2)))
                              for m in sorted(set(D.mes[mask]))}
        out[f"{fase}_aeropuerto"] = {a: float(np.sqrt(np.mean((y - pred)[mask & (apt == a)] ** 2))) for a in sorted(set(apt))}
    return out


def correr(D, cfg, semilla=None):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    from sklearn.metrics import roc_auc_score
    salida = OUT / cfg.replace(":", "-")   # Windows no admite ":" en carpetas
    if (salida / "metricas.json").exists():
        print(f"{cfg}: ya existe", flush=True)
        return
    t0 = time.time()
    if semilla is None:
        semilla = semilla_de(cfg)
    cols = config_a_columnas(cfg, D.X.columns)
    cats = [c for c in D.cats if c in cols]
    X = corregir_signo(D.X)[cols] if "+signo" in cfg else D.X[cols]
    print(f"== {cfg}: {len(cols)} variables ({len(cats)} categóricas), semilla {semilla}", flush=True)
    comun = dict(task_type="GPU", devices="0", gpu_ram_part=0.8, max_ctr_complexity=1, border_count=128,
                 random_seed=semilla, od_type="Iter", od_wait=300, allow_writing_files=False, verbose=2000)
    tr, va, copia = D.tr, D.va, D.copia
    clf = CatBoostClassifier(iterations=8000, depth=8, learning_rate=0.06, loss_function="Logloss", l2_leaf_reg=10, **comun)
    clf.fit(Pool(X.loc[tr], copia[tr].astype(int), cat_features=cats),
            eval_set=Pool(X.loc[va], copia[va].astype(int), cat_features=cats), use_best_model=True)
    t_clf = time.time() - t0
    p = clf.predict_proba(Pool(X, cat_features=cats))[:, 1]
    reg = CatBoostRegressor(iterations=16000, depth=10, learning_rate=0.06, loss_function="RMSE", l2_leaf_reg=10, **comun)
    reg.fit(Pool(X.loc[tr & ~copia], D.y[tr & ~copia], cat_features=cats),
            eval_set=Pool(X.loc[va & ~copia], D.y[va & ~copia], cat_features=cats), use_best_model=True)
    rodaje = np.maximum(reg.predict(Pool(X, cat_features=cats)) + D.ref, 0)
    cuerpo = p * D.sig + (1 - p) * rodaje
    pred = FU.proyectar(D.relojes, cola.aplicar(D.d, cuerpo, D.cal, D.so, D.no))
    apto = D.d.apto_entrenamiento.to_numpy(bool)
    res = {"config": cfg, "n_variables": len(cols), "semilla": semilla, "variables": cols,
           "it_clf": int(clf.tree_count_), "it_reg": int(reg.tree_count_),
           "segundos": time.time() - t0, "segundos_clf": t_clf,
           "auc_copia_test": float(roc_auc_score(copia[D.tem & apto], p[D.tem & apto])),
           "logloss_val": float(clf.get_best_score()["validation"]["Logloss"]),
           "rmse_val_rodaje_residuo": float(reg.get_best_score()["validation"]["RMSE"]),
           **metricas(D, pred)}
    salida.mkdir(parents=True, exist_ok=True)
    imp = pd.DataFrame({"clf": clf.get_feature_importance(), "reg": reg.get_feature_importance()}, index=cols)
    imp = imp / imp.sum() * 100
    imp.sort_values("reg", ascending=False).to_csv(salida / "importancias.csv")
    m = D.vam | D.tem
    pd.DataFrame({"MVT_ID_mvt": D.d.MVT_ID_mvt, "month": D.mes, "p_copia": p, "rodaje": rodaje, "cuerpo": cuerpo,
                  "pred": pred}).loc[m].to_parquet(salida / "pred.parquet")
    (salida / "metricas.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{cfg}: val {res['val']:.2f} test {res['test']:.2f} it {res['it_clf']}/{res['it_reg']} "
          f"{res['segundos']:.0f} s", flush=True)


TODO = (["completo", "completo_s7"] + [f"sin_{g}" for g in GRUPOS] + [f"sin_{g}" for g in COMBOS]
        + ["top40", "top60", "top80", "top100", "top130", "completo_s123"])

if __name__ == "__main__":
    cfgs = [c for a in sys.argv[1:] for c in (TODO if a == "todo" else [a])]
    D = Datos()
    print({g: len(miembros(g, D.X.columns)) for g in GRUPOS}, flush=True)
    for cfg in cfgs:
        try:
            correr(D, cfg)
        except Exception as e:  # noqa: BLE001  (sigue con la siguiente configuración)
            print(f"{cfg}: ERROR {e!r}", flush=True)
