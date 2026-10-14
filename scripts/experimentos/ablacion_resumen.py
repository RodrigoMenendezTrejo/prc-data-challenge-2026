"""Resumen de la ablación: tabla de RMSE por configuración, con diferencias frente al control (media de semillas).

Además del RMSE total (dominado por unas pocas filas de la cola), calcula a partir de pred.parquet el RMSE del
régimen normal (NM 0–7.200 s y etiqueta apta), que es donde actúan casi todas las variables.

Uso: python scripts/experimentos/ablacion_resumen.py  -> imprime markdown y escribe Data/ablacion/resumen.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "experimentos"))
from experimento import K  # noqa: E402

OUT = ROOT / "Data" / "ablacion"


def base():
    d = pd.read_parquet(K.feature("combinado", 2025), columns=["MVT_ID_mvt", "month", "target", "taxi_nm_sec",
                                                                "apto_entrenamiento"])
    d["normal"] = d.taxi_nm_sec.between(0, 7200) & d.apto_entrenamiento.astype(bool)
    return d


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def main():
    d = base()
    filas = []
    for f in sorted(OUT.glob("*/metricas.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        p = pd.read_parquet(f.parent / "pred.parquet", columns=["MVT_ID_mvt", "pred"]).merge(d, on="MVT_ID_mvt")
        e = p.target - p.pred
        val, test = p.month.isin([11, 12]), p.month.isin([1, 7])
        filas.append({"config": m["config"], "n_var": m["n_variables"], "val": m["val"], "test": m["test"],
                      "val_normal": rmse(e[val & p.normal]), "test_normal": rmse(e[test & p.normal]),
                      "test_ene_normal": rmse(e[(p.month == 1) & p.normal]),
                      "test_jul_normal": rmse(e[(p.month == 7) & p.normal]),
                      "auc_copia": m["auc_copia_test"], "it_clf": m["it_clf"], "it_reg": m["it_reg"],
                      "min": m["segundos"] / 60})
    t = pd.DataFrame(filas).set_index("config")
    ctrl = t.loc[[c for c in t.index if c.split("+")[0] in ("completo", "completo_s7", "completo_s123")
                  and "+" not in c]]
    ref = ctrl[["val", "test", "val_normal", "test_normal"]].mean()
    for c in ["val", "test", "val_normal", "test_normal"]:
        t[f"d_{c}"] = t[c] - ref[c]
    t.to_csv(OUT / "resumen.csv")
    print(f"control (media de {len(ctrl)} semillas):", ref.round(2).to_dict())
    print("desviación entre semillas:", ctrl[["val", "test", "val_normal", "test_normal"]].std().round(2).to_dict())
    cols = ["n_var", "val", "d_val", "test", "d_test", "val_normal", "d_val_normal", "test_normal", "d_test_normal",
            "auc_copia", "it_clf", "it_reg", "min"]
    print(markdown(t[cols].sort_values("d_val_normal").round(2)))


def markdown(df):
    lineas = ["| config | " + " | ".join(df.columns) + " |", "|---" * (len(df.columns) + 1) + "|"]
    for i, r in df.iterrows():
        lineas.append(f"| {i} | " + " | ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lineas)


if __name__ == "__main__":
    main()
