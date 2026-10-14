"""Tablas y figuras del modelo para la documentación (Documentacion/tablas/modelo y Documentacion/figuras/modelo).

`tablas` las calcula a partir de Data/ y Submissions/ (locales, fuera del repositorio): error de la receta v30 en la
partición histórica, importancia por bloque y entrega final frente a v30. `figuras` dibuja solo a partir de esas
tablas, así que basta con el repositorio. Las figuras del EDA son históricas y no se rehacen aquí.

El error histórico es el del experimento `mezcla_fuentes+adsb+adsb_stand_lr03` (misma receta, sin los meses de
validación y test): el modelo enviado se entrenó con todo 2025 y no tiene meses reservados.

Uso: python scripts/figuras.py [tablas] [figuras]   (sin argumentos, los dos)
Requiere matplotlib (solo para las figuras; la inferencia no lo usa).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "Documentacion" / "tablas" / "modelo"
FIG = ROOT / "Documentacion" / "figuras" / "modelo"
EXPERIMENTO = ROOT / "Data" / "experimentos" / "mezcla_fuentes+adsb+adsb_stand_lr03"
PARTICIONES = ROOT / "Data" / "comparacion" / "v27_particiones"
FINAL = ROOT / "Submissions" / "v30_final_modelos"
V30 = ROOT / "Submissions" / "v30"
ID = "MVT_ID_mvt"
MESES = {1: "Ene", 2: "Feb", 6: "Jun", 7: "Jul", 11: "Nov", 12: "Dic"}
GRUPOS = {"base_V27": "Base heredada de V27", "fuente_offblock": "Fuente del off-block",
          "ADS_B": "ADS-B y stand", "puerta_stand": "ADS-B y stand"}
ETAPAS = [("rodaje", "cuerpo", "Mezcla con la hora programada"),
          ("cuerpo", "calibrado", "Cola de Roma, consenso y suelo"),
          ("calibrado", "pred", "Proyección LOBT ±3.606 s")]


def _rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def _guardar(df, nombre):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / nombre, index=False, float_format="%.6g")
    print(f"tabla {nombre}: {len(df)} filas")


# --- Tablas (necesitan Data/ y Submissions/) ----------------------------------------------------------------------

def tablas_error_historico():
    p = pd.read_parquet(EXPERIMENTO / "pred.parquet", columns=[ID, "pred"])
    por_aeropuerto, por_mes, concentracion = [], [], []
    for parte, nombre in (("val", "validacion"), ("test", "test")):
        t = pd.read_parquet(PARTICIONES / f"historical_{parte}.parquet",
                            columns=[ID, "target", "airport", "month", "apto_entrenamiento", "taxi_nm_sec"])
        m = t.merge(p, on=ID, validate="one_to_one")
        m["airport"] = m.airport.astype(str)
        m["error"] = m.pred.astype(float) - m.target.astype(float)
        nm = m.taxi_nm_sec.astype(float)
        m["nm_apto"] = nm.between(0, 7200) & m.apto_entrenamiento.astype(bool)
        for apt, g in [("Todos", m), *m.groupby("airport")]:
            por_aeropuerto.append({"particion": nombre, "aeropuerto": apt, "salidas": len(g), "rmse_s": _rmse(g.error),
                                   "sesgo_s": float(g.error.mean()), "salidas_nm_apto": int(g.nm_apto.sum()),
                                   "rmse_nm_apto_s": _rmse(g.error[g.nm_apto])})
        for (mes, apt), g in [*(((mes, "Todos"), g) for mes, g in m.groupby("month")), *m.groupby(["month", "airport"])]:
            por_mes.append({"particion": nombre, "mes": int(mes), "aeropuerto": apt, "salidas": len(g),
                            "rmse_s": _rmse(g.error), "rmse_nm_apto_s": _rmse(g.error[g.nm_apto])})
        cuadrado = np.sort(m.error.to_numpy() ** 2)[::-1]
        acumulado = np.cumsum(cuadrado) / cuadrado.sum()
        n = len(cuadrado)
        for k in np.unique(np.r_[np.round(np.geomspace(1, n, 80)), np.round(n * np.array([.0001, .001, .01, .1]))]):
            k = int(max(k, 1))
            concentracion.append({"particion": nombre, "salidas": k, "fraccion_salidas": k / n,
                                  "fraccion_error_cuadratico": float(acumulado[k - 1])})
    a = pd.DataFrame(por_aeropuerto)
    total = a[a.aeropuerto == "Todos"].set_index("particion").rmse_s
    assert abs(total["validacion"] - 184.3672) < 1e-3 and abs(total["test"] - 304.6691) < 1e-3, total
    _guardar(a, "error_aeropuerto.csv")
    _guardar(pd.DataFrame(por_mes), "error_mes_aeropuerto.csv")
    _guardar(pd.DataFrame(concentracion), "error_concentracion.csv")


def tabla_importancia():
    v = pd.read_csv(TAB / "variables_v30.csv", encoding="utf-8-sig")
    v["grupo"] = v.bloque.map(GRUPOS)
    assert v.grupo.notna().all(), v.bloque.unique()
    t = v.groupby("grupo", sort=False).agg(variables=("variable", "size"),
                                           importancia_clasificador_pct=("importancia_clasificador", "sum"),
                                           importancia_regresor_pct=("importancia_regresor", "sum")).reset_index()
    t.insert(2, "variables_pct", 100 * t.variables / t.variables.sum())
    _guardar(t, "importancia_bloques.csv")


def tablas_final():
    e = pd.read_parquet(FINAL / "etapas.parquet")
    s = pd.read_parquet(FINAL / "knowledgeable-alligator_v30_final_modelos.parquet")
    o = pd.read_parquet(V30 / "knowledgeable-alligator_v30.parquet")
    for d in (e, s, o):
        d[ID] = d[ID].astype("int64")
    m = e.merge(s, on=ID, validate="one_to_one")
    m["airport"] = m.airport.astype(str)
    m["enviado"] = m.TAXITIME_SEC_mvt.astype(float)

    medias = []
    for mes in [*sorted(m.month.unique()), "Todos"]:
        gm = m if mes == "Todos" else m[m.month == mes]
        for apt, g in [*gm.groupby("airport"), ("Todos", gm)]:
            medias.append({"mes": mes, "aeropuerto": apt, "salidas": len(g), "media_s": g.enviado.mean(),
                           "mediana_s": g.enviado.median(), "p90_s": g.enviado.quantile(.9)})
    _guardar(pd.DataFrame(medias), "final_media_mes_aeropuerto.csv")

    etapas = []
    grupos = [(("Todos", "Todos"), m), *(((mes, "Todos"), g) for mes, g in m.groupby("month")),
              *((("Todos", apt), g) for apt, g in m.groupby("airport"))]
    for (mes, apt), g in grupos:
        for antes, despues, nombre in ETAPAS:
            d = g[despues].to_numpy(float) - g[antes].to_numpy(float)
            mod = np.abs(d) >= 1
            etapas.append({"mes": mes, "aeropuerto": apt, "etapa": nombre, "salidas": len(g),
                           "modificadas": int(mod.sum()), "modificadas_pct": 100 * float(mod.mean()),
                           "cambio_medio_s": float(d.mean()),
                           "cambio_abs_medio_modificadas_s": float(np.abs(d[mod]).mean()) if mod.any() else 0.0})
    _guardar(pd.DataFrame(etapas), "final_etapas.csv")

    c = m.merge(o.rename(columns={"TAXITIME_SEC_mvt": "v30"}), on=ID, validate="one_to_one")
    c["dif"] = c.enviado - c.v30.astype(float)
    filas = []
    for apt, g in [("Todos", c), *c.groupby("airport")]:
        d = g.dif.to_numpy()
        filas.append({"aeropuerto": apt, "salidas_comunes": len(g), "iguales_pct": 100 * float(np.mean(d == 0)),
                      "dif_media_s": float(d.mean()), "dif_abs_mediana_s": float(np.median(np.abs(d))),
                      "rms_dif_s": _rmse(d), "dif_abs_p99_s": float(np.percentile(np.abs(d), 99)),
                      "fuera_100s_pct": 100 * float(np.mean(np.abs(d) > 100))})
    _guardar(pd.DataFrame(filas), "final_vs_v30.csv")
    bordes = np.arange(-100, 102, 2)
    cuenta, _ = np.histogram(c.dif.clip(-101, 101), bins=bordes)
    _guardar(pd.DataFrame({"desde_s": bordes[:-1], "hasta_s": bordes[1:], "salidas": cuenta}), "final_vs_v30_histograma.csv")


# --- Figuras (solo leen las tablas) ---------------------------------------------------------------------------------

# Paleta: los tres primeros colores categóricos y la rampa azul de la paleta de referencia
SUPERFICIE, TINTA, TINTA2, APAGADO, REJILLA, EJE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIE = ["#2a78d6", "#eb6834", "#1baf7a"]
RAMPA = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": TINTA,
        "axes.facecolor": SUPERFICIE, "figure.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": EJE, "axes.linewidth": .8, "axes.labelcolor": TINTA2, "axes.titlesize": 12,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "xtick.color": APAGADO, "ytick.color": APAGADO, "xtick.labelcolor": TINTA2, "ytick.labelcolor": TINTA2,
        "grid.color": REJILLA, "grid.linewidth": .8, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": 9, "lines.solid_capstyle": "round"})
    return plt


def _es(texto):
    """Coma decimal y punto de miles, como en los documentos."""
    return texto.replace(",", " ").replace(".", ",").replace(" ", ".")


def _pct(v):
    return _es(f"{v:.1f} %" if v >= 1 or v == 0 else f"{v:.2f} %" if v >= .01 else f"{v:.3f} %")


def _leer(nombre):
    return pd.read_csv(TAB / nombre, encoding="utf-8-sig")


def _nota(fig, texto, y=-.02):
    fig.text(.01, y, texto, fontsize=8.5, color=TINTA2, ha="left", va="top")


def _guardar_fig(plt, fig, nombre):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{nombre}.png", dpi=170, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)
    print(f"figura {nombre}.png")


def _rejilla(ax, eje="x"):
    ax.grid(axis=eje, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _mapa_calor(ax, valores, filas, columnas, formato="{:.0f}"):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", RAMPA)
    v = np.asarray(valores, float)
    im = ax.imshow(v, cmap=cmap, aspect="auto", vmin=np.nanmin(v), vmax=np.nanmax(v))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                r, g, b, _ = im.cmap(im.norm(v[i, j]))
                claro = .2126 * r + .7152 * g + .0722 * b > .5
                ax.text(j, i, formato.format(v[i, j]), ha="center", va="center", fontsize=9,
                        color=TINTA if claro else "white")
    ax.set_xticks(range(len(columnas)), columnas)
    ax.set_yticks(range(len(filas)), filas)
    ax.tick_params(length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(columnas)), minor=True)
    ax.set_yticks(np.arange(-.5, len(filas)), minor=True)
    ax.grid(which="minor", color=SUPERFICIE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    return im


def figura_evolucion(plt):
    t = _leer("evolucion_oficial.csv")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(t))
    ax.plot(x, t.rmse_oficial_s, color=SERIE[0], linewidth=2, marker="o", markersize=7,
            markeredgecolor=SUPERFICIE, markeredgewidth=2, zorder=3)
    for i in (0, len(t) - 1):
        ax.annotate(_es(f"{t.rmse_oficial_s[i]:.2f} s"), (x[i], t.rmse_oficial_s[i]), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color=TINTA)
    ax.set_xticks(x, t.version)
    ax.set(ylabel="RMSE oficial (s)", ylim=(230, 365), title="Evolución del RMSE oficial en el ranking inicial")
    _rejilla(ax, "y")
    _nota(fig, "Enero y julio de 2026, 344.841 salidas; menor es mejor. v30 se subió con el nombre v29.")
    _guardar_fig(plt, fig, "evolucion_oficial")


def figura_ablacion(plt):
    d = _leer("ablacion.csv").set_index("config")
    grupos = [("sin_adsb_todo", "Todo ADS-B"), ("sin_contexto", "Contexto y puerta"),
              ("sin_cat_alta", "Categóricas de alta cardinalidad"), ("sin_adsb_stand", "Distancias a stand/puerta"),
              ("sin_fuentes", "Fuente del off-block"), ("sin_trafico", "Tráfico")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(grupos))
    for k, (col, nombre) in enumerate((("d_val", "Validación (nov–dic 2025)"), ("d_test", "Test (ene + jul 2025)"))):
        ax.barh(y + (k - .5) * .38, [d.loc[g, col] for g, _ in grupos], height=.36, color=SERIE[k], label=nombre,
                zorder=3)
    ax.set_yticks(y, [n for _, n in grupos])
    ax.invert_yaxis()
    ax.axvline(0, color=EJE, linewidth=.8)
    ax.set(xlabel="Aumento del RMSE al retirar el bloque (s)", title="Ablación: qué se pierde al quitar cada bloque")
    ax.legend(loc="lower right")
    _rejilla(ax, "x")
    _nota(fig, "Partición histórica de 2025, lr 0,06; diferencia frente a la media de tres controles. Las retiradas "
               "se solapan y sus efectos no se suman.")
    _guardar_fig(plt, fig, "ablacion")


def figura_cobertura(plt):
    t = _leer("cobertura_final.csv")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(t))
    for k, (col, nombre) in enumerate((("nm_0_7200_%", "NM válido (0–7.200 s)"), ("adsb_%", "Traza ADS-B enlazada"),
                                       ("puerta_%", "Traza que empieza en la puerta"))):
        ax.bar(x + (k - 1) * .26, t[col], width=.24, color=SERIE[k], label=nombre, zorder=3)
    ax.set_xticks(x, [MESES[m] + " 2026" for m in t.mes])
    ax.set(ylabel="Salidas cubiertas (%)", ylim=(0, 112), title="Fuentes disponibles en el conjunto final")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.0))
    _rejilla(ax, "y")
    _nota(fig, "670.790 salidas. Mide disponibilidad de datos, no precisión.")
    _guardar_fig(plt, fig, "cobertura_final")


def figura_error_aeropuerto(plt):
    a = _leer("error_aeropuerto.csv")
    orden = (a[(a.particion == "test") & (a.aeropuerto != "Todos")].sort_values("rmse_s", ascending=False)
             .aeropuerto.tolist())
    filas = ["Todos", *orden]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    y = np.arange(len(filas))
    for ax, col, titulo in ((ejes[0], "rmse_s", "Todas las salidas"),
                            (ejes[1], "rmse_nm_apto_s", "NM válido y etiqueta apta")):
        for k, (parte, nombre) in enumerate((("validacion", "Validación (nov–dic 2025)"),
                                             ("test", "Test (ene + jul 2025)"))):
            v = a[a.particion == parte].set_index("aeropuerto").loc[filas, col]
            ax.barh(y + (k - .5) * .38, v, height=.36, color=SERIE[k], label=nombre, zorder=3)
            ax.annotate(f"{v.iloc[0]:.0f}", (v.iloc[0], (k - .5) * .38), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=TINTA2)
        ax.set_title(titulo, fontsize=10.5)
        ax.set_xlabel("RMSE (s)")
        ax.axhline(.5, color=EJE, linewidth=.8)
        _rejilla(ax, "x")
    ejes[0].set_yticks(y, filas)
    ejes[0].invert_yaxis()
    ejes[1].legend(loc="lower right")
    fig.suptitle("Error de la receta v30 por aeropuerto (partición histórica de 2025)", x=.01, ha="left",
                 fontsize=12, fontweight="semibold")
    _nota(fig, "Entrenada con feb–jun + ago–oct de 2025; no es el modelo enviado (todo 2025). Derecha: AOBT_3 da "
               "0–7.200 s y la etiqueta está en 60–7.200 s.")
    _guardar_fig(plt, fig, "error_aeropuerto")


def figura_error_mes(plt):
    t = _leer("error_mes_aeropuerto.csv")
    columnas = [("validacion", 11), ("validacion", 12), ("test", 1), ("test", 7)]
    aeropuertos = sorted(set(t.aeropuerto) - {"Todos"})
    filas = ["Todos", *aeropuertos]
    idx = t.set_index(["particion", "mes", "aeropuerto"]).rmse_s
    valores = [[idx.get((p, m, apt), np.nan) for p, m in columnas] for apt in filas]
    fig, ax = plt.subplots(figsize=(7, 5.6))
    _mapa_calor(ax, valores, filas, ["Nov\nvalidación", "Dic\nvalidación", "Ene\ntest", "Jul\ntest"])
    ax.axhline(.5, color=SUPERFICIE, linewidth=4)
    ax.set_title("RMSE por aeropuerto y mes (s)")
    _nota(fig, "Receta v30 en la partición histórica de 2025; todas las salidas. Mismas cifras en "
               "tablas/modelo/error_mes_aeropuerto.csv.", y=.01)
    _guardar_fig(plt, fig, "error_mes_aeropuerto")


def figura_concentracion(plt):
    c = _leer("error_concentracion.csv")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k, (parte, nombre) in enumerate((("validacion", "Validación"), ("test", "Test"))):
        g = c[c.particion == parte]
        ax.plot(100 * g.fraccion_salidas, 100 * g.fraccion_error_cuadratico, color=SERIE[k], linewidth=2,
                label=nombre, zorder=3)
        uno = g.iloc[(g.fraccion_salidas - .01).abs().argmin()]
        ax.plot(100 * uno.fraccion_salidas, 100 * uno.fraccion_error_cuadratico, "o", color=SERIE[k], markersize=7,
                markeredgecolor=SUPERFICIE, markeredgewidth=2, zorder=4)
        ax.annotate(f"1 % → {100 * uno.fraccion_error_cuadratico:.0f} %", (1, 100 * uno.fraccion_error_cuadratico),
                    xytext=(8, -6) if k else (-10, 6), textcoords="offset points", fontsize=9, color=TINTA2,
                    va="top" if k else "bottom", ha="left" if k else "right")
        ax.annotate(_es(f"1 salida → {100 * g.fraccion_error_cuadratico.iloc[0]:.1f} %"),
                    (100 * g.fraccion_salidas.iloc[0], 100 * g.fraccion_error_cuadratico.iloc[0]),
                    xytext=(4, -8), textcoords="offset points", fontsize=8.5, color=TINTA2, va="top")
    ax.set_xscale("log")
    ax.set_xticks([.001, .01, .1, 1, 10, 100], ["0,001 %", "0,01 %", "0,1 %", "1 %", "10 %", "100 %"])
    ax.set(xlim=(2e-4, 100), ylim=(0, 102), xlabel="Salidas con mayor error, de más a menos (escala logarítmica)",
           ylabel="Error cuadrático acumulado (%)", title="El RMSE lo decide una fracción pequeña de salidas")
    ax.legend(loc="upper left")
    _rejilla(ax, "both")
    _nota(fig, "Receta v30 en la partición histórica de 2025. Cada curva empieza en la salida con mayor error; una "
               "curva alta a la izquierda indica error concentrado en pocos vuelos.")
    _guardar_fig(plt, fig, "error_concentracion")


def figura_importancia(plt):
    v = _leer("variables_v30.csv")
    v["grupo"] = v.bloque.map(GRUPOS)
    colores = {g: SERIE[i] for i, g in enumerate(dict.fromkeys(GRUPOS.values()))}
    fig, ejes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, col, titulo in ((ejes[0], "importancia_clasificador", "Clasificador de copia del horario"),
                            (ejes[1], "importancia_regresor", "Regresor del rodaje")):
        top = v.nlargest(15, col)
        y = np.arange(len(top))
        ax.barh(y, top[col], height=.62, color=[colores[g] for g in top.grupo], zorder=3)
        ax.set_yticks(y, top.variable, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(titulo, fontsize=10.5)
        ax.set_xlabel("Importancia (% del total del modelo)")
        _rejilla(ax, "x")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=g) for g, c in colores.items()], loc="upper right", ncol=3,
               bbox_to_anchor=(1, 1.0))
    fig.suptitle("Las 15 variables más importantes de cada modelo congelado", x=.01, ha="left", fontsize=12,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, .95))
    _nota(fig, "CatBoost PredictionValuesChange de Submissions/v30/{clf,model}.cbm. Importancia no es efecto causal; "
               "variables correlacionadas se reparten el mérito.")
    _guardar_fig(plt, fig, "importancia_variables")


def figura_importancia_bloques(plt):
    t = _leer("importancia_bloques.csv")
    filas = [("variables_pct", "Proporción de variables"), ("importancia_clasificador_pct", "Importancia, clasificador"),
             ("importancia_regresor_pct", "Importancia, regresor")]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for i, (col, _) in enumerate(filas):
        izquierda = 0.0
        for k, r in t.iterrows():
            ancho = r[col]
            ax.barh(i, ancho, left=izquierda, height=.56, color=SERIE[k], edgecolor=SUPERFICIE, linewidth=2,
                    label=r.grupo if i == 0 else None, zorder=3)
            if ancho >= 7:
                ax.text(izquierda + ancho / 2, i, f"{ancho:.0f} %", ha="center", va="center", fontsize=9,
                        color="white" if k == 0 else TINTA)
            izquierda += ancho
    ax.set_yticks(range(len(filas)), [n for _, n in filas])
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="%", title="Peso de cada bloque: número de variables frente a importancia")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(.5, -.32))
    ax.tick_params(length=0)
    _guardar_fig(plt, fig, "importancia_bloques")


def figura_final_media(plt):
    t = _leer("final_media_mes_aeropuerto.csv")
    t["mes"] = t.mes.astype(str)
    meses = [m for m in ("1", "2", "6", "7")]
    aeropuertos = sorted(set(t.aeropuerto) - {"Todos"})
    filas = ["Todos", *aeropuertos]
    idx = t.set_index(["mes", "aeropuerto"]).media_s
    valores = [[idx.get((m, apt), np.nan) for m in meses] for apt in filas]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    _mapa_calor(ax, valores, filas, [MESES[int(m)] + " 2026" for m in meses])
    ax.axhline(.5, color=SUPERFICIE, linewidth=4)
    ax.set_title("Entrega final: taxi-out medio predicho (s)")
    _nota(fig, "670.790 salidas de final_submitting.parquet; v30 con los modelos congelados.", y=.01)
    _guardar_fig(plt, fig, "final_media_mes_aeropuerto")


def figura_final_etapas(plt):
    t = _leer("final_etapas.csv")
    t = t[(t.mes.astype(str) == "Todos") & (t.aeropuerto == "Todos")]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    y = np.arange(len(t))
    for ax, col, etiqueta in ((ejes[0], "modificadas_pct", "Salidas modificadas (%)"),
                              (ejes[1], "cambio_medio_s", "Cambio de la media de predicción (s)")):
        ax.barh(y, t[col], height=.5, color=SERIE[0], zorder=3)
        for i, (v, n) in enumerate(zip(t[col], t.modificadas)):
            if col == "modificadas_pct":
                miles = f"{n:,}".replace(",", ".")
                texto = _pct(v) if v >= 1 else f"{_pct(v)} ({miles} salidas)"
            else:
                texto = _es(f"{v:+.1f} s")
            ax.annotate(texto, (max(v, 0), i), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=9, color=TINTA2)
        ax.set_xlabel(etiqueta)
        ax.axvline(0, color=EJE, linewidth=.8)
        _rejilla(ax, "x")
        ax.margins(x=.18)
    ejes[0].set_yticks(y, t.etapa)
    ejes[0].invert_yaxis()
    fig.suptitle("Entrega final: efecto de cada etapa del posproceso", x=.01, ha="left", fontsize=12,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, .95))
    _nota(fig, "Modificada = cambio de al menos 1 s. La cola de Roma toca el 0,5 % de las salidas de LIRF y aun así mueve "
               "la media global. Detalle por mes y aeropuerto en tablas/modelo/final_etapas.csv.")
    _guardar_fig(plt, fig, "final_etapas")


def figura_final_vs_v30(plt):
    h = _leer("final_vs_v30_histograma.csv")
    t = _leer("final_vs_v30.csv")
    total = t[t.aeropuerto == "Todos"].iloc[0]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = ejes[0]
    ax.bar(h.desde_s + 1, 100 * h.salidas / h.salidas.sum(), width=1.8, color=SERIE[0], zorder=3)
    ax.set(xlabel="Entrega final − v30 enviado (s)", ylabel="Salidas (%)", xlim=(-100, 100))
    ax.annotate(_es(f"Mediana |Δ| {total.dif_abs_mediana_s:.0f} s · RMS {total.rms_dif_s:.1f} s\n"
                    f"Iguales {total.iguales_pct:.1f} % · |Δ| > 100 s {total.fuera_100s_pct:.2f} %"),
                (.02, .97), xycoords="axes fraction", va="top", fontsize=9, color=TINTA2)
    _rejilla(ax, "y")
    ax = ejes[1]
    a = t[t.aeropuerto != "Todos"].sort_values("rms_dif_s", ascending=False)
    y = np.arange(len(a))
    ax.barh(y, a.rms_dif_s, height=.6, color=SERIE[0], zorder=3)
    ax.set_yticks(y, a.aeropuerto)
    ax.invert_yaxis()
    ax.set_xlabel("RMS de la diferencia (s)")
    _rejilla(ax, "x")
    fig.suptitle("Enero y julio: entrega final frente a v30 sobre las mismas salidas", x=.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, .93))
    _nota(fig, f"{int(total.salidas_comunes):,} salidas con el mismo ID en ambos envíos. Mismos modelos; cambian los "
               "relojes (hasta ±12 s) y las agregaciones con los cuatro meses.".replace(",", "."))
    _guardar_fig(plt, fig, "final_vs_v30")


def tablas():
    tablas_error_historico()
    tabla_importancia()
    tablas_final()


def figuras():
    plt = _estilo()
    for f in (figura_evolucion, figura_ablacion, figura_cobertura, figura_error_aeropuerto, figura_error_mes,
              figura_concentracion, figura_importancia, figura_importancia_bloques, figura_final_media,
              figura_final_etapas, figura_final_vs_v30):
        f(plt)


if __name__ == "__main__":
    pasos = sys.argv[1:] or ["tablas", "figuras"]
    for paso in pasos:
        {"tablas": tablas, "figuras": figuras}[paso]()
