"""Estilo común de las figuras del EDA y de la evaluación inicial (la tubería de v30 no lo usa).

Paleta categórica validada (orden fijo, nunca cíclico), una sola serie siempre en el color 1, rejilla y ejes
en líneas finas y recesivas, y un solo eje Y por gráfica.
"""

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from prc import config as C

# Paleta categórica en su orden validado. Serie 1 = azul, serie 2 = naranja.
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIE = PALETA[0]
SERIE_2 = PALETA[1]

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
APAGADO = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
RELLENO_SUAVE = "#cde2fb"   # paso 100 de la rampa azul, para rellenos grandes (cajas)

# Rampa secuencial de un solo tono (azul), de claro a oscuro, para magnitudes (hexbin, mapas de calor).
SECUENCIAL = LinearSegmentedColormap.from_list(
    "prc_azul", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

# Divergente para magnitudes con signo (sesgo = pred − real): azul por debajo, rojo por encima, gris neutro en
# el cero. Mismo nº de pasos en cada brazo. Centrarla siempre en 0 (TwoSlopeNorm o vmin = −vmax).
DIVERGENTE = LinearSegmentedColormap.from_list(
    "prc_divergente",
    ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec", "#f9d2d1", "#f09a99", "#e34948", "#a92c2c"])


def aplicar_estilo() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "text.color": TINTA,
        "axes.edgecolor": EJE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlecolor": TINTA, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": TINTA_2, "axes.labelsize": 9,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": REJILLA, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": EJE, "ytick.color": EJE, "xtick.labelcolor": TINTA_2, "ytick.labelcolor": TINTA_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2, "lines.markersize": 6,
        "patch.linewidth": 0,
        "axes.prop_cycle": mpl.cycler(color=PALETA),
    })


def guardar(fig, subcarpeta: str, nombre: str):
    carpeta = C.FIGURES_DIR / subcarpeta
    carpeta.mkdir(parents=True, exist_ok=True)
    path = carpeta / f"{nombre}.png"
    fig.savefig(path)
    return path


def rejilla_aeropuertos(aeropuertos: list[str], alto_fila: float = 2.4, sharex: bool = True,
                        sharey: bool = False, ncols: int = 5):
    """Small multiples: un panel por aeropuerto. Devuelve (fig, {icao: ax})."""
    nrows = math.ceil(len(aeropuertos) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, alto_fila * nrows),
                             sharex=sharex, sharey=sharey, constrained_layout=True, squeeze=False)
    for ax, apt in zip(axes.flat, aeropuertos):
        ax.set_title(f"{apt} · {C.AIRPORTS[apt][0]}", fontsize=9)
    for ax in list(axes.flat)[len(aeropuertos):]:
        ax.set_visible(False)
    return fig, dict(zip(aeropuertos, axes.flat))


def cajas_desde_cuantiles(ax, stats, etiqueta: str, p_bajo: str = "p10", p_alto: str = "p90",
                          vertical: bool = False) -> None:
    """Diagrama de cajas a partir de cuantiles ya calculados (sin cargar millones de filas).

    `stats` es un DataFrame con columnas etiqueta, p_bajo, p25, p50, p75, p_alto. Bigotes = p_bajo..p_alto.
    """
    cajas = [dict(label=str(r[etiqueta]), whislo=r[p_bajo], q1=r["p25"], med=r["p50"], q3=r["p75"],
                  whishi=r[p_alto], fliers=[]) for _, r in stats.iterrows()]
    ax.bxp(cajas, orientation="vertical" if vertical else "horizontal", showfliers=False, patch_artist=True, widths=0.6,
           boxprops=dict(facecolor=RELLENO_SUAVE, edgecolor=SERIE, linewidth=1),
           medianprops=dict(color=SERIE, linewidth=2),
           whiskerprops=dict(color=APAGADO, linewidth=1), capprops=dict(color=APAGADO, linewidth=1))
