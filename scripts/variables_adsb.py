"""Variables ADS-B por salida: enlace de cada salida con su traza y medidas del rodaje observado.

Para cada aeropuerto y día:
1. Puntos ADS-B del día y de las 4 h anteriores, sin duplicados entre fuentes.
2. Despegues detectados: primer punto en vuelo (altitud > 0 y velocidad > 100 kt) precedido por un punto en tierra
   de la misma aeronave en los 30 min anteriores (adsb_last_ground_gap mide el hueco).
3. Enlace con las salidas del reto: mismo callsign normalizado y |despegue ADS-B − MVT_TIME| <= 240 s; si no hay
   callsign, candidato único del mismo tipo de avión a <= 90 s.
4. Segmento en tierra antes del despegue (huecos < 20 min) y sus medidas relativas al despegue del reto.

Ninguna medida usa el target ni el off-block. Las variables son relativas a MVT_TIME (conocido en 2026).
Uso: python scripts/variables_adsb.py 2025 [procesos]   |   python scripts/variables_adsb.py 2026 [procesos]
Salida: Data/adsb/por_dia/<anio>/<APT>_<fecha>.parquet y, al final, Data/features/adsb_<anio>.parquet
"""
import datetime as dt
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "adsb" / "raw"
sys.path.insert(0, str(ROOT / "scripts"))
import periodo  # noqa: E402

OUT = periodo.ADSB_DIA
FEAT = periodo.FEAT

GAP_SEG = 20 * 60          # hueco máximo dentro de un segmento en tierra (s)
STOP_KT = 1.0              # parado
MOVE_M = 25.0              # desplazamiento que cuenta como inicio de movimiento
MATCH_S = 240
MATCH_TYPE_S = 90
MAX_GAP_TO = 1800          # hueco máximo entre el último punto en tierra y el primero en vuelo


def norm_cs(col):
    return pl.col(col).str.strip_chars().str.to_uppercase()


RAW_LOL = ROOT / "Data" / "adsb" / "raw_lol"
FUENTE = "adsb.lol"   # solo datos con licencia abierta (ODbL)


def fichero_dia(apt: str, d: dt.date) -> Path:
    """Extracción del archivo de GitHub de adsb.lol si existe (julio 2026); si no, la de ADSB.exposed."""
    f = RAW_LOL / apt / f"{d:%Y-%m-%d}.parquet"
    return f if f.exists() else RAW / apt / f"{d:%Y-%m-%d}.parquet"


def cargar_puntos(apt: str, dia: dt.date) -> pl.DataFrame | None:
    files = [fichero_dia(apt, d) for d in (dia - dt.timedelta(days=1), dia)]
    partes = []
    for k, f in enumerate(files):
        if not f.exists():
            if k == 1:
                return None
            continue
        p = pl.read_parquet(f)
        if p.height == 0:
            continue
        if k == 0:  # solo las últimas 4 h del día anterior
            p = p.filter(pl.col("time") >= pl.lit(dt.datetime.combine(dia, dt.time()) - dt.timedelta(hours=4)).dt.replace_time_zone("UTC"))
        partes.append(p.select(["time", "icao", "t", "aircraft_flight", "lat", "lon", "altitude", "ground_speed", "data_source"]))
    if not partes:
        return None
    p = pl.concat(partes, how="vertical_relaxed")
    # solo adsb.lol (licencia abierta ODbL), en los dos años
    p = p.with_columns(pl.col("data_source").cast(pl.Utf8)).filter(pl.col("data_source") == FUENTE)
    if p.height == 0:
        return None
    p = (p.with_columns(pl.col("data_source").cast(pl.Utf8),
                        pl.when(norm_cs("aircraft_flight").str.len_chars() > 0).then(norm_cs("aircraft_flight")).otherwise(None).alias("cs"),
                        (pl.col("time").dt.epoch("ms") / 1000.0).alias("ts"))
           .unique(subset=["icao", "ts"], keep="first", maintain_order=True)   # determinista: el primero leído
           .sort(["icao", "ts"]))
    return p


def despegues(p: pl.DataFrame) -> pl.DataFrame:
    """Instante de despegue por aeronave: primer punto en vuelo con un punto en tierra < MAX_GAP_TO s antes.

    MAX_GAP_TO es amplio (30 min) porque en pistas largas o con poca cobertura falta la carrera de despegue: la
    aeronave se ve en tierra durante el rodaje y luego directamente en vuelo. El hueco queda en adsb_last_ground_gap."""
    q = p.with_columns(
        ((pl.col("altitude") <= 0) | (pl.col("ground_speed") < 40)).alias("tierra"),
        ((pl.col("altitude") > 0) & (pl.col("ground_speed") > 100)).alias("aire"),
    )
    q = q.with_columns(
        pl.col("ts").shift(1).over("icao").alias("ts_prev"),
        pl.col("tierra").shift(1).over("icao").alias("tierra_prev"),
    )
    # último punto en tierra antes de cada punto (para tolerar puntos intermedios en carrera de despegue)
    q = q.with_columns(pl.when(pl.col("tierra")).then(pl.col("ts")).otherwise(None).forward_fill().over("icao").alias("ult_tierra"))
    to = q.filter(pl.col("aire") & (pl.col("ts") - pl.col("ult_tierra") <= MAX_GAP_TO))
    to = to.with_columns(pl.col("ult_tierra").alias("seg_key"))
    to = to.group_by(["icao", "seg_key"]).agg(pl.col("ts").min().alias("ts_to"), pl.col("t").first().alias("t_to"))
    # callsign de la traza: el más frecuente de la aeronave entre 1 h antes y 15 min después del despegue
    cs = (to.select(["icao", "ts_to"]).join(p.filter(pl.col("cs").is_not_null()).select(["icao", "ts", "cs"]), on="icao")
            .filter((pl.col("ts") >= pl.col("ts_to") - 3600) & (pl.col("ts") <= pl.col("ts_to") + 900))
            .group_by(["icao", "ts_to", "cs"]).len().sort(["len", "cs"], descending=[True, False])   # empate: alfabético
            .unique(subset=["icao", "ts_to"], keep="first", maintain_order=True).select(["icao", "ts_to", pl.col("cs").alias("cs_to")]))
    to = to.join(cs, on=["icao", "ts_to"], how="left")
    return to.select(["icao", "ts_to", "cs_to", "t_to"])


def medir(p_icao: np.ndarray, ts_to: float) -> dict:
    """Medidas del segmento en tierra que termina en el despegue. p_icao: columnas ts, lat, lon, gs, alt, tierra."""
    ts, lat, lon, gs, tierra = p_icao
    m = (ts < ts_to) & tierra
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        return {}
    # cortar el segmento en el último hueco > GAP_SEG antes del despegue
    t = ts[idx]
    gaps = np.diff(t)
    corte = np.flatnonzero(gaps > GAP_SEG)
    if len(corte):
        idx = idx[corte[-1] + 1:]
        t = ts[idx]
    la, lo, v = lat[idx], lon[idx], gs[idx]
    x = (lo - lo[0]) * 111320 * np.cos(np.radians(la[0]))
    y = (la - la[0]) * 110540
    dist0 = np.hypot(x, y)
    movido = np.flatnonzero(dist0 > MOVE_M)
    first_move = t[movido[0]] if len(movido) else np.nan
    # último punto todavía en la posición inicial: el inicio real del movimiento está entre ambos instantes
    last_still = t[movido[0] - 1] if len(movido) and movido[0] > 0 else np.nan
    # observado parado en la posición inicial antes de moverse
    stat_obs = (first_move - t[0]) if len(movido) else (t[-1] - t[0])
    # tiempo parado después del primer movimiento
    if len(movido):
        k0 = movido[0]
        dt_ = np.diff(t[k0:])
        parado = (v[k0:-1] < STOP_KT) if len(dt_) else np.array([], bool)
        stop_time = float(np.sum(np.minimum(dt_, 120)[parado])) if len(dt_) else 0.0
        seg = np.hypot(np.diff(x[k0:]), np.diff(y[k0:]))
        path = float(seg.sum())
    else:
        stop_time, path = np.nan, np.nan
    gmax = float(np.max(np.diff(t))) if len(t) > 1 else np.nan
    # posición inicial respecto al último punto en tierra (cerca de la cabecera)
    dx0, dy0 = float(x[-1] - x[0]), float(y[-1] - y[0])
    # cola: tiempo parado o casi parado (< 5 kt) en los últimos 600 s antes del despegue
    ult = t >= ts_to - 600
    if ult.sum() > 1:
        tt, vv = t[ult], v[ult]
        cola = float(np.sum(np.minimum(np.diff(tt), 120)[vv[:-1] < 5]))
    else:
        cola = np.nan
    paradas = int(np.sum((v[1:] < STOP_KT) & (v[:-1] >= STOP_KT))) if len(v) > 1 else 0
    return {
        "adsb_first_seen": ts_to - t[0],
        "adsb_first_move": ts_to - first_move,
        "adsb_last_still": ts_to - last_still,
        "adsb_move_gap": first_move - last_still,
        "adsb_stat_obs": stat_obs,
        "adsb_v0": float(v[0]),
        "adsb_stop_time": stop_time,
        "adsb_path_m": path,
        "adsb_max_gap": gmax,
        "adsb_last_ground_gap": ts_to - t[-1],
        "adsb_d0_m": float(np.hypot(dx0, dy0)),
        "adsb_queue600": cola,
        "adsb_n_stops": paradas,
        "adsb_lat0": float(la[0]),
        "adsb_lon0": float(lo[0]),
    }


def procesar(apt: str, dia: dt.date, deps: pl.DataFrame) -> pl.DataFrame:
    p = cargar_puntos(apt, dia)
    if p is None or p.height == 0:
        return pl.DataFrame()
    to = despegues(p)
    if to.height == 0:
        return pl.DataFrame()
    d = deps.with_columns(norm_cs("FLIGHT_mvt").alias("cs1"), norm_cs("CALLSIGN_flt").alias("cs2"))
    # enlace por callsign
    c1 = d.join(to, left_on="cs1", right_on="cs_to", how="inner")
    c2 = d.join(to, left_on="cs2", right_on="cs_to", how="inner")
    c = pl.concat([c1, c2], how="diagonal_relaxed").with_columns((pl.col("ts_to") - pl.col("ts_mvt")).alias("dto"))
    c = c.filter(pl.col("dto").abs() <= MATCH_S).with_columns(pl.lit(1).alias("adsb_link_cs"))
    c = (c.sort([pl.col("dto").abs(), "icao", "ts_to", "adsb_link_cs"])   # empates de |dto|: regla fija
           .unique(subset=["MVT_ID_mvt"], keep="first", maintain_order=True))
    # sin callsign: tipo de avión y candidato único
    rest = d.filter(~pl.col("MVT_ID_mvt").is_in(c["MVT_ID_mvt"].implode()))
    to_free = to.filter(~pl.col("icao").is_in(c["icao"].implode()))
    c3 = rest.join(to_free, left_on="AIRCRAFT_TYPE_mvt", right_on="t_to", how="inner").with_columns(
        (pl.col("ts_to") - pl.col("ts_mvt")).alias("dto")).filter(pl.col("dto").abs() <= MATCH_TYPE_S)
    c3 = c3.filter(pl.len().over("MVT_ID_mvt") == 1).filter(pl.len().over(["icao", "ts_to"]) == 1)
    c3 = c3.with_columns(pl.lit(0).alias("adsb_link_cs"))
    enl = pl.concat([c.select(["MVT_ID_mvt", "icao", "ts_to", "dto", "adsb_link_cs"]),
                     c3.select(["MVT_ID_mvt", "icao", "ts_to", "dto", "adsb_link_cs"])], how="vertical_relaxed")
    # una traza (icao, despegue) solo para un vuelo
    enl = (enl.sort([pl.col("dto").abs(), "MVT_ID_mvt"])
             .unique(subset=["icao", "ts_to"], keep="first", maintain_order=True))
    if enl.height == 0:
        return pl.DataFrame()
    q = p.with_columns(((pl.col("altitude") <= 0) | (pl.col("ground_speed") < 40)).alias("tierra"))
    grupos = {k[0]: g for k, g in zip(q.partition_by("icao", as_dict=True).keys(), q.partition_by("icao", as_dict=False))}
    filas = []
    for r in enl.iter_rows(named=True):
        g = grupos.get(r["icao"])
        if g is None:
            continue
        arr = (g["ts"].to_numpy(), g["lat"].to_numpy(), g["lon"].to_numpy(), g["ground_speed"].to_numpy().astype(float),
               g["tierra"].to_numpy())
        f = medir(arr, r["ts_to"])
        if not f:
            continue
        # relativas al despegue del reto (no al del ADS-B)
        shift = r["dto"]
        for k in ("adsb_first_seen", "adsb_first_move", "adsb_last_still"):
            f[k] = f[k] + shift
        f.update({"MVT_ID_mvt": r["MVT_ID_mvt"], "adsb_dto": shift, "adsb_link_cs": r["adsb_link_cs"],
                  "adsb_source": g["data_source"][0]})
        filas.append(f)
    return pl.DataFrame(filas) if filas else pl.DataFrame()


def salidas(anio: int) -> pl.DataFrame:
    src = (ROOT / "Data" / "raw" / "training_*.parquet").as_posix() if anio == 2025 else periodo.RANKING.as_posix()
    d = (pl.scan_parquet(src).filter(pl.col("PHASE_mvt") == "DEP")
           .select(["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"])
           .with_columns((pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000.0).alias("ts_mvt"),
                         pl.col("MVT_TIME_UTC_mvt").dt.date().alias("dia"))
           .collect())
    return d


def tarea(apt, dia, deps):
    out = OUT / str(dia.year) / f"{apt}_{dia:%Y-%m-%d}.parquet"
    if out.exists():
        return f"{apt} {dia} ya"
    if not fichero_dia(apt, dia).exists():
        return f"{apt} {dia} sin datos"
    f = procesar(apt, dia, deps)
    out.parent.mkdir(parents=True, exist_ok=True)
    if f.height == 0:
        f = pl.DataFrame({"MVT_ID_mvt": pl.Series("MVT_ID_mvt", [], pl.Float64)})
    f.write_parquet(out)
    return f"{apt} {dia} {f.height}/{deps.height}"


def main(anio: int, procesos=6):
    d = salidas(anio)
    trabajos = []
    for (apt, dia), g in d.group_by(["ADEP_mvt", "dia"]):
        trabajos.append((apt, dia, g))
    with ProcessPoolExecutor(procesos) as ex:
        futs = [ex.submit(tarea, a, di, g) for a, di, g in trabajos]
        for k, f in enumerate(as_completed(futs)):
            r = f.result()
            if k % 50 == 0:
                print(k, r, flush=True)
    files = sorted((OUT / str(anio)).glob("*.parquet"))
    tot = pl.concat([pl.read_parquet(f) for f in files if pl.read_parquet_schema(f).get("adsb_first_seen")], how="diagonal_relaxed")
    FEAT.mkdir(parents=True, exist_ok=True)
    tot.write_parquet(FEAT / f"adsb_{anio}.parquet")
    print("total", tot.height, "de", d.height)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 6)
