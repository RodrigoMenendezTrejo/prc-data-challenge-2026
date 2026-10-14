"""Variables sobre la fuente del off-block registrado (copias de la hora programada y emparejamiento NM).

Todas se calculan con columnas visibles en 2026 (llegadas completas, FLIGHT_ID, horas de plan), salvo las
codificaciones históricas `te_*`, que usan etiquetas de 2025 de otros meses y nunca de 2026.

- share_other / n_other: fracción de las otras salidas del mismo (aeropuerto, número de vuelo) con FLIGHT_ID
  (2025+2026). Sin FLIGHT_ID hoy pero emparejado casi siempre => el off-block quedó a > 1 h del LOBT.
- prev_*: llegada anterior al mismo stand (in-block <= despegue − 600 s): si copió su hora programada, si es la
  misma aerolínea, hueco hasta la hora programada de la salida y retraso de esa llegada.
- arr_copy_pref_6h / arr_copy_apt_2h: tasa de copia de la hora programada en llegadas cercanas.
- fl_rep / fl_nostd: formato raro del número de vuelo (p. ej. ITYTY680).
- te_copy_fl / te_copy_pref / te_n_fl: tasa histórica (2025) de off-block == hora programada.
- cross_day_local: la hora programada es de un día local anterior al despegue.

Uso: python scripts/variables_fuente.py [meses_ajuste]   (por defecto todos los de 2025)
     meses_ajuste p.ej. "2,3,4,5,6,8,9,10" para la partición histórica (te_* solo con esos meses).
Salida: Data/features/fuentes_{2025,2026}[_<etiqueta>].parquet
"""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "raw"
sys.path.insert(0, str(ROOT / "scripts"))
import periodo  # noqa: E402

OUT = periodo.FEAT
TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
      "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome",
      "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}


def construir(meses_ajuste: list[int], etiqueta: str = ""):
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql("SET threads TO 16")
    tz = " ".join(f"WHEN '{a}' THEN '{z}'" for a, z in TZ.items())
    con.sql(f"""create table mv as
      select *, 2025 anio from '{(RAW / 'training_*.parquet').as_posix()}'
      union all by name select *, 2026 anio from '{periodo.RANKING.as_posix()}'""")
    con.sql(f"""create table dep as select MVT_ID_mvt id, anio, ADEP_mvt airport, STAND_mvt st, FLIGHT_mvt fl,
        coalesce(regexp_extract(FLIGHT_mvt, '^[A-Z]{{3}}'), '') pref, FLIGHT_ID_mvt is null noid,
        MVT_TIME_UTC_mvt m, SCHED_TIME_UTC_mvt s, month(MVT_TIME_UTC_mvt) mes,
        case when anio = 2025 then (abs(epoch(BLOCK_TIME_UTC_mvt) - epoch(SCHED_TIME_UTC_mvt)) <= 30)::int end cpy,
        case ADEP_mvt {tz} end tzn
      from mv where PHASE_mvt = 'DEP'""")
    con.sql("""create table arr as select ADES_mvt airport, STAND_mvt st, coalesce(regexp_extract(FLIGHT_mvt, '^[A-Z]{3}'), '') pref,
        BLOCK_TIME_UTC_mvt ib, SCHED_TIME_UTC_mvt s,
        (abs(epoch(BLOCK_TIME_UTC_mvt) - epoch(SCHED_TIME_UTC_mvt)) <= 30)::int acpy,
        epoch(BLOCK_TIME_UTC_mvt) - epoch(SCHED_TIME_UTC_mvt) adelay
      from mv where PHASE_mvt = 'ARR' and BLOCK_TIME_UTC_mvt is not null""")
    # 1) historial de emparejamiento NM del número de vuelo (sin etiquetas)
    con.sql("""create table fl as select airport, fl, count(*) n, sum((not noid)::int) nid from dep where fl is not null group by all""")
    # Las uniones asof eligen una fila entre las que empatan en la hora, y con llegadas de igual in-block el
    # resultado cambiaba entre ejecuciones (~0,2 % de filas en v30). Agrupando antes por instante, cada clave tiene
    # una sola fila, el resultado es determinista y se cuentan todas las llegadas empatadas.
    # 2) llegada anterior al mismo stand (empates: regla fija sobre las llegadas de ese instante)
    con.sql("""create table prev as select d.id, a.ib prev_ib, a.acpy prev_acpy, a.pref prev_pref, a.adelay prev_adelay
        from dep d asof left join (select airport, st, ib, max(acpy) acpy, min(pref) pref, max(adelay) adelay
                                   from arr where st is not null group by airport, st, ib) a
        on d.airport = a.airport and d.st = a.st and a.ib <= d.m - interval 600 second""")
    # 3) tasas de copia en llegadas cercanas, con sumas acumuladas por instante y uniones asof (lineal)
    con.sql("""create table ac_pref as select airport, pref, ib,
        sum(c) over (partition by airport, pref order by ib rows unbounded preceding) cp_c,
        sum(n) over (partition by airport, pref order by ib rows unbounded preceding) cp_n
        from (select airport, pref, ib, sum(acpy) c, count(*) n from arr group by airport, pref, ib)""")
    con.sql("""create table ac_apt as select airport, ib,
        sum(c) over (partition by airport order by ib rows unbounded preceding) ap_c,
        sum(n) over (partition by airport order by ib rows unbounded preceding) ap_n
        from (select airport, ib, sum(acpy) c, count(*) n from arr group by airport, ib)""")
    con.sql("""create table q as select id, airport, pref, s - interval 6 hour p_lo, s + interval 1 hour p_hi,
        s - interval 2 hour a_lo, s + interval 2 hour a_hi from dep""")
    con.sql("""create table w as
      select q.id,
        (h1.cp_c - coalesce(l1.cp_c, 0)) / nullif(h1.cp_n - coalesce(l1.cp_n, 0), 0) arr_copy_pref_6h,
        coalesce(h1.cp_n - coalesce(l1.cp_n, 0), 0) arr_n_pref_6h,
        (h2.ap_c - coalesce(l2.ap_c, 0)) / nullif(h2.ap_n - coalesce(l2.ap_n, 0), 0) arr_copy_apt_2h
      from q
      asof left join ac_pref h1 on q.airport = h1.airport and q.pref = h1.pref and h1.ib <= q.p_hi
      asof left join ac_pref l1 on q.airport = l1.airport and q.pref = l1.pref and l1.ib < q.p_lo
      asof left join ac_apt h2 on q.airport = h2.airport and h2.ib <= q.a_hi
      asof left join ac_apt l2 on q.airport = l2.airport and l2.ib < q.a_lo""")
    # 4) codificación histórica de la copia (2025, solo meses de ajuste, excluyendo el mes de la fila)
    lista = ",".join(str(m) for m in meses_ajuste)
    con.sql(f"""create table h as select * from dep where anio = 2025 and mes in ({lista})""")
    con.sql("""create table hfl_m as select airport, fl, mes, sum(cpy) c, count(*) n from h group by all""")
    con.sql("""create table hfl_t as select airport, fl, sum(c) c, sum(n) n from hfl_m group by all""")
    con.sql("""create table hpf_m as select airport, pref, mes, sum(cpy) c, count(*) n from h group by all""")
    con.sql("""create table hpf_t as select airport, pref, sum(c) c, sum(n) n from hpf_m group by all""")
    con.sql("""create table te0 as select d.id,
        coalesce(ft.c, 0) - coalesce(fm.c, 0) fc, coalesce(ft.n, 0) - coalesce(fm.n, 0) fn,
        coalesce(pt.c, 0) - coalesce(pm.c, 0) pc, coalesce(pt.n, 0) - coalesce(pm.n, 0) pn
      from dep d
      left join hfl_t ft on ft.airport = d.airport and ft.fl = d.fl
      left join hfl_m fm on fm.airport = d.airport and fm.fl = d.fl and fm.mes = d.mes and d.anio = 2025
      left join hpf_t pt on pt.airport = d.airport and pt.pref = d.pref
      left join hpf_m pm on pm.airport = d.airport and pm.pref = d.pref and pm.mes = d.mes and d.anio = 2025""")
    con.sql("""create table te as select id,
        (pc + 50 * 0.05) / (pn + 50) te_copy_pref,
        (fc + 20 * ((pc + 50 * 0.05) / (pn + 50))) / (fn + 20) te_copy_fl, fn te_n_fl from te0""")
    con.sql("""create table res as select d.id MVT_ID_mvt, d.anio,
        (fl.nid - (not d.noid)::int) / nullif(fl.n - 1, 0) share_other, fl.n - 1 n_other,
        pv.prev_acpy, (pv.prev_pref = d.pref)::int prev_same_pref, epoch(d.s) - epoch(pv.prev_ib) prev_gap_sched,
        epoch(d.m) - epoch(pv.prev_ib) prev_gap_takeoff, pv.prev_adelay,
        w.arr_copy_pref_6h, w.arr_n_pref_6h, w.arr_copy_apt_2h,
        (substr(d.fl, 4, 1) = substr(d.fl, 3, 1))::int fl_rep, (not regexp_matches(coalesce(d.fl, ''), '^[A-Z]{3}[0-9]'))::int fl_nostd,
        length(d.fl) fl_len,
        te.te_copy_fl, te.te_n_fl, te.te_copy_pref,
        (cast(timezone(d.tzn, d.s) as date) < cast(timezone(d.tzn, d.m) as date))::int cross_day_local
      from dep d left join fl on fl.airport = d.airport and fl.fl = d.fl
      left join prev pv on pv.id = d.id left join w on w.id = d.id left join te on te.id = d.id""")
    for anio in (2025, 2026):
        f = OUT / f"fuentes_{anio}{etiqueta}.parquet"
        con.sql(f"copy (select * exclude (anio) from res where anio = {anio} order by MVT_ID_mvt) to '{f.as_posix()}' (format parquet)")
        print(f, con.sql(f"select count(*) from res where anio = {anio}").fetchone())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        meses = [int(x) for x in sys.argv[1].split(",")]
        construir(meses, "_" + (sys.argv[2] if len(sys.argv) > 2 else "cv"))
    else:
        construir(list(range(1, 13)))
