"""Utilidades de los cuadernos de Notebooks/: periodo, interruptores y ejecución de scripts.

El cuaderno 00 deja el periodo ("ranking" o "final") en V30_PERIODO y los interruptores en V30_<NOMBRE>; un
cuaderno abierto suelto usa sus valores por defecto. `run` ejecuta un script con el mismo Python y el periodo
elegido, enseña la salida en directo y para el cuaderno si el script falla.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe") if (ROOT / ".venv" / "Scripts" / "python.exe").exists() else sys.executable
PERIODOS = ("ranking", "final")
PAQUETES = ("catboost", "duckdb", "lightgbm", "minio", "nbclient", "nbformat", "numpy", "pandas", "polars", "pyarrow",
            "requests", "scipy", "sklearn")   # módulos de requirements.txt que importan los cuadernos y los scripts


def elegir_periodo(defecto: str = "ranking") -> str:
    periodo = os.environ.get("V30_PERIODO", defecto)
    assert periodo in PERIODOS, f"periodo {periodo!r}: debe ser uno de {PERIODOS}"
    # prc/ y scripts/periodo.py leen PRC_PERIODO al importarse: cambiar de periodo exige un kernel nuevo
    for modulo in ("periodo", "prc.base.config", "prc.config"):
        if modulo in sys.modules and sys.modules[modulo].FINAL != (periodo == "final"):
            raise RuntimeError(f"{modulo} ya está importado con otro periodo: reinicia el kernel")
    os.environ["PRC_PERIODO"] = periodo
    return periodo


def comprobar_paquetes():
    """Para en seco si al Python del kernel le falta algún paquete (los cuadernos y scripts usan ese mismo Python)."""
    import importlib.util
    faltan = [m for m in PAQUETES if importlib.util.find_spec(m) is None]
    if faltan:
        raise RuntimeError(f"Al kernel ({sys.executable}) le faltan {faltan}. Elige otro kernel o instala con el "
                           f'kernel apagado: "{sys.executable}" -m pip install -r requirements.txt')


def interruptor(nombre: str, defecto: bool) -> bool:
    v = os.environ.get(f"V30_{nombre.upper()}")
    return defecto if v is None else v.strip().lower() in ("1", "true", "si", "sí", "yes")


def existen(*rutas) -> bool:
    return all((ROOT / r).exists() for r in rutas)


def run(script: str, *args, periodo: str | None = None):
    periodo = periodo or os.environ.get("PRC_PERIODO", "ranking")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "PRC_PERIODO": periodo}
    t0 = time.time()
    print(f"$ python {script} {' '.join(args)}   [PRC_PERIODO={periodo}]", flush=True)
    p = subprocess.Popen([PY, "-W", "ignore", script, *args], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for linea in p.stdout:
        print(linea, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"{script} terminó con código {p.returncode}")
    print(f"[{script}: {time.time() - t0:.0f} s]", flush=True)
