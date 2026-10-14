"""Sube un fichero de predicciones al bucket del equipo. Solo con el visto bueno del equipo.

Valida el fichero contra la plantilla del periodo y le pone el nombre que toca: la siguiente versión libre
(knowledgeable-alligator_v<N>.parquet) en el ranking, o knowledgeable-alligator_final.parquet en la fase final, que
solo se sustituye con `reemplazar=True`. Rechaza ficheros ya enviados (mismo SHA-256), escribe el registro antes de
subir y espera el `<fichero>_result.json` con la puntuación.

Desde el cuaderno 07: `reg = preparar(ruta, nota)` valida y enseña el destino sin subir nada; `subir(reg)` sube.
Uso: [PRC_PERIODO=final] python scripts/enviar.py <ruta_parquet> "<nota>"
"""
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc.base import entrega as EN  # noqa: E402
from prc import config as C  # noqa: E402
from prc import data  # noqa: E402


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def preparar(ruta, nota="", reemplazar=False):
    """Valida el fichero y decide su nombre en el bucket. No sube nada: devuelve el registro previsto."""
    ruta = Path(ruta)
    if not ruta.is_absolute():
        ruta = ROOT / ruta
    v = EN.validar(ruta)
    assert v["valido"], v
    c = data.cliente_bucket()
    objetos = {o.object_name for o in c.list_objects(C.TEAM_BUCKET, recursive=True)}
    patron = re.compile(re.escape(C.TEAM_NAME) + r"_(v\d+|final)\.parquet$")
    usadas = sorted(int(m.group(1)[1:]) for n in objetos if (m := patron.fullmatch(n)) and m.group(1) != "final")
    # no repetir un fichero ya enviado
    sha = v["sha256"]
    for n in sorted(objetos):
        if patron.fullmatch(n) and sha256_bytes(c.get_object(C.TEAM_BUCKET, n).read()) == sha:
            raise SystemExit(f"Este fichero ya se envió como {n}")
    if C.FINAL:
        version, objeto = "final", f"{C.TEAM_NAME}_final.parquet"
        if objeto in objetos and not reemplazar:
            raise SystemExit(f"{objeto} ya está en el bucket: para sustituirlo, reemplazar=True")
    else:
        version = max(usadas, default=0) + 1
        objeto = f"{C.TEAM_NAME}_v{version}.parquet"
    registro = ruta.parent / "registro_envio.json"
    assert not registro.exists(), f"Ya existe {registro}: este candidato ya se envió"
    return {"periodo": C.PERIODO, "version": version, "objeto": objeto, "reemplaza": objeto in objetos,
            "sha256": sha, "fichero_local": str(ruta.relative_to(ROOT)), "filas": C.N_RANKING_DEP,
            "nota": nota, "versiones_previas": usadas}


def subir(reg, esperar_s=1800):
    """Escribe el registro, sube el fichero y espera la puntuación (None si no llega en `esperar_s` segundos)."""
    c = data.cliente_bucket()
    registro = ROOT / Path(reg["fichero_local"]).parent / "registro_envio.json"
    assert not registro.exists(), f"Ya existe {registro}: este candidato ya se envió"
    reg = {**reg, "preparado_utc": datetime.now(timezone.utc).isoformat()}
    registro.write_text(json.dumps(reg, indent=1, ensure_ascii=False), encoding="utf-8")
    c.fput_object(C.TEAM_BUCKET, reg["objeto"], str(ROOT / reg["fichero_local"]))
    reg["subido_utc"] = datetime.now(timezone.utc).isoformat()
    registro.write_text(json.dumps(reg, indent=1, ensure_ascii=False), encoding="utf-8")
    print("subido", reg["objeto"], flush=True)
    res = data.esperar_resultado(c, reg["objeto"], timeout_s=esperar_s, cada_s=20)
    reg["resultado"] = res
    registro.write_text(json.dumps(reg, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
    return reg


def main(ruta, nota):
    subir(preparar(ruta, nota))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
