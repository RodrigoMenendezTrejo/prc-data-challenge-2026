"""Descarga del bucket del reto (OpenSky S3). Solo lectura sobre el bucket.

    python scripts/descarga_reto.py          training_*.parquet, ranking.parquet y submitting.parquet -> Data/raw/
    python scripts/descarga_reto.py final    final_ranking.parquet y final_submitting.parquet -> Data/raw_final/
    python scripts/descarga_reto.py list     lista los objetos de los buckets
"""
import json, sys
from pathlib import Path
from minio import Minio

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
cred = json.loads((ROOT / "credentials.json").read_text())
c = Minio("s3.opensky-network.org", access_key=cred["accessKey"], secret_key=cred["secretKey"], secure=True)

buckets = [b.name for b in c.list_buckets()]
print("buckets:", buckets)
if len(sys.argv) > 1 and sys.argv[1] == "list":
    for b in buckets:
        for o in c.list_objects(b, recursive=True):
            print(b, o.object_name, o.size)
    sys.exit()

if len(sys.argv) > 1 and sys.argv[1] == "final":
    FINAL = ROOT / "Data" / "raw_final"
    FINAL.mkdir(parents=True, exist_ok=True)
    for n in ("final_ranking.parquet", "final_submitting.parquet"):
        c.fget_object("prc-2026-datasets", n, str(FINAL / n))
        print("descargado", n, (FINAL / n).stat().st_size)
    sys.exit()


def wanted(name):
    n = name.split("/")[-1]
    return n.endswith(".parquet") and not n.startswith("final") and (n.startswith("training_") or n in ("ranking.parquet", "submitting.parquet"))

for b in buckets:
    for o in c.list_objects(b, recursive=True):
        if wanted(o.object_name):
            dest = RAW / o.object_name.split("/")[-1]
            if dest.exists() and dest.stat().st_size == o.size:
                print("ya está", dest.name); continue
            c.fget_object(b, o.object_name, str(dest))
            print("descargado", b, o.object_name, o.size)
