"""Periodo a predecir, compartido por los scripts (misma variable PRC_PERIODO que `prc/config.py` y `prc/base`).

- "ranking" (por defecto): ene + jul 2026 de ranking.parquet; variables en Data/features/.
- "final": ene, feb, jun y jul 2026 de Data/raw_final/final_ranking.parquet; variables en Data/features_final/
  y enlaces ADS-B por día en Data/adsb/por_dia_final/ (las horas de despegue de ene/jul cambian ±12 s).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = os.environ.get("PRC_PERIODO", "ranking") == "final"
RANKING = ROOT / "Data" / ("raw_final/final_ranking.parquet" if FINAL else "raw/ranking.parquet")
FEAT = ROOT / "Data" / ("features_final" if FINAL else "features")
ADSB_DIA = ROOT / "Data" / "adsb" / ("por_dia_final" if FINAL else "por_dia")
