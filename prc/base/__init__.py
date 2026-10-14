"""Variables base: las 137 variables heredadas que usa el modelo (de V21–V27), desde los ficheros del reto.

Módulos, en el orden de la tubería:

- `config`: rutas, periodo (PRC_PERIODO) y constantes.
- `procesamiento`: ficheros oficiales -> salidas y llegadas limpias (`Data/interim/`).
- `tablas`: las once tablas de variables por vuelo (`Data/features/base/`): base, extras, METAR, combinado, meteo,
  tráfico, puerta, alineadas y relojes.
- `matriz`: matriz de 137 variables, referencia del residuo (NM o P10) y referencias guardadas.
- `post_proceso`: cola de Roma de V27, consenso y proyección LOBT ± 3606 s.
- `entrega`: validación del fichero contra la plantilla.
- `utiles`: JSON, hashes y registro.
"""
