# PRC Data Challenge 2026 · guía del proyecto

Somos el equipo `knowledgeable-alligator`. Aquí va la visión general; el detalle está en los otros tres documentos.

## 1. El problema y el resultado

Hay que predecir el taxi-out de cada salida: los segundos entre el fuera de calzos (off-block) y el despegue. El objetivo es `TAXITIME_SEC_mvt` y se puntúa con RMSE. De las salidas de 2026 conocemos el despegue, el stand, la pista y el plan de vuelo, pero no el off-block. Hay diez aeropuertos: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH y LTFM.

| Conjunto | Periodo | Salidas | Ficheros |
|---|---|---:|---|
| Entrenamiento | Todo 2025 | 2.085.047 | `Data/raw/training_2025-*.parquet` |
| Ranking inicial | Enero y julio de 2026 | 344.841 | `Data/raw/ranking.parquet` y `submitting.parquet` |
| Fase final | Enero, febrero, junio y julio de 2026 | 670.790 | `Data/raw_final/final_ranking.parquet` y `final_submitting.parquet` |

Nuestro mejor envío del ranking inicial fue v30, con **247,8635 s** ([registro](../Submissions/v30/registro_envio.json)). En la fase final aplicamos esos mismos modelos, sin reentrenar, a los cuatro meses: `knowledgeable-alligator_final.parquet`, subido el 14/10/2026 y aceptado con 670.790 pares ([registro](../Submissions/v30_final_modelos/registro_envio.json)). Su puntuación aún no se ha publicado.

## 2. Los documentos

| Documento | Contenido |
|---|---|
| [00_General.md](00_General.md) | Esta guía |
| [01_Modelo_final_v30.md](01_Modelo_final_v30.md) | La receta de v30: variables, entrenamiento, error, importancia, posproceso, ablación y entrega final |
| [02_Modelos_anteriores.md](02_Modelos_anteriores.md) | Cómo llegamos a v30 y qué descartamos |
| [03_Datos_y_analisis.md](03_Datos_y_analisis.md) | Datos, EDA, calidad de las etiquetas, fuentes externas y dónde están las figuras y tablas |

## 3. La solución en pocas palabras

La referencia es el taxi-out del Network Manager (NM), calculado con AOBT_3, o un P10 jerárquico cuando NM falta o se sale de rango. Sobre ella trabajan dos CatBoost con las mismas 177 variables: uno estima la probabilidad de que el off-block registrado sea la hora programada copiada y el otro aprende el rodaje de los vuelos que no copian. Mezclamos ambas estimaciones, calibramos la cola de Roma y aplicamos la restricción LOBT. Lo que más ayudó frente a la versión anterior fueron las trazas abiertas de adsb.lol, que muestran parte del movimiento real; las distancias al stand dicen si la traza empieza en el pushback o a mitad del rodaje. El [documento del modelo final](01_Modelo_final_v30.md) lo explica con detalle.

## 4. Organización del proyecto

```text
PRC-CHALLENGE/
├── Documentacion/              los cuatro documentos
│   ├── figuras/                figuras que citan (las de modelo/ salen de scripts/figuras.py)
│   └── tablas/                 cifras de cada figura y demás tablas citadas
├── Notebooks/                  00–07; ejecutados/ guarda las salidas
├── scripts/                    descarga, variables, matriz, entrenamiento, figuras y envío
│   └── experimentos/           selección y ablación de modelos
├── prc/                        utilidades compartidas
│   └── base/                   procesamiento y las 137 variables heredadas de V27
├── configuracion/              receta, recuperación de vuelos y referencias congeladas
├── Data/                       entradas, variables, comparaciones y experimentos (local)
└── Submissions/                modelos, predicciones, etapas y registros (en git, solo los JSON)
```

`Data/` y los modelos y predicciones de `Submissions/` no están en el repositorio público: los datos se descargan con los scripts y los modelos se publican aparte, como explica el `README.md`.

| Ruta | Contenido |
|---|---|
| `Documentacion/figuras/`, `Documentacion/tablas/` | Lo que citan los documentos; el resto de la antigua `reports/` está archivado en `Data/archivo/` |
| `configuracion/` | Lista y orden de las variables base, P10, categorías, calibración y la tabla de recuperación de operador, segmento y estela |
| `Data/raw/`, `Data/raw_final/` | Ficheros oficiales y plantillas, sin tocar |
| `Data/interim/`, `Data/features/` | Movimientos limpios, tablas base y variables de fuente, ADS-B y stand del ranking inicial |
| `Data/interim_final/`, `Data/features_final/` | Lo mismo, recalculado para los cuatro meses finales |
| `Data/metar/`, `Data/adsb/` | METAR y trazas, con cachés por día |
| `Data/comparacion/`, `Data/experimentos/`, `Data/ablacion/` | Predicción de V27, particiones de referencia y resultados de la selección (no son envíos) |
| `Submissions/v30/` | Fichero enviado, los dos modelos congelados, etapas e informes |
| `Submissions/v30_final_modelos/` | La entrega final con sus etapas, informe y registro |
| `Submissions/v30_recreado*/` | Reproducción del ranking inicial con los modelos guardados y reentrenando |
| `Submissions/v28/`, `v29/`, `v31/`, `cand_fuentes/` | Versión anterior y candidatas (ver el documento 02) |

## 5. Cómo ejecutarlo

Los cuadernos de `Notebooks/` llaman a los scripts de `scripts/` con el Python de su kernel, que necesita los paquetes de `requirements.txt` (el 00 lo comprueba al empezar).

| Cuaderno | Qué hace |
|---|---|
| `00_v30_completo` | Ejecuta del 01 al 06 con los interruptores y el periodo elegidos |
| `01_datos` | Descarga los ficheros del reto (necesita `credentials.json`) |
| `02_variables_base` | METAR y las 137 variables base |
| `03_adsb` | Trazas de adsb.lol y variables ADS-B y de puerta/stand |
| `04_variables_fuente` | Variables de fuente del off-block |
| `05_experimentos` | Experimentos de selección; opcional, solo en el periodo del ranking inicial |
| `06_modelo` | Predice con los modelos congelados de v30 y, si se pide, reentrena |
| `07_envio` | Sube un fichero al bucket del equipo; el 00 no lo ejecuta |

En el 00, `PERIODO` elige el conjunto: `"ranking"` (enero y julio de 2026) o `"final"` (los cuatro meses de la fase final). En modo final, los datos de 2026 van a `Data/interim_final/`, `Data/features_final/` y `Data/adsb/por_dia_final/`, y la predicción a `Submissions/v30_final_modelos/` (o a `v30_final/` si se reentrena). Por defecto se reutiliza lo ya calculado; los interruptores `DESCARGAR_*` y `REHACER_*` fuerzan cada paso. Al cambiar de periodo hay que reiniciar el kernel.

El 07 comprueba el fichero contra la plantilla, le pone el nombre que toca en el bucket (`_v<N>.parquet` en la fase inicial, `_final.parquet` en la final), guarda `registro_envio.json` y espera la puntuación. Con `ENVIAR = 0` no sube nada.
