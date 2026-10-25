# Reader v3: revisión y búsqueda en PHerc0800 — 25/10/2026

**Resultado:** Reader v3 recupera claramente letras conocidas en el control. El barrido completo de los seis segmentos públicos de PHerc0800 produce manchas y estructuras, sin caracteres nuevos que podamos defender como legibles. Esto no demuestra que el rollo carezca de tinta.

## Qué aportan los enlaces

- [Reader v3](https://github.com/DomRusso2/reader-v3): pesos publicados y ventanas de 384 × 384 píxeles para dar contexto a los trazos. Es una opción práctica para buscar sin entrenar otro modelo. Los resultados publicados son prometedores, pero la mejora no es uniforme entre ventanas. La afirmación de más de diez veces menos falsa tinta corresponde a un control concreto de fondo.
- [Tutorial oficial de detección](https://scrollprize.org/tutorial5): explica la señal de textura y relieve que puede sobrevivir en el CT, y cómo pasar de una superficie renderizada a una predicción inspeccionable.
- [Villa](https://github.com/ScrollPrize/villa/tree/main): proporciona las herramientas para segmentar, renderizar, entrenar y ejecutar modelos. Su rama principal ya reorganizó la detección en `vesuvius.ink_detection`; el README de Reader todavía utiliza `koine_machines`. Aquí fijamos una versión compatible en lugar de mezclar ambas interfaces.

Se ejecutó el verificador de cifras incluido en Reader: `ALL NUMBERS VERIFIED`. Esa comprobación contrasta sus JSON con sus afirmaciones; **no reproduce de forma independiente su evaluación**. El script de velocidad depende de módulos locales que ese repositorio no publica.

## Fuentes fijadas

| Componente | Revisión |
|---|---|
| Reader, documentación y resultados | `4a9f1c633d2f30d9b522922ec1cb52c082013183` |
| Pesos [domenicor046/reader-v3](https://huggingface.co/domenicor046/reader-v3) | `aee1d64bced4eb35546348f3d276b9e2a0b99c46` |
| Villa, ejecución compatible | `47a9c37f6f8e99afdab47bb46ad425aa726f5dc0` |
| Villa, rama principal revisada | `322cb0899ac9efe2588d5a3395dce6dc2dea3c89` |

Los pesos contienen 34.546.498 parámetros y cargan sin claves ausentes ni sobrantes. Se reutilizan sin modificar los módulos originales de modelo, normalización, selección de capas, parches, ventana Hann e inferencia. Un adaptador acumula en NumPy para conservar los mapas en coma flotante, antes de convertirlos en imágenes de consulta.

## Protocolo ejecutado

1. Control PHerc0139 w039, recorte nativo de 2048 × 2048 con etiquetas humanas. **Está incluido en el entrenamiento de Reader**: comprueba la ejecución y su aspecto visual, no la generalización. La puerta mínima prevista era AP ≥ 0,70 en al menos una orientación.
2. Los seis segmentos públicos de PHerc0800, seleccionados antes de ejecutar Reader. PHerc0800 no aparece en la configuración de entrenamiento publicada. Sigue incluido en la [lista oficial de First Letters](https://scrollprize.org/prizes#first-letters-prizes), lo cual no acredita la novedad de un futuro candidato concreto.
3. Nivel 0 a 8,64 µm, sin reescalado; cada volumen tiene 31 capas. Reader consume 17 capas: índices 7…23 y 23…7 en PHerc0800. En el control de 28 capas utiliza 6…22 y 22…6, siguiendo el centrado original de Villa.
4. Parches 384 × 384, solapamiento del 50 %, mezcla Hann y normalización robusta original. Se conserva cada sentido por separado. RTX 4090, PyTorch 2.8.0+cu128, FP16, sin compilación ni reflejos en el barrido inicial.
5. Revisión visual de los doce mapas completos, con contraste común 0,25–0,75 y contraste local mediana–percentil 99,5. Los valores son **puntuaciones relativas del modelo**, no probabilidades calibradas de tinta.

| Mapa | Prefijo del segmento | Forma nativa Z × Y × X | Área de píxeles no nulos |
|---|---|---|---|
| 00 | `20251028213516` | 31 × 2160 × 2240 | 1,704 cm² |
| 01 | `20251028220042` | 31 × 2040 × 2000 | 1,576 cm² |
| 02 | `20251028220955` | 31 × 2100 × 2100 | 1,527 cm² |
| 03 | `20251028222030` | 31 × 2580 × 2580 | 2,337 cm² |
| 04 | `20251028225813` | 31 × 2300 × 2260 | 1,832 cm² |
| 05 | `20251029010146` | 31 × 980 × 980 | 0,356 cm² |

Suma de superficie renderizada: 9,331 cm², **sin descontar posibles solapamientos entre segmentos**. No equivale a haber inspeccionado 9,331 cm² únicos del rollo.

Los metadatos declaran chunks sin compresión y valor de relleno cero. Se descargaron todos los chunks existentes: 951 presentes y 729 ausentes con el relleno declarado por Zarr. Otros errores de red abortan la preparación. Se conservaron metadatos, URL y hashes de los archivos; no se añadieron valores de CT para rellenar zonas de papiro.

## Resultado y límites

| Control w039 | AP sobre píxeles revisados | AUC |
|---|---|---|
| Capas en sentido forward | 0,991876 | 0,996675 |
| Capas en sentido reverse | 0,346580 | 0,541011 |

El control muestra letras completas y una dependencia clara del sentido de las capas. Son datos de entrenamiento del modelo; estas cifras no predicen su precisión en PHerc0800. En PHerc0800 no hay etiquetas para calcular una precisión real.

Los mapas de PHerc0800 muestran agrupaciones oscuras, con frecuencia alineadas con fibras, fracturas o irregularidades del CT. No identificamos una secuencia de caracteres legibles. Una puntuación alta o una forma que recuerda a una letra aislada no basta para acreditar escritura.

Se eligió de forma exploratoria el grupo más llamativo del segmento 03, orientación forward, coordenadas nativas **x = 768…1919, y = 1536…2303**. Se comprobó localmente con desplazamientos de profundidad −4, 0 y +4 capas (±34,56 µm), además del promedio de cuatro reflejos espaciales en la profundidad central. Estas comprobaciones cambian el contexto y examinan estabilidad; no certifican tinta ni novedad. Los resultados y tiempos están en `verification.json`.

Las manchas principales persisten, con correlaciones de puntuación respecto al recorte central de 0,900 y 0,865 en las profundidades desplazadas, y 0,988 con reflejos. **Siguen sin formar caracteres legibles.** Esa estabilidad tampoco permite distinguir por sí sola tinta de una estructura estable del papiro. La revisión es visual y exploratoria, sin validación papirológica.

No se entrenó ni ajustó Reader, ni se dibujaron trazos o generaron letras. No se ha demostrado una mejora de Kintsugi con este barrido ni una superioridad de Reader sobre otros modelos en datos independientes.

## Resultados locales y reproducción

En `runs/reader_v3_survey_20261025/`:

- `known_letters_control.png`: control junto a la anotación humana.
- `p800_survey_overview.png`: doce mapas en conjunto.
- `p800_00_comparison.png` … `p800_05_comparison.png`: CT original, direcciones y escalas de contraste.
- `depth_and_mirror_verification.png`: comprobación del grupo del segmento 03.
- `*.npy`: puntuaciones originales; `summary.json`, `protocol.json`, `verification.json` y `artifacts.json`: protocolo, métricas, tiempos y hashes.
- `billing_final.json`: lectura final de saldo y recursos; `execution_source_sha256.json`: huellas del código ejecutado en la GPU.

Los originales y el modelo están en `data/reader_v3_survey_20261025/`, con `sources.json` y `hashes.json`. La preparación reutiliza el control y los módulos fijados del experimento anterior en `data/controlled_ink_20261025/`; no es un instalador independiente.

```bash
PYTHONPATH=. python scripts/reader_v3_survey.py --stage prepare
PYTHONPATH=. python scripts/run_controlled_runpod.py --study reader-v3 --max-spend 2 --data-center US-IL-1
PYTHONPATH=. python scripts/reader_v3_verify.py
PYTHONPATH=. python scripts/render_reader_v3_survey.py
```

Se requieren el entorno de PyTorch y las dependencias compatibles descritas en los scripts. El lanzador rechaza sobrescribir una ejecución completa, limita el gasto y descarga/verifica las salidas antes de retirar el almacenamiento temporal. Los cambios y resultados permanecen locales.

## Coste y siguiente trabajo útil

**Esta tanda costó 0,090883 $; saldo final 6,467736 $.** Se verificó gasto activo de 0 $/hora y listas vacías de pods y volúmenes. Los importes proceden de la lectura posterior a la actualización de facturación; la primera lectura de cierre todavía estaba retrasada. La inferencia del control y del barrido tardó 138 segundos, aparte de preparación, transferencia y comprobaciones locales.

Priorizar superficies bien alineadas y otros segmentos sin anotaciones ofrece una prueba más informativa que seguir retocando el contraste de estas manchas. Para PHerc0800, conviene inspeccionar su geometría y extraer un render mejor centrado en la superficie antes de volver a gastar. Un candidato convincente necesitará contexto de escritura, respaldo en el CT, comprobación de publicaciones y revisión experta.
