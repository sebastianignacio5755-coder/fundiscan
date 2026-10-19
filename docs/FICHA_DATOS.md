# 📄 Ficha de datos · FundiScan

## Origen

| Campo | Detalle |
|---|---|
| Nombre | *Casting product image data for quality inspection* (Kaggle: *real-life industrial dataset of casting product*) |
| Creador | Ravirajsinh Dabhi, con fotos tomadas en **Pilot Technocast**, Shapar, Rajkot (India) |
| Año | 2020 |
| Enlace | https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product |
| Cómo se obtuvo | Fotos cenitales (vista superior) de impulsores de bomba sumergible, tomadas con una cámara fija e iluminación estable, en escala de grises y a 300×300 px |
| Licencia | ✏️ *Copiar la licencia exacta que aparece en la página de Kaggle (sección "License", a la derecha o al final de la página)* |
| Descarga en el notebook | `kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")` |

## Clases

| Carpeta original | Clase en el proyecto | Descripción |
|---|---|---|
| `ok_front` | **conforme** | Pieza sin defectos visibles |
| `def_front` | **defectuosa** | Pieza con defectos de fundición: sopladuras o poros, rechupes, rebabas, grietas o bordes dañados |

## Número de imágenes y división

El dataset trae `train` (6.633) y `test` (715). Nosotros:
1. Dejamos el **test original intacto** como conjunto de **prueba**.
2. Separamos un **15 % del train original como validación**, **estratificado por clase** (`train_test_split`, semilla 42).
3. La validación se usa para elegir la mejor época y el **umbral de decisión**. La prueba se usa **una sola vez** al final.

| Conjunto | Conforme | Defectuosa | Total | % defectuosa |
|---|---|---|---|---|
| Entrenamiento | 2.443 | 3.194 | 5.637 | 56,7 % |
| Validación | 432 | 564 | 996 | 56,6 % |
| Prueba | 262 | 453 | 715 | 63,4 % |
| **Total** | **3.137** | **4.211** | **7.348** | 57,3 % |

**Desbalance:** es moderado y la clase mayoritaria es *defectuosa*. Un modelo que siempre dijera "defectuosa" obtendría 63 % de accuracy en prueba, así que también reportamos **accuracy balanceada** y **recall de la clase defectuosa**.

## Limitaciones

1. **Una sola fuente.** Una planta, un tipo de pieza, una cámara, una iluminación y siempre la cara frontal. El modelo no se puede generalizar a otras piezas, fondos o fotos de celular.
2. **Posibles imágenes casi repetidas.** No hay un identificador de pieza, así que fotos muy parecidas de la misma pieza podrían quedar en entrenamiento y en prueba (fuga de datos). Esto puede inflar las métricas.
3. **Sin tipo de defecto.** Solo se sabe si la pieza es defectuosa, no si el defecto es un poro, una rebaba o una grieta. Por eso la recomendación del LLM habla de causas *posibles*.
4. **Baja resolución.** Al reducir las fotos a 224×224 px, los defectos muy pequeños pueden perderse.

## Consideraciones éticas

El dataset no contiene personas ni datos sensibles. Es público y se cita su fuente. No se sube al repositorio: el notebook lo descarga directamente desde Kaggle.
