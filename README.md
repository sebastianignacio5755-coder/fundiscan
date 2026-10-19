# ⚙️ FundiScan · Inspección visual de piezas de fundición con IA

> **Certamen I** · IA Generativa para la Gestión Industrial · Universidad del Desarrollo · 2026
> App web que clasifica la foto de un impulsor de fundición como **conforme** o **defectuoso** y genera una recomendación para el inspector con IA generativa.

| | |
|---|---|
| 🌐 **App publicada** | https://fundiscan.streamlit.app |
| 💻 **Repositorio** | https://github.com/sebastianignacio5755-coder/fundiscan |
| 📒 **Notebook** | [`notebooks/entrenamiento.ipynb`](notebooks/entrenamiento.ipynb) |
| 📄 **Ficha de datos** | [`docs/FICHA_DATOS.md`](docs/FICHA_DATOS.md) |

📷 Imagen → 🧠 EfficientNet-B0 (clase + confianza) → 📊 Resultado visual → 💬 Gemini → ✅ Recomendación

---

## 1. Problema y motivación

En una fundición, al solidificar el metal pueden aparecer **poros, sopladuras, rechupes, rebabas o grietas**. En los **impulsores de bombas sumergibles**, una pieza defectuosa que llega al cliente puede provocar la falla de la bomba, reclamos de garantía y el rechazo del lote completo. Hoy la inspección final es **manual**: lenta, cansadora y dependiente del criterio de cada inspector.

**FundiScan** apoya al **inspector de calidad al final de la línea**. Con una foto de la pieza entrega la clase, su confianza y una recomendación (acción inmediata, posibles causas en el proceso y qué registrar). **No reemplaza al inspector.** Le ayuda a priorizar su atención y deja trazabilidad.

**Error más costoso:** clasificar una pieza **defectuosa como conforme** (falso negativo), porque la pieza mala sale de la planta. Por eso la métrica principal es el **recall de la clase defectuosa**, y el umbral de decisión se ajustó para reducir ese error.

## 2. Datos

| | |
|---|---|
| Dataset | *Casting product image data for quality inspection*. R. Dabhi, Pilot Technocast (Rajkot, India), 2020 |
| Fuente | [Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (no se sube al repositorio) |
| Licencia | CC BY-NC-ND 4.0 (uso no comercial con atribución) |
| Clases | `conforme` (ok_front) · `defectuosa` (def_front) |
| Imágenes | 7.348 fotos de 300×300 px en escala de grises |
| División | train 85 % / val 15 % del *train* original (estratificado, semilla 42) · test = *test* original (715) |

| Conjunto | Conforme | Defectuosa | Total |
|---|---|---|---|
| Entrenamiento | 2.443 | 3.194 | 5.637 |
| Validación | 432 | 564 | 996 |
| Prueba | 262 | 453 | 715 |

Hay un desbalance moderado, con un 57–63 % de piezas defectuosas. **Limitaciones:** todas las fotos vienen de una sola planta, una sola pieza, una cámara y una iluminación; pueden existir fotos casi repetidas entre los conjuntos, y el dataset no indica el tipo de defecto. Más detalle en la [ficha de datos](docs/FICHA_DATOS.md).

## 3. Modelo: comparación y elección

Los tres enfoques se evaluaron con **las mismas 715 imágenes de prueba** y las mismas métricas (umbral 0,5):

| Enfoque | Librería | Accuracy | Acc. balanceada | Recall defectuosa | Tamaño |
|---|---|---|---|---|---|
| A · CLIP sin entrenar (línea base) | Hugging Face | 52,4 % | 62,4 % | 25,2 % | ~600 MB |
| **B · EfficientNet-B0 (transfer learning)** ✅ | timm | 99,4 % | 99,6 % | 99,1 % | ~16 MB |
| C · YOLO11n-cls | Ultralytics | 99,7 % | 99,8 % | 99,6 % | ~3 MB |

> Entrenado en Google Colab con GPU T4 (6 épocas). La app muestra estos números en la pestaña 📊.

**¿Por qué EfficientNet-B0?** YOLO quedó levemente arriba (se le escapan 2 de 453 piezas malas; a EfficientNet, 4), pero la diferencia es mínima y, con el umbral ajustado, EfficientNet baja a 3. Lo elegimos porque en la app solo necesita `torch` y `timm`, y controlamos directamente el umbral y las probabilidades. CLIP sin entrenar demuestra que un modelo general no sabe de control de calidad: se le escapan 339 de 453 piezas malas.

**Umbral de decisión:** se eligió en **validación** minimizando el costo `20 × falsos negativos + 1 × falsos positivos`. El umbral elegido fue **0,1**. Métricas finales en prueba: **accuracy 98,9 %, recall defectuosa 99,3 %, precisión defectuosa 98,9 %** (se escapan 3 de 453 piezas malas y hay 5 falsas alarmas de 262 piezas buenas).

## 4. La app

- 📷 Carga de imagen (archivo o cámara) y 5 ejemplos del conjunto de prueba para probar.
- 🧠 Veredicto con color (verde, rojo o ámbar), **confianza** y **barras de probabilidad** de ambas clases.
- 📊 Pestaña de **desempeño** con accuracy, recall, precisión, matriz de confusión y comparación de modelos. Se explica la diferencia entre la *accuracy* del modelo y la *confianza* de cada foto.
- 💬 Botón de **recomendación con Gemini**. Solo se envía texto (clase, confianza y métricas), nunca la imagen.
- ⚠️ **Advertencias**: aviso de prototipo educativo y revisión humana obligatoria cuando la confianza es menor a 85 %.

**Prompt responsable** (ver `INSTRUCCIONES` en `app.py`): limita el tema a la calidad en fundición, prohíbe inventar datos (las causas son siempre *posibles*), exige revisión humana con confianza baja, recuerda que es un prototipo y fija un formato breve. Si la API falla o no hay clave, la app muestra una **recomendación de respaldo**, así la demo nunca queda en blanco.

**Clave API:** se lee de los *Secrets* de Streamlit Community Cloud (`GEMINI_API_KEY`) o de una variable de entorno. **No está en el repositorio.**

## 5. Estructura del repositorio

```
fundiscan/
├── app.py                      ← app Streamlit (clasificación + Gemini)
├── requirements.txt            ← librerías
├── README.md                   ← este archivo
├── .streamlit/config.toml      ← tema de la app (colores industriales)
├── notebooks/
│   └── entrenamiento.ipynb     ← datos, comparación de 3 modelos, métricas y exportación
├── modelo/
│   ├── efficientnet_b0_fundicion.pt   ← pesos (8 MB, en media precisión)
│   └── config.json                    ← clases, umbral, métricas en prueba
├── ejemplos/                   ← 5 imágenes de prueba
└── docs/
    ├── FICHA_DATOS.md
    ├── GUIA_PUBLICACION.md
    └── GUION_EXPOSICION.md
```

## 6. Cómo ejecutarla

**En línea:** abrir https://fundiscan.streamlit.app

**Localmente:**
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="tu_clave"      # en Windows: set GEMINI_API_KEY=tu_clave
streamlit run app.py                  # abre http://localhost:8501
```

**Reentrenar:** abrir `notebooks/entrenamiento.ipynb` en Google Colab con GPU T4 → *Ejecutar todas* → se descarga `fundiscan_export.zip` con `modelo/` y `ejemplos/`.

> Nota: el notebook exporta los pesos en `float32` (16 MB). Para subirlos por la web los guardamos en media precisión (`float16`, 8 MB); `load_state_dict` los convierte de vuelta y las predicciones no cambian (diferencia < 0,004 en probabilidad).

## 7. Integrantes

| Integrante | Rol principal |
|---|---|
| Sebastián Muñoz | Datos y notebook (secciones 1–4) |
| Renato Valdés | Modelos y métricas (secciones 5–10) |
| Miguel González | App, Gemini y publicación |

## 8. Declaración de uso de IA

| Herramienta | Para qué la usamos |
|---|---|
| Claude (Anthropic) | Estructura inicial del notebook y de `app.py`, redacción de la documentación y del guion de la exposición |
| Google Gemini (API) | Genera las recomendaciones dentro de la app |

Adaptamos el código a nuestro problema (tres enfoques comparados, umbral elegido por costo en validación, interfaz con identidad industrial y recomendación de respaldo). Ejecutamos el notebook y verificamos los resultados. Todos los integrantes revisamos y podemos explicar cada parte del código.

## 9. Ética y limitaciones

- No hay fotos de personas ni datos sensibles.
- Es un **prototipo educativo**, no un sistema de inspección certificado.
- **Sesgo de dominio:** el modelo solo conoce un tipo de pieza fotografiado en condiciones controladas. Con otras piezas, otra luz o fotos de celular puede equivocarse con confianza alta.
- **Mejoras futuras:** validar con fotos de otra línea, Grad-CAM para mostrar dónde está el defecto y clasificar el tipo de defecto.
