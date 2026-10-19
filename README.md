---
title: FundiScan
emoji: ⚙️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.11"
pinned: false
short_description: Inspección visual de piezas de fundición con IA
---

# ⚙️ FundiScan · Inspección visual de piezas de fundición con IA

> **Certamen I** · IA Generativa para la Gestión Industrial · Universidad del Desarrollo · 2026
> App web que clasifica la foto de un impulsor de fundición como **conforme** o **defectuoso** y genera una recomendación para el inspector con IA generativa.

| | |
|---|---|
| 🌐 **App publicada** | https://huggingface.co/spaces/USUARIO/fundiscan |
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
| A · CLIP sin entrenar (línea base) | Hugging Face | __ % | __ % | __ % | ~600 MB |
| **B · EfficientNet-B0 (transfer learning)** ✅ | timm | __ % | __ % | __ % | ~16 MB |
| C · YOLO11n-cls | Ultralytics | __ % | __ % | __ % | ~3 MB |

> ✏️ Completar con los números de la sección 7 del notebook. La app también los muestra en la pestaña 📊.

**¿Por qué EfficientNet-B0?** Tiene un recall en piezas defectuosas igual o mejor que el de YOLO (✏️ confirmar con sus números), un archivo liviano que cabe en GitHub y solo necesita `torch` y `timm` en la app. CLIP sin entrenar demuestra que un modelo general no sabe de control de calidad: hay que adaptarlo con datos de la planta.

**Umbral de decisión:** se eligió en **validación** minimizando el costo `20 × falsos negativos + 1 × falsos positivos`. Métricas finales en prueba con ese umbral: accuracy __ %, recall defectuosa __ %, precisión defectuosa __ %.

## 4. La app

- 📷 Carga de imagen (archivo, cámara o portapapeles) y 5 ejemplos clicables del conjunto de prueba.
- 🧠 Veredicto con color (verde, rojo o ámbar), **confianza** y **barras de probabilidad** de ambas clases.
- 📊 Pestaña de **desempeño** con accuracy, recall, precisión, matriz de confusión y comparación de modelos. Se explica la diferencia entre la *accuracy* del modelo y la *confianza* de cada foto.
- 💬 Botón de **recomendación con Gemini**. Solo se envía texto (clase, confianza y métricas), nunca la imagen.
- ⚠️ **Advertencias**: aviso de prototipo educativo y revisión humana obligatoria cuando la confianza es menor a 85 %.

**Prompt responsable** (ver `INSTRUCCIONES` en `app.py`): limita el tema a la calidad en fundición, prohíbe inventar datos (las causas son siempre *posibles*), exige revisión humana con confianza baja, recuerda que es un prototipo y fija un formato breve. Si la API falla o no hay clave, la app muestra una **recomendación de respaldo**, así la demo nunca queda en blanco.

**Clave API:** se lee de la variable de entorno `GEMINI_API_KEY`, guardada como *secret* del Space. **No está en el repositorio.**

## 5. Estructura del repositorio

```
fundiscan/
├── app.py                      ← app Gradio (clasificación + Gemini)
├── requirements.txt            ← librerías
├── README.md                   ← este archivo (también es la ficha del Space)
├── notebooks/
│   └── entrenamiento.ipynb     ← datos, comparación de 3 modelos, métricas y exportación
├── modelo/
│   ├── efficientnet_b0_fundicion.pt   ← pesos (~16 MB)
│   └── config.json                    ← clases, umbral, métricas en prueba
├── ejemplos/                   ← 5 imágenes de prueba
└── docs/
    ├── FICHA_DATOS.md
    ├── GUIA_PUBLICACION.md
    └── GUION_EXPOSICION.md
```

## 6. Cómo ejecutarla

**En línea:** abrir el link del Space.

**Localmente:**
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="tu_clave"      # en Windows: set GEMINI_API_KEY=tu_clave
python app.py                         # abre http://127.0.0.1:7860
```

**Reentrenar:** abrir `notebooks/entrenamiento.ipynb` en Google Colab con GPU T4 → *Ejecutar todas* → se descarga `fundiscan_export.zip` con `modelo/` y `ejemplos/`.

## 7. Integrantes

| Integrante | Rol principal |
|---|---|
| _Nombre 1_ | Datos y notebook (secciones 1–4) |
| _Nombre 2_ | Modelos y métricas (secciones 5–10) |
| _Nombre 3_ | App, Gemini y publicación |

## 8. Declaración de uso de IA

| Herramienta | Para qué la usamos |
|---|---|
| Claude (Anthropic) | Estructura inicial del notebook y de `app.py`, redacción de la documentación y del guion de la exposición |
| Google Gemini (API) | Genera las recomendaciones dentro de la app |
| _Agregar otras (ChatGPT, Copilot, etc.)_ | |

Adaptamos el código a nuestro problema (tres enfoques comparados, umbral elegido por costo en validación, interfaz con identidad industrial y recomendación de respaldo). Ejecutamos el notebook y verificamos los resultados. Todos los integrantes revisamos y podemos explicar cada parte del código.

## 9. Ética y limitaciones

- No hay fotos de personas ni datos sensibles.
- Es un **prototipo educativo**, no un sistema de inspección certificado.
- **Sesgo de dominio:** el modelo solo conoce un tipo de pieza fotografiado en condiciones controladas. Con otras piezas, otra luz o fotos de celular puede equivocarse con confianza alta.
- **Mejoras futuras:** validar con fotos de otra línea, Grad-CAM para mostrar dónde está el defecto y clasificar el tipo de defecto.
