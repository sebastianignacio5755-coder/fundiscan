# 🎤 Guion de la exposición (10 minutos · 3 integrantes)

**Roles:** **A** = datos y problema · **B** = modelos y métricas · **C** = app y Gemini.
Todos deben poder responder sobre **todo** el código: el profesor puede preguntarle a cualquiera.

| Bloque | Tiempo | Quién | Diapositiva / pantalla |
|---|---|---|---|
| 1. Problema y motivación | 1:30 | A | Foto de una pieza buena y una mala · tabla de costos de error |
| 2. Datos | 1:30 | A | Ficha · gráfico de barras por clase · limitaciones |
| 3. Modelo | 2:00 | B | Tabla de comparación · matriz de confusión · umbral |
| 4. Demo en vivo | 3:00 | C | App publicada (celular o navegador) |
| 5. Código | 1:30 | B + C | Notebook (entrenamiento + evaluar) · `app.py` (clasificar + recomendar) |
| 6. Cierre | 0:30 | A | Aprendizajes, riesgos y mejoras |

---

## 1. Problema (A · 1:30)
> "En una fundición, cuando el metal se enfría pueden quedar poros, rechupes, rebabas o grietas. Estas piezas son impulsores de bombas sumergibles: si una defectuosa llega al cliente, la bomba puede fallar bajo el agua, y eso significa garantía, retiro del lote y daño a la marca. Hoy la inspección es visual y manual: lenta, cansadora y distinta según cada inspector. **FundiScan** es una ayuda para el inspector al final de la línea: sube la foto, recibe *conforme* o *defectuosa* con su confianza y una recomendación de qué hacer. No lo reemplaza: le ayuda a priorizar."
>
> "Para nosotros el error más grave es **dejar pasar una pieza defectuosa**. Una falsa alarma solo cuesta una reinspección."

## 2. Datos (A · 1:30)
- Dataset público de Kaggle con fotos reales de una fundición en India (Pilot Technocast, 2020). Son **7.348 fotos** de 300×300 px.
- Dividimos en **train / val / test**: el test original quedó intacto (715 fotos) y separamos el 15 % del train como validación, estratificado y con semilla fija.
- **Desbalance:** ≈ 57–63 % son defectuosas. Por eso un modelo "flojo" que siempre dijera defectuosa ya tendría 63 % de accuracy, y además miramos accuracy balanceada y recall.
- **Limitaciones:** una sola planta, cámara y luz; posibles fotos casi repetidas entre conjuntos (pueden inflar las métricas); no dice el tipo de defecto.

## 3. Modelo (B · 2:00)
> "Comparamos tres enfoques con **las mismas 715 fotos de prueba** y las mismas métricas:
> – **CLIP sin entrenar** como línea base: obtuvo __ %, cerca del azar, porque un modelo general no sabe de control de calidad.
> – **EfficientNet-B0** con transfer learning: __ % de recall en defectuosas.
> – **YOLO11n-cls**: __ %.
> Elegimos **EfficientNet** por el recall en defectuosas, porque pesa 16 MB y porque en la app solo necesita torch y timm."
>
> "Además **no usamos el umbral de 0,5**: lo elegimos en **validación**, con un costo 20 veces mayor para una pieza defectuosa que se escapa. Con umbral __ se escapan __ de 453 piezas malas y hay __ falsas alarmas."

Mostrar la matriz de confusión y señalar la casilla **real defectuosa / predice conforme**.

## 4. Demo (C · 3:00)
1. Abrir el link público, idealmente **desde el celular** proyectado (muestra que está en internet).
2. Clic en un ejemplo **defectuoso** → tarjeta roja, confianza y barras.
3. **Generar recomendación** → leer en voz alta la acción inmediata y las posibles causas.
4. Ejemplo **conforme** → tarjeta verde.
5. Ejemplo **dudoso**, o subir una foto que no es una pieza (por ejemplo, una moneda) → tarjeta ámbar y aviso de revisión humana. Explicar que *el modelo no sabe decir "no sé"*: por eso existe el aviso.
6. Pestaña 📊: "la **accuracy** es del modelo completo en prueba; la **confianza** es de esta foto".

🎥 Tener el video de respaldo abierto en otra pestaña.

## 5. Código (B + C · 1:30)
**Notebook (B):**
- `train_test_split(..., test_size=0.15, random_state=42)` por clase → división estratificada.
- `timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)` → reutiliza lo aprendido en ImageNet y cambia la salida a 2 clases.
- El ciclo de entrenamiento guarda la época con **menor pérdida de validación**.
- La función `evaluar()` es la misma para los tres modelos → comparación justa.

**app.py (C):**
- `cargar_modelo()` usa `@st.cache_resource`: el modelo se carga una sola vez, no en cada clic.
- `clasificar()`: aplica las mismas transformaciones que el notebook → `softmax` → aplica el **umbral** → calcula la confianza.
- `recomendar()`: lee la clave desde `st.secrets["GEMINI_API_KEY"]` (Secrets de Streamlit Cloud) y envía **solo texto** a Gemini con `INSTRUCCIONES` (prompt responsable). Si falla, muestra una recomendación de respaldo.

## 6. Cierre (A · 0:30)
> "Aprendimos que un modelo general no basta: hay que adaptarlo con datos del dominio, y que la accuracy no es lo importante, sino cuántas piezas malas se escapan. El principal riesgo es que solo conoce una pieza y una cámara. Como mejora, validaríamos con fotos de otra línea y agregaríamos Grad-CAM para mostrar dónde está el defecto."

---

## ❓ Preguntas probables del profesor (y respuestas cortas)

**¿Qué es transfer learning y por qué lo usaron?**
Partir de una red que ya aprendió a ver bordes, texturas y formas con millones de imágenes (ImageNet) y solo ajustarla a nuestro problema. Con pocas épocas y pocos datos llega a un desempeño alto.

**¿Por qué `lr=1e-4` y no algo más grande?**
Con una tasa grande se "borra" lo aprendido en ImageNet. Con una pequeña se ajusta suavemente.

**¿Qué hace `softmax`?**
Convierte las dos salidas de la red en probabilidades que suman 1.

**¿Cuál es la diferencia entre la accuracy y la confianza?**
La accuracy es el % de aciertos del modelo sobre las 715 fotos de prueba y no cambia. La confianza es la probabilidad que el modelo da a la clase elegida en *una* foto.

**¿Por qué no eligieron el umbral mirando el test?**
Porque entonces el test dejaría de ser "nuevo" para el modelo y las métricas serían optimistas. El test se usa una sola vez, al final.

**¿Para qué sirve el conjunto de validación?**
Para elegir la mejor época y el umbral sin tocar la prueba.

**¿Qué pasa si suben una foto de un gato?**
El modelo igual responde conforme o defectuosa, porque no tiene la opción "no es una pieza". Por eso hay aviso de revisión humana con confianza baja y la advertencia de uso. Una mejora sería agregar una clase "otro" o un detector de fuera de distribución.

**¿Le mandan la imagen a Gemini?**
No. Solo se envía texto: clase, confianza, probabilidad de defecto y métricas del modelo.

**¿Dónde está la clave API?**
En *Settings → Secrets* de la app en Streamlit Community Cloud, como `GEMINI_API_KEY`. El código la lee con `st.secrets` (o `os.environ` en local). No está en GitHub.

**¿Qué hace el prompt para ser responsable?**
Limita el tema a la calidad en fundición, prohíbe inventar datos (las causas son "posibles"), obliga a pedir revisión humana con confianza menor a 85 % y recuerda que es un prototipo.

**¿Por qué EfficientNet y no YOLO, si YOLO es más liviano?**
Por el recall en defectuosas (ver la tabla), porque 16 MB cabe sin problemas y porque en la app solo necesita torch y timm. YOLO es una buena alternativa si tuviera que correr en un equipo muy limitado.

**¿Por qué las métricas son tan altas? ¿Hay que desconfiar?**
Un poco sí: las fotos son muy uniformes (misma cámara y luz) y pueden existir fotos casi repetidas entre train y test. En una planta real habría que validar con fotos de otro día o de otra línea.

**¿Qué es el aumento de datos y por qué giraron las fotos?**
Son variaciones artificiales que se aplican solo al entrenar, para que el modelo no memorice. La pieza es circular, así que girarla no cambia si está buena o mala.
