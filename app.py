"""
FundiScan · Inspección visual de piezas de fundición con IA
Certamen I · IA Generativa para la Gestión Industrial (UDD)

Flujo:  📷 imagen → 🧠 EfficientNet-B0 (clase + confianza) → 📊 resultado visual → 💬 Gemini → ✅ recomendación

Archivos que necesita (los genera notebooks/entrenamiento.ipynb):
    modelo/efficientnet_b0_fundicion.pt   pesos del modelo
    modelo/config.json                    clases, umbral, transformaciones y métricas en prueba
    ejemplos/*.jpeg                       imágenes de ejemplo (opcional)

La clave de Gemini se lee de la variable de entorno GEMINI_API_KEY
(en Hugging Face: Settings → Variables and secrets). Nunca se escribe en el código.
"""

import json
import os
from pathlib import Path

import gradio as gr
import timm
import torch
from PIL import Image
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuración y carga del modelo
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
CONFIG = json.loads((BASE / "modelo" / "config.json").read_text(encoding="utf-8"))

CLASES = CONFIG["clases"]                      # ['conforme', 'defectuosa'] (orden alfabético, igual que en el notebook)
POS = CONFIG["clase_positiva"]                 # 'defectuosa': la clase que nos importa detectar
I_POS = CLASES.index(POS)
UMBRAL = float(CONFIG["umbral"])               # si P(defectuosa) ≥ UMBRAL → defectuosa (elegido en validación)
CONFIANZA_MINIMA = 0.85                        # bajo esta confianza se exige revisión humana
METRICAS = CONFIG["metricas_prueba"]           # métricas del conjunto de prueba (no cambian con cada foto)

# Mismo modelo y mismas transformaciones que en el notebook (tf_eval)
modelo = timm.create_model(CONFIG["arquitectura"], pretrained=False, num_classes=len(CLASES))
modelo.load_state_dict(torch.load(BASE / "modelo" / "efficientnet_b0_fundicion.pt", map_location="cpu"))
modelo.eval()                                  # modo evaluación: sin dropout ni aumento de datos

preparar = transforms.Compose([
    transforms.Resize((CONFIG["tam"], CONFIG["tam"])),
    transforms.ToTensor(),
    transforms.Normalize(CONFIG["media"], CONFIG["desv"]),
])

NOMBRE_VISIBLE = {"conforme": "Conforme", "defectuosa": "Defectuosa"}
EJEMPLOS = sorted(str(p) for p in (BASE / "ejemplos").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clasificación
# ─────────────────────────────────────────────────────────────────────────────
def clasificar(imagen):
    """Recibe una imagen PIL y devuelve: tarjeta de veredicto, probabilidades, estado y botón de IA."""
    if imagen is None:
        gr.Warning("Primero sube o fotografía una pieza.")
        return "", None, None, gr.update(interactive=False), ""

    x = preparar(imagen.convert("RGB")).unsqueeze(0)          # 1 × 3 × 224 × 224
    with torch.no_grad():
        probs = torch.softmax(modelo(x), dim=1)[0].tolist()   # probabilidades que suman 1

    p_def = probs[I_POS]
    clase = POS if p_def >= UMBRAL else CLASES[1 - I_POS]     # se aplica el umbral del notebook
    confianza = probs[CLASES.index(clase)]                    # probabilidad de la clase decidida
    baja = confianza < CONFIANZA_MINIMA

    estado = {"clase": clase, "confianza": confianza, "p_defectuosa": p_def, "baja_confianza": baja}
    barras = {NOMBRE_VISIBLE[c]: p for c, p in zip(CLASES, probs)}
    return tarjeta_veredicto(estado), barras, estado, gr.update(interactive=True), ""


def tarjeta_veredicto(e):
    """HTML con el resultado: color según la clase y aviso si la confianza es baja."""
    if e["baja_confianza"]:
        estilo, icono, titulo = "dudosa", "⚠️", f"Resultado incierto · probablemente {e['clase']}"
        accion = "Confianza baja: <b>un inspector debe revisar la pieza</b> antes de liberarla o descartarla."
    elif e["clase"] == POS:
        estilo, icono, titulo = "defecto", "❌", "Pieza DEFECTUOSA"
        accion = "Separar la pieza, identificarla y enviarla a reinspección o a refusión."
    else:
        estilo, icono, titulo = "ok", "✅", "Pieza CONFORME"
        accion = "Puede continuar a la siguiente etapa (embalaje o mecanizado)."
    return f"""
    <div class="veredicto {estilo}">
      <div class="v-icono">{icono}</div>
      <div>
        <div class="v-titulo">{titulo}</div>
        <div class="v-conf">Confianza en esta foto: <b>{e['confianza']:.1%}</b>
             · P(defectuosa) = {e['p_defectuosa']:.3f} · umbral = {UMBRAL}</div>
        <div class="v-accion">{accion}</div>
      </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Recomendación con IA generativa (Google Gemini)
# ─────────────────────────────────────────────────────────────────────────────
# Alias "latest": Google los actualiza solo a la versión más nueva. Si uno falla se prueba el siguiente.
# Para fijar otro modelo, crear la variable GEMINI_MODEL en el Space (Settings → Variables and secrets).
MODELOS_GEMINI = [os.environ.get("GEMINI_MODEL", "gemini-flash-latest"), "gemini-flash-lite-latest", "gemini-3.5-flash-lite"]

INSTRUCCIONES = f"""Eres FundiScan, asistente de control de calidad en una planta de fundición que fabrica
impulsores (rodetes) de bombas sumergibles. Recibes el resultado de un modelo de visión que clasificó la foto
de UNA pieza como "conforme" o "defectuosa", con su confianza. No ves la imagen.

Reglas obligatorias:
1. Habla solo de inspección, calidad y proceso de fundición de esta pieza. Si te piden otra cosa, di que está fuera de alcance.
2. No inventes datos que no tienes (medidas, tipo exacto de defecto, ubicación): las causas son siempre POSIBLES, no confirmadas.
3. Si la confianza es menor a {CONFIANZA_MINIMA:.0%}, tu primera línea debe decir que la pieza requiere revisión de un inspector antes de liberarla o descartarla.
4. Recuerda en una línea final que FundiScan es un prototipo educativo y que la decisión final la toma una persona de calidad.
5. Escribe en español, tono profesional y claro, máximo 170 palabras, en Markdown con estas secciones en negrita:
   **Diagnóstico**, **Acción inmediata**, **Posibles causas y prevención**, **Registro**."""


def armar_prompt(e):
    return f"""Resultado de la inspección automática:
- Clase predicha: {e['clase']}
- Confianza del modelo en esta foto: {e['confianza']:.1%} ({'BAJA' if e['baja_confianza'] else 'adecuada'})
- Probabilidad de defecto: {e['p_defectuosa']:.1%} (umbral de rechazo: {UMBRAL:.0%})
- Desempeño del modelo en el conjunto de prueba: accuracy {METRICAS['accuracy']:.1%}, recall de piezas defectuosas {METRICAS['recall_defectuosa']:.1%}
Genera la recomendación para el inspector de calidad."""


def recomendacion_respaldo(e):
    """Texto fijo por si no hay clave o la API falla: la demo nunca queda en blanco."""
    if e["baja_confianza"]:
        inicio = "**Diagnóstico:** resultado incierto. **La pieza requiere revisión de un inspector** antes de decidir.\n\n"
    elif e["clase"] == POS:
        inicio = "**Diagnóstico:** el modelo detecta un posible defecto superficial.\n\n"
    else:
        inicio = "**Diagnóstico:** no se observan defectos visibles.\n\n"
    if e["baja_confianza"]:
        resto = ("**Acción inmediata:** retener la pieza (no liberarla ni descartarla) hasta que un inspector la revise "
                 "con lupa y buena iluminación; si es posible, tomar otra foto frontal.\n\n"
                 "**Posibles causas y prevención:** el resultado incierto puede deberse a la foto (luz, ángulo, foco) o a "
                 "un defecto pequeño (poro, rebaba). Verificar ambas cosas.\n\n"
                 "**Registro:** anotar el caso como dudoso con lote, turno y molde.")
    elif e["clase"] == POS:
        resto = ("**Acción inmediata:** segregar la pieza en la zona de no conformes y revisarla con lupa.\n\n"
                 "**Posibles causas y prevención:** porosidad por gases (revisar desgasificado y venteo del molde), "
                 "rechupes (revisar alimentadores y temperatura de colada), rebabas (revisar cierre del molde).\n\n"
                 "**Registro:** anotar lote, turno, hora y molde para buscar patrones.")
    else:
        resto = ("**Acción inmediata:** liberar a la siguiente etapa según el plan de muestreo.\n\n"
                 "**Posibles causas y prevención:** mantener los parámetros actuales de colada y moldeo.\n\n"
                 "**Registro:** registrar la inspección en la planilla del lote.")
    return inicio + resto + "\n\n_Prototipo educativo: la decisión final la toma una persona de calidad._"


def recomendar(estado):
    if not estado:
        return "Primero inspecciona una pieza."
    clave = os.environ.get("GEMINI_API_KEY")
    if not clave:
        return ("> ⚠️ No hay clave `GEMINI_API_KEY` configurada; se muestra la recomendación de respaldo.\n\n"
                + recomendacion_respaldo(estado))
    try:
        from google import genai
        from google.genai import types
        cliente = genai.Client(api_key=clave)
        ultimo_error = None
        for nombre in dict.fromkeys(MODELOS_GEMINI):          # prueba los modelos en orden hasta que uno responda
            try:
                r = cliente.models.generate_content(
                    model=nombre,
                    contents=armar_prompt(estado),
                    config=types.GenerateContentConfig(
                        system_instruction=INSTRUCCIONES,
                        temperature=0.3,                      # respuestas sobrias y consistentes
                        max_output_tokens=8192,                # holgado: los modelos nuevos también "piensan" con estos tokens
                    ),
                )
                if r.text:
                    return f"{r.text.strip()}\n\n<sub>Generado por {nombre}</sub>"
            except Exception as err:                          # modelo no disponible → probar el siguiente
                ultimo_error = err
        raise RuntimeError(ultimo_error or "respuesta vacía")
    except Exception as err:
        print("Error Gemini:", err)                           # queda en los Logs del Space para diagnosticar
        return (f"> ⚠️ No se pudo conectar con Gemini ({str(err)[:120]}). "
                "Se muestra la recomendación de respaldo.\n\n" + recomendacion_respaldo(estado))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Interfaz (Gradio)
# ─────────────────────────────────────────────────────────────────────────────
TEMA = gr.themes.Soft(
    primary_hue="orange",                      # naranjo de seguridad industrial
    neutral_hue="slate",                       # gris acero
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

CSS = """
.cabecera {display:flex; align-items:center; gap:18px; padding:22px 26px; border-radius:16px;
           background:linear-gradient(120deg,#1e293b 0%,#334155 60%,#c2410c 140%); color:#f8fafc;}
.cabecera h1 {margin:0; font-size:2rem; color:#f8fafc;}
.cabecera p {margin:4px 0 0; color:#cbd5e1;}
.cabecera svg {flex-shrink:0;}
.veredicto {display:flex; gap:16px; align-items:flex-start; padding:18px 20px; border-radius:14px; border:2px solid;}
.veredicto.ok {background:#ecfdf5; border-color:#16a34a;}
.veredicto.ok, .veredicto.ok * {color:#14532d !important;}
.veredicto.defecto {background:#fef2f2; border-color:#dc2626;}
.veredicto.defecto, .veredicto.defecto * {color:#7f1d1d !important;}
.veredicto.dudosa {background:#fffbeb; border-color:#d97706;}
.veredicto.dudosa, .veredicto.dudosa * {color:#78350f !important;}
.v-icono {font-size:2.4rem; line-height:1;}
.v-titulo {font-size:1.45rem; font-weight:700;}
.v-conf {margin-top:4px; font-size:.95rem;}
.v-accion {margin-top:8px;}
.aviso {padding:10px 14px; border-radius:10px; background:#fff7ed; border:1px solid #fdba74; font-size:.9rem;}
.aviso, .aviso * {color:#7c2d12 !important;}
.tarjetas {display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px;}
.tarjeta {padding:16px; border-radius:12px; background:#f8fafc; border:1px solid #e2e8f0; color:#0f172a;}
.tarjeta .valor {font-size:1.8rem; font-weight:700; color:#c2410c !important;}
.tarjeta .nombre {font-size:.85rem; color:#475569 !important;}
table.cm {border-collapse:collapse; margin-top:8px;}
table.cm td, table.cm th {border:1px solid #cbd5e1; padding:8px 14px; text-align:center;}
table.cm td.malo {background:#fee2e2; font-weight:700;}
table.cm td.malo, table.cm td.malo * {color:#7f1d1d !important;}
table.cm td.bien {background:#dcfce7;}
table.cm td.bien, table.cm td.bien * {color:#14532d !important;}
.tabla-scroll {overflow-x:auto;}
table.comp {border-collapse:collapse; width:100%;}
table.comp th, table.comp td {border:1px solid #cbd5e1; padding:8px 12px; white-space:nowrap; text-align:left;}
/* Celular: pestañas más compactas para que las tres quepan sin esconderse en el menú "⋯" */
@media (max-width: 640px) {
  .tab-container button {padding:6px 8px !important; font-size:.85rem !important;}
  .cabecera {padding:16px 18px;}
  .cabecera h1 {font-size:1.6rem;}
}
"""

ICONO = """<svg width="64" height="64" viewBox="0 0 64 64" fill="none" aria-hidden="true">
  <circle cx="32" cy="32" r="27" stroke="#fb923c" stroke-width="4"/>
  <circle cx="32" cy="32" r="8" fill="#fb923c"/>
  <path d="M32 24 C40 14 50 16 54 22 M40 32 C50 40 48 50 42 54 M32 40 C24 50 14 48 10 42 M24 32 C14 24 16 14 22 10"
        stroke="#f8fafc" stroke-width="3.5" stroke-linecap="round"/>
</svg>"""

CABECERA = f"""
<div class="cabecera">{ICONO}
  <div><h1>FundiScan</h1>
  <p>Inspección visual de impulsores de fundición con IA · ¿conforme o defectuosa?</p></div>
</div>"""

AVISO = f"""<div class="aviso">🎓 <b>Prototipo educativo</b> (Certamen I, UDD). No reemplaza la inspección de calidad:
si la confianza es menor a {CONFIANZA_MINIMA:.0%} la pieza debe revisarla una persona. Entrenado solo con fotos de
impulsores de una planta, con una cámara y una iluminación; con otras piezas o fotos el resultado no es confiable.</div>"""


def html_desempeno():
    m, cm = METRICAS, CONFIG["matriz_confusion"]
    tarjetas = "".join(
        f'<div class="tarjeta"><div class="valor">{v:.1%}</div><div class="nombre">{n}</div></div>'
        for n, v in [("Accuracy", m["accuracy"]), ("Recall piezas defectuosas ⭐", m["recall_defectuosa"]),
                     ("Precisión piezas defectuosas", m["precision_defectuosa"]),
                     ("Accuracy balanceada", m["accuracy_balanceada"])])
    i_ok = 1 - I_POS
    matriz = f"""
    <table class="cm">
      <tr><th></th><th>Predice conforme</th><th>Predice defectuosa</th></tr>
      <tr><th>Real conforme</th><td class="bien">{cm[i_ok][i_ok]}</td><td>{cm[i_ok][I_POS]} <small>(falsa alarma)</small></td></tr>
      <tr><th>Real defectuosa</th><td class="malo">{cm[I_POS][i_ok]} <small>(se escapa 🔴)</small></td><td class="bien">{cm[I_POS][I_POS]}</td></tr>
    </table>"""
    return f'<div class="tarjetas">{tarjetas}</div><h3>Matriz de confusión ({CONFIG["n_prueba"]} piezas de prueba)</h3>{matriz}'


def html_comparacion():
    """Tabla de los 3 enfoques; en pantallas angostas se desplaza de lado en vez de cortar las palabras."""
    filas = "".join(
        f"<tr><td>{nombre}</td><td>{r['accuracy']:.1%}</td><td>{r['accuracy_balanceada']:.1%}</td>"
        f"<td>{r['recall_defectuosa']:.1%}</td><td>{r.get('tamano_MB', '–')} MB</td></tr>"
        for nombre, r in sorted(CONFIG["comparacion"].items()))   # A, B, C
    return ('<div class="tabla-scroll"><table class="comp"><tr><th>Enfoque</th><th>Accuracy</th>'
            '<th>Acc. balanceada</th><th>Recall defectuosa</th><th>Tamaño</th></tr>' + filas + '</table></div>')


with gr.Blocks(title="FundiScan · Inspección de fundición con IA") as demo:
    estado = gr.State(None)                    # guarda el último resultado para enviarlo al LLM
    gr.HTML(CABECERA)

    with gr.Tabs():
        # ── Pestaña 1: inspección ───────────────────────────────────────────
        with gr.Tab("🔍 Inspección"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    imagen = gr.Image(type="pil", label="Foto de la pieza (vista frontal)",
                                      sources=["upload", "webcam", "clipboard"], height=330)
                    with gr.Row():
                        btn_limpiar = gr.ClearButton(value="🧹 Limpiar")
                        btn_inspeccionar = gr.Button("🔍 Inspeccionar pieza", variant="primary")
                    ejemplos_slot = gr.Column()        # los ejemplos se agregan más abajo (necesitan las salidas)
                with gr.Column(scale=6):
                    veredicto = gr.HTML()
                    barras = gr.Label(label="Probabilidad de cada clase para esta foto", num_top_classes=2,
                                      show_heading=False)   # el veredicto (con umbral) ya está arriba
                    gr.Markdown(f"La **confianza** es la seguridad del modelo en *esta foto*. El desempeño general "
                                f"del modelo (accuracy **{METRICAS['accuracy']:.1%}** en {CONFIG['n_prueba']} piezas de "
                                f"prueba) está en la pestaña 📊.")
                    btn_ia = gr.Button("💬 Generar recomendación con IA (Gemini)", variant="secondary",
                                       interactive=False)
                    recomendacion = gr.Markdown(label="Recomendación", min_height=60)
            gr.HTML(AVISO)

        # ── Pestaña 2: desempeño del modelo ────────────────────────────────
        with gr.Tab("📊 Desempeño"):
            gr.Markdown("### Resultados en el conjunto de **prueba** (fotos que el modelo nunca vio al entrenar)")
            gr.HTML(html_desempeno())
            gr.Markdown(f"""
**¿Accuracy o confianza?** La *accuracy* ({METRICAS['accuracy']:.1%}) mide cuántas piezas de prueba clasificó bien
el modelo y **no cambia** al subir una foto. La *confianza* que ves en la pestaña de inspección es la probabilidad
que el modelo asigna a **una foto en particular**.

**Umbral de rechazo = {UMBRAL}.** Dejar pasar una pieza defectuosa cuesta mucho más que reinspeccionar una buena,
por eso el umbral se eligió en **validación** minimizando el costo de ambos errores
(falso negativo = {CONFIG['costos']['falso_negativo']} × falso positivo).
""")
            gr.Markdown(f"### Comparación de enfoques (mismas {CONFIG['n_prueba']} imágenes de prueba, umbral 0,5)")
            gr.HTML(html_comparacion())
            gr.Markdown("Elegimos **EfficientNet-B0** por su recall en piezas defectuosas, su tamaño (~16 MB) "
                        "y lo simple que es usarlo en la app.")

        # ── Pestaña 3: acerca del proyecto ─────────────────────────────────
        with gr.Tab("ℹ️ Acerca"):
            gr.Markdown(f"""
### ¿Qué es FundiScan?
Una ayuda para el **inspector de calidad** al final de una línea de fundición de impulsores de bombas sumergibles.
Clasifica la foto de la pieza como **conforme** o **defectuosa** (poros, rechupes, rebabas, grietas) y genera
una recomendación con IA sobre qué hacer y qué revisar en el proceso.

### Cómo usarla
1. Sube una foto frontal de la pieza, tómala con la cámara o haz clic en un ejemplo.
2. Presiona **Inspeccionar pieza** y revisa el veredicto y las probabilidades.
3. Presiona **Generar recomendación con IA**. Al LLM solo se envía texto (clase y confianza), nunca la imagen.

### Datos
*Casting product image data for quality inspection*, de R. Dabhi y Pilot Technocast (India), publicado en
[Kaggle]({CONFIG['dataset']}). Son fotos de 300×300 px en escala de grises; usamos
{sum(CONFIG['conteo']['train'].values())} para entrenar, {sum(CONFIG['conteo']['val'].values())} para validar y
{CONFIG['n_prueba']} para la prueba.

### Equipo
_Nombre 1_ · _Nombre 2_ · _Nombre 3_ — IA Generativa para la Gestión Industrial, UDD 2026.

Código: [repositorio en GitHub](https://github.com/sebastianignacio5755-coder/fundiscan)
""")

    # ── Eventos ─────────────────────────────────────────────────────────────
    salidas = [veredicto, barras, estado, btn_ia, recomendacion]
    if EJEMPLOS:
        with ejemplos_slot:                    # clic en un ejemplo → se carga y se inspecciona de inmediato
            gr.Examples(examples=EJEMPLOS, inputs=imagen, outputs=salidas, fn=clasificar,
                        run_on_click=True, cache_examples=False,
                        label="Piezas de ejemplo del conjunto de prueba (haz clic para probar)")
    btn_inspeccionar.click(clasificar, inputs=imagen, outputs=salidas)
    imagen.upload(clasificar, inputs=imagen, outputs=salidas)        # inspecciona apenas se sube la foto
    btn_limpiar.add([imagen, veredicto, barras, recomendacion])
    btn_limpiar.click(lambda: (None, gr.update(interactive=False)), outputs=[estado, btn_ia])
    btn_ia.click(lambda: "⏳ Consultando a Gemini…", outputs=recomendacion).then(
        recomendar, inputs=estado, outputs=recomendacion)


if __name__ == "__main__":
    demo.launch(theme=TEMA, css=CSS)
