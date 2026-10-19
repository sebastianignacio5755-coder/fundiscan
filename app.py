"""
FundiScan · Inspección visual de piezas de fundición con IA
Certamen I · IA Generativa para la Gestión Industrial (UDD)

Flujo:  📷 imagen → 🧠 EfficientNet-B0 (clase + confianza) → 📊 resultado visual → 💬 Gemini → ✅ recomendación

App hecha con Streamlit y publicada en Streamlit Community Cloud.
Ejecutar localmente:  streamlit run app.py

Archivos que necesita (los genera notebooks/entrenamiento.ipynb):
    modelo/efficientnet_b0_fundicion.pt   pesos del modelo
    modelo/config.json                    clases, umbral, transformaciones y métricas en prueba
    ejemplos/*.jpeg                       imágenes de ejemplo

La clave de Gemini NUNCA se escribe en el código: se lee de los Secrets de Streamlit Cloud
(Settings → Secrets: GEMINI_API_KEY = "...") o de la variable de entorno GEMINI_API_KEY.
"""

import hashlib
import json
import os
from pathlib import Path

import streamlit as st
import timm
import torch
from PIL import Image
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# 0. Página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FundiScan · Inspección de fundición con IA", page_icon="⚙️", layout="wide")

BASE = Path(__file__).parent
CONFIG = json.loads((BASE / "modelo" / "config.json").read_text(encoding="utf-8"))

CLASES = CONFIG["clases"]                      # ['conforme', 'defectuosa'] (orden alfabético, igual que en el notebook)
POS = CONFIG["clase_positiva"]                 # 'defectuosa': la clase que nos importa detectar
I_POS = CLASES.index(POS)
UMBRAL = float(CONFIG["umbral"])               # si P(defectuosa) ≥ UMBRAL → defectuosa (elegido en validación)
CONFIANZA_MINIMA = 0.85                        # bajo esta confianza se exige revisión humana
METRICAS = CONFIG["metricas_prueba"]           # métricas del conjunto de prueba (no cambian con cada foto)
NOMBRE_VISIBLE = {"conforme": "Conforme", "defectuosa": "Defectuosa"}
EJEMPLOS = sorted(p for p in (BASE / "ejemplos").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Modelo (se carga una sola vez y queda en caché)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    """Mismo modelo y mismas transformaciones que en el notebook (tf_eval)."""
    modelo = timm.create_model(CONFIG["arquitectura"], pretrained=False, num_classes=len(CLASES))
    modelo.load_state_dict(torch.load(BASE / "modelo" / "efficientnet_b0_fundicion.pt", map_location="cpu"))
    modelo.eval()                              # modo evaluación: sin dropout ni aumento de datos
    preparar = transforms.Compose([
        transforms.Resize((CONFIG["tam"], CONFIG["tam"])),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["media"], CONFIG["desv"]),
    ])
    return modelo, preparar


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clasificación
# ─────────────────────────────────────────────────────────────────────────────
def clasificar(imagen):
    """Recibe una imagen PIL y devuelve clase, confianza y probabilidades de ambas clases."""
    modelo, preparar = cargar_modelo()
    x = preparar(imagen.convert("RGB")).unsqueeze(0)          # 1 × 3 × 224 × 224
    with torch.no_grad():
        probs = torch.softmax(modelo(x), dim=1)[0].tolist()   # probabilidades que suman 1

    p_def = probs[I_POS]
    clase = POS if p_def >= UMBRAL else CLASES[1 - I_POS]     # se aplica el umbral del notebook
    confianza = probs[CLASES.index(clase)]                    # probabilidad de la clase decidida
    return {"clase": clase, "confianza": confianza, "p_defectuosa": p_def,
            "baja_confianza": confianza < CONFIANZA_MINIMA, "probs": dict(zip(CLASES, probs))}


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
MODELOS_GEMINI = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.5-flash-lite"]

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


def leer_secreto(nombre):
    """Busca primero en los Secrets de Streamlit y luego en las variables de entorno."""
    try:
        if nombre in st.secrets:
            return st.secrets[nombre]
    except Exception:                          # no hay archivo de secrets (por ejemplo, en local)
        pass
    return os.environ.get(nombre)


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
        return ("**Diagnóstico:** resultado incierto. **La pieza requiere revisión de un inspector** antes de decidir.\n\n"
                "**Acción inmediata:** retener la pieza (no liberarla ni descartarla) hasta que un inspector la revise "
                "con lupa y buena iluminación; si es posible, tomar otra foto frontal.\n\n"
                "**Posibles causas y prevención:** el resultado incierto puede deberse a la foto (luz, ángulo, foco) o a "
                "un defecto pequeño (poro, rebaba). Verificar ambas cosas.\n\n"
                "**Registro:** anotar el caso como dudoso con lote, turno y molde.\n\n"
                "_Prototipo educativo: la decisión final la toma una persona de calidad._")
    if e["clase"] == POS:
        return ("**Diagnóstico:** el modelo detecta un posible defecto superficial.\n\n"
                "**Acción inmediata:** segregar la pieza en la zona de no conformes y revisarla con lupa.\n\n"
                "**Posibles causas y prevención:** porosidad por gases (revisar desgasificado y venteo del molde), "
                "rechupes (revisar alimentadores y temperatura de colada), rebabas (revisar cierre del molde).\n\n"
                "**Registro:** anotar lote, turno, hora y molde para buscar patrones.\n\n"
                "_Prototipo educativo: la decisión final la toma una persona de calidad._")
    return ("**Diagnóstico:** no se observan defectos visibles.\n\n"
            "**Acción inmediata:** liberar a la siguiente etapa según el plan de muestreo.\n\n"
            "**Posibles causas y prevención:** mantener los parámetros actuales de colada y moldeo.\n\n"
            "**Registro:** registrar la inspección en la planilla del lote.\n\n"
            "_Prototipo educativo: la decisión final la toma una persona de calidad._")


def recomendar(estado):
    clave = leer_secreto("GEMINI_API_KEY")
    if not clave:
        return ("> ⚠️ No hay clave `GEMINI_API_KEY` configurada; se muestra la recomendación de respaldo.\n\n"
                + recomendacion_respaldo(estado))
    modelo_extra = leer_secreto("GEMINI_MODEL")          # opcional: fijar otro modelo desde los Secrets
    modelos = ([modelo_extra] if modelo_extra else []) + MODELOS_GEMINI
    try:
        from google import genai
        from google.genai import types
        cliente = genai.Client(api_key=clave)
        ultimo_error = None
        for nombre in dict.fromkeys(modelos):  # prueba los modelos en orden hasta que uno responda
            try:
                r = cliente.models.generate_content(
                    model=nombre,
                    contents=armar_prompt(estado),
                    config=types.GenerateContentConfig(
                        system_instruction=INSTRUCCIONES,
                        temperature=0.3,                # respuestas sobrias y consistentes
                        max_output_tokens=8192,         # holgado: los modelos nuevos también "piensan" con estos tokens
                    ),
                )
                if r.text:
                    return f"{r.text.strip()}\n\n<sub>Generado por {nombre}</sub>"
            except Exception as err:                   # modelo no disponible → probar el siguiente
                ultimo_error = err
        raise RuntimeError(ultimo_error or "respuesta vacía")
    except Exception as err:
        print("Error Gemini:", err)                    # queda en los registros de la app para diagnosticar
        return (f"> ⚠️ No se pudo conectar con Gemini ({str(err)[:120]}). "
                "Se muestra la recomendación de respaldo.\n\n" + recomendacion_respaldo(estado))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Estilo e identidad visual (colores de seguridad industrial: gris acero + naranjo)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.cabecera {display:flex; align-items:center; gap:18px; padding:22px 26px; border-radius:16px; margin-bottom:8px;
           background:linear-gradient(120deg,#1e293b 0%,#334155 60%,#c2410c 140%);}
.cabecera h1 {margin:0; padding:0; font-size:2rem; color:#f8fafc !important;}
.cabecera p {margin:4px 0 0; color:#cbd5e1 !important;}
.cabecera svg {flex-shrink:0;}
.veredicto {display:flex; gap:16px; align-items:flex-start; padding:18px 20px; border-radius:14px; border:2px solid; margin-bottom:12px;}
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
.aviso {padding:10px 14px; border-radius:10px; background:#fff7ed; border:1px solid #fdba74; font-size:.9rem; margin-top:16px;}
.aviso, .aviso * {color:#7c2d12 !important;}
.tarjetas {display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:12px;}
.tarjeta {padding:16px; border-radius:12px; background:#f8fafc; border:1px solid #e2e8f0;}
.tarjeta .valor {font-size:1.8rem; font-weight:700; color:#c2410c !important;}
.tarjeta .nombre {font-size:.85rem; color:#475569 !important;}
.tabla-scroll {overflow-x:auto;}
table.cm, table.comp {border-collapse:collapse; margin-top:8px;}
table.comp {width:100%;}
table.cm td, table.cm th, table.comp td, table.comp th {border:1px solid #cbd5e1; padding:8px 12px;}
table.cm td, table.cm th {text-align:center;}
table.comp th, table.comp td {white-space:nowrap; text-align:left;}
table.cm td.malo {background:#fee2e2; font-weight:700;}
table.cm td.malo, table.cm td.malo * {color:#7f1d1d !important;}
table.cm td.bien {background:#dcfce7;}
table.cm td.bien, table.cm td.bien * {color:#14532d !important;}
</style>
""", unsafe_allow_html=True)

ICONO = """<svg width="64" height="64" viewBox="0 0 64 64" fill="none" aria-hidden="true">
  <circle cx="32" cy="32" r="27" stroke="#fb923c" stroke-width="4"/>
  <circle cx="32" cy="32" r="8" fill="#fb923c"/>
  <path d="M32 24 C40 14 50 16 54 22 M40 32 C50 40 48 50 42 54 M32 40 C24 50 14 48 10 42 M24 32 C14 24 16 14 22 10"
        stroke="#f8fafc" stroke-width="3.5" stroke-linecap="round"/>
</svg>"""

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
    matriz = f"""<div class="tabla-scroll"><table class="cm">
      <tr><th></th><th>Predice conforme</th><th>Predice defectuosa</th></tr>
      <tr><th>Real conforme</th><td class="bien">{cm[i_ok][i_ok]}</td><td>{cm[i_ok][I_POS]} <small>(falsa alarma)</small></td></tr>
      <tr><th>Real defectuosa</th><td class="malo">{cm[I_POS][i_ok]} <small>(se escapa 🔴)</small></td><td class="bien">{cm[I_POS][I_POS]}</td></tr>
    </table></div>"""
    return f'<div class="tarjetas">{tarjetas}</div><h4>Matriz de confusión ({CONFIG["n_prueba"]} piezas de prueba)</h4>{matriz}'


def html_comparacion():
    """Tabla de los 3 enfoques; en pantallas angostas se desplaza de lado en vez de cortar las palabras."""
    filas = "".join(
        f"<tr><td>{nombre}</td><td>{r['accuracy']:.1%}</td><td>{r['accuracy_balanceada']:.1%}</td>"
        f"<td>{r['recall_defectuosa']:.1%}</td><td>{r.get('tamano_MB', '–')} MB</td></tr>"
        for nombre, r in sorted(CONFIG["comparacion"].items()))   # A, B, C
    return ('<div class="tabla-scroll"><table class="comp"><tr><th>Enfoque</th><th>Accuracy</th>'
            '<th>Acc. balanceada</th><th>Recall defectuosa</th><th>Tamaño</th></tr>' + filas + '</table></div>')


# ─────────────────────────────────────────────────────────────────────────────
# 5. Interfaz
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""<div class="cabecera">{ICONO}
  <div><h1>FundiScan</h1>
  <p>Inspección visual de impulsores de fundición con IA · ¿conforme o defectuosa?</p></div>
</div>""", unsafe_allow_html=True)

with st.sidebar:                               # barra lateral: resumen y ayuda
    st.markdown("### ⚙️ FundiScan")
    st.caption("Ayuda para el inspector de calidad al final de la línea de fundición.")
    st.metric("Accuracy en prueba", f"{METRICAS['accuracy']:.1%}")
    st.metric("Piezas defectuosas detectadas (recall)", f"{METRICAS['recall_defectuosa']:.1%}")
    st.markdown(f"**Umbral de rechazo:** P(defectuosa) ≥ {UMBRAL}  \n"
                f"**Revisión humana:** confianza < {CONFIANZA_MINIMA:.0%}")
    st.divider()
    st.markdown("**Cómo usarla**\n1. Elige un ejemplo, sube una foto o usa la cámara.\n"
                "2. Revisa el veredicto y las probabilidades.\n3. Pide la recomendación con IA.")

tab_insp, tab_desemp, tab_acerca = st.tabs(["🔍 Inspección", "📊 Desempeño", "ℹ️ Acerca"])

# ── Pestaña 1: inspección ──────────────────────────────────────────────────
with tab_insp:
    col_img, col_res = st.columns([5, 6], gap="large")

    with col_img:
        fuente = st.radio("¿Cómo quieres cargar la pieza?", ["🧪 Ejemplo", "📁 Subir foto", "📷 Cámara"],
                          horizontal=True)
        imagen, nombre_img = None, None
        if fuente == "📁 Subir foto":
            archivo = st.file_uploader("Foto frontal de la pieza", type=["jpg", "jpeg", "png"])
            if archivo:
                imagen, nombre_img = Image.open(archivo), archivo.name
        elif fuente == "📷 Cámara":
            foto = st.camera_input("Toma una foto frontal de la pieza")
            if foto:
                imagen, nombre_img = Image.open(foto), "foto de cámara"
        elif EJEMPLOS:
            elegido = st.selectbox("Piezas de ejemplo del conjunto de prueba", EJEMPLOS,
                                   format_func=lambda p: p.stem.replace("_", " · "))
            imagen, nombre_img = Image.open(elegido), elegido.name
        else:
            st.info("No hay imágenes de ejemplo en la carpeta ejemplos/.")
        if imagen is not None:
            st.image(imagen, caption=nombre_img, width="stretch")

    with col_res:
        if imagen is None:
            st.info("👈 Carga una foto de la pieza para inspeccionarla.")
        else:
            resultado = clasificar(imagen)
            st.markdown(tarjeta_veredicto(resultado), unsafe_allow_html=True)

            st.markdown("**Probabilidad de cada clase para esta foto**")
            for clase in CLASES:
                p = resultado["probs"][clase]
                st.progress(p, text=f"{NOMBRE_VISIBLE[clase]}: {p:.1%}")
            st.caption(f"La **confianza** es la seguridad del modelo en *esta foto*. El desempeño general del modelo "
                       f"(accuracy **{METRICAS['accuracy']:.1%}** en {CONFIG['n_prueba']} piezas de prueba) "
                       f"está en la pestaña 📊.")

            # la recomendación se guarda por foto: si cambia la imagen, se borra
            clave_img = hashlib.md5(imagen.tobytes()).hexdigest()
            if st.session_state.get("rec_img") != clave_img:
                st.session_state["rec_img"], st.session_state["rec"] = clave_img, None
            if st.button("💬 Generar recomendación con IA (Gemini)", type="primary", width="stretch"):
                with st.spinner("Consultando a Gemini…"):
                    st.session_state["rec"] = recomendar(resultado)
            if st.session_state.get("rec"):
                with st.container(border=True):
                    st.markdown(st.session_state["rec"], unsafe_allow_html=True)

    st.markdown(AVISO, unsafe_allow_html=True)

# ── Pestaña 2: desempeño del modelo ────────────────────────────────────────
with tab_desemp:
    st.markdown("### Resultados en el conjunto de **prueba** (fotos que el modelo nunca vio al entrenar)")
    st.markdown(html_desempeno(), unsafe_allow_html=True)
    st.markdown(f"""
**¿Accuracy o confianza?** La *accuracy* ({METRICAS['accuracy']:.1%}) mide cuántas piezas de prueba clasificó bien
el modelo y **no cambia** al subir una foto. La *confianza* que ves en la pestaña de inspección es la probabilidad
que el modelo asigna a **una foto en particular**.

**Umbral de rechazo = {UMBRAL}.** Dejar pasar una pieza defectuosa cuesta mucho más que reinspeccionar una buena,
por eso el umbral se eligió en **validación** minimizando el costo de ambos errores
(falso negativo = {CONFIG['costos']['falso_negativo']} × falso positivo).
""")
    st.markdown(f"#### Comparación de enfoques (mismas {CONFIG['n_prueba']} imágenes de prueba, umbral 0,5)")
    st.markdown(html_comparacion(), unsafe_allow_html=True)
    st.markdown("YOLO quedó levemente arriba (2 piezas malas escapadas vs. 4 de 453), pero elegimos **EfficientNet-B0** "
                "porque con el umbral ajustado se escapan solo 3, en la app solo necesita `torch` + `timm` "
                "y controlamos directamente el umbral y las probabilidades.")

# ── Pestaña 3: acerca del proyecto ─────────────────────────────────────────
with tab_acerca:
    st.markdown(f"""
### ¿Qué es FundiScan?
Una ayuda para el **inspector de calidad** al final de una línea de fundición de impulsores de bombas sumergibles.
Clasifica la foto de la pieza como **conforme** o **defectuosa** (poros, rechupes, rebabas, grietas) y genera
una recomendación con IA sobre qué hacer y qué revisar en el proceso.

### Cómo usarla
1. Elige un ejemplo, sube una foto frontal de la pieza o tómala con la cámara.
2. Revisa el veredicto, la confianza y las probabilidades.
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
