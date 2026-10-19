# 🚀 Guía paso a paso: del notebook a la app publicada

Tiempo total aproximado: **1 a 1,5 horas** la primera vez.

## Paso 1 · Entrenar en Colab (≈ 30 min)
1. Abrir https://colab.research.google.com → *Archivo → Subir notebook* → `notebooks/entrenamiento.ipynb`.
2. *Entorno de ejecución → Cambiar tipo de entorno de ejecución → **GPU T4***.
3. *Entorno de ejecución → **Ejecutar todas***. Si `kagglehub` pide credenciales, se crea una cuenta gratis en Kaggle y se sigue la instrucción que muestra la celda.
4. Al final se descarga **`fundiscan_export.zip`**. Descomprímanlo: trae las carpetas `modelo/` y `ejemplos/`.
   - Si el navegador bloquea la descarga automática: en el panel izquierdo de Colab (📁 *Archivos*) busquen `/content/fundiscan_export.zip` → clic derecho → *Descargar*. Háganlo **antes** de cerrar Colab, porque al desconectarse se borra.
5. **Guardar el notebook con los resultados visibles:** *Archivo → Descargar → Descargar .ipynb*. Esa versión reemplaza a `notebooks/entrenamiento.ipynb` del repo (la pauta exige el notebook **ejecutado**).
6. Copiar los números de la sección 7 y del final de la sección 9 en las tablas del `README.md` (donde dice `__ %`).

## Paso 2 · GitHub (≈ 15 min)
1. Cada integrante crea su cuenta en https://github.com.
2. Un integrante crea el repositorio: **New repository** → nombre `fundiscan` → **Public** → *Create*.
3. *Settings → Collaborators → Add people* para invitar a los otros dos.
4. **Add file → Upload files** y arrastrar **todo** el contenido de la carpeta del proyecto: `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `notebooks/`, `modelo/` (con el `.pt` y el `config.json` del zip), `ejemplos/` y `docs/`.
   - GitHub permite subir por la web archivos de hasta 25 MB. El modelo pesa ≈ 16 MB, así que no hay problema.
   - ⚠️ **Nunca** suban un archivo con la clave de Gemini.
5. En `README.md` y en `app.py` (pestaña "ℹ️ Acerca"), reemplazar `USUARIO` por el usuario real y `Nombre 1/2/3` por sus nombres.

## Paso 3 · Clave de Gemini (≈ 5 min)
1. Entrar a https://aistudio.google.com con una cuenta de Google.
2. **Get API key → Create API key**. Copiarla y guardarla en un lugar privado.

## Paso 4 · Hugging Face Spaces (≈ 20 min)
1. Crear una cuenta en https://huggingface.co.
2. **New → Space** → nombre `fundiscan` → SDK **Gradio** → hardware **CPU basic (free)** → **Public** → *Create Space*.
3. Pestaña **Files → Add file → Upload files**: subir los **mismos archivos** del repositorio, respetando las carpetas. El `README.md` ya trae el encabezado que el Space necesita (`sdk: gradio`, `app_file: app.py`).
4. **Settings → Variables and secrets → New secret**: nombre `GEMINI_API_KEY`, valor = la clave del paso 3.
5. Esperar a que el estado pase de *Building* a **Running** (3 a 8 minutos). Si aparece *Runtime error*, revisar la pestaña **Logs**.
6. Copiar el link `https://huggingface.co/spaces/USUARIO/fundiscan` y pegarlo en el README de GitHub.

## Paso 5 · Probar (≈ 10 min)
- [ ] Abrir el link **desde un celular** o desde otro computador.
- [ ] Clic en un ejemplo → se ve el veredicto, la confianza y las barras.
- [ ] Botón **Generar recomendación** → responde Gemini y abajo aparece "Generado por gemini-…".
  - Si dice "recomendación de respaldo", la clave no se cargó: revisar el nombre exacto del secret y reiniciar el Space (*Settings → Restart*).
- [ ] Probar el ejemplo **"dudosa"**: debería salir el aviso ámbar de revisión humana (si su modelo quedó muy seguro, la confianza puede ser alta igual; en ese caso, mostrar el aviso con una foto que no sea de una pieza).
- [ ] Pestaña 📊: aparecen las métricas reales del notebook.
- [ ] **Grabar un video corto (1–2 min) de la demo** como respaldo por si falla internet en la exposición.

## Paso 6 · Entregar en Canvas (un integrante)
1. Link al dataset + ficha (`docs/FICHA_DATOS.md`).
2. `notebooks/entrenamiento.ipynb` **ejecutado** + `app.py`.
3. Link del repositorio público de GitHub.
4. Link de la app en Hugging Face.
5. PDF de la presentación.

## Problemas frecuentes
| Síntoma | Solución |
|---|---|
| `FileNotFoundError: modelo/config.json` en el Space | Faltó subir la carpeta `modelo/`, o quedó dentro de otra carpeta |
| El Space se queda en *Building* mucho rato | Es normal la primera vez (instala PyTorch). Si pasan más de 15 min, *Settings → Factory rebuild* |
| `size mismatch` al cargar el modelo | El `.pt` y el `config.json` deben venir del mismo zip |
| Gemini responde con error de modelo | En *Variables and secrets*, crear la **variable** `GEMINI_MODEL` con un modelo vigente de Google AI Studio (ej. `gemini-3.5-flash-lite`; la lista está en https://ai.google.dev/gemini-api/docs/models) |
