# Carpeta del modelo

Aquí van los **2 archivos** que genera el notebook (`fundiscan_export.zip` → carpeta `modelo/`):

- `efficientnet_b0_fundicion.pt` → pesos del modelo entrenado (≈ 16 MB)
- `config.json` → clases, umbral de decisión, transformaciones y métricas en prueba

Sin estos archivos la app no arranca. El `.pt` y el `config.json` deben venir **del mismo entrenamiento**.
