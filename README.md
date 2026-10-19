
# ☀️ SolarCheck IA – Diagnóstico visual de paneles solares

App web que clasifica el estado de un panel solar a partir de una foto y genera una recomendación de mantenimiento con IA generativa.

**Flujo:** 📷 Imagen → 🧠 EfficientNet-B0 (clase + confianza) → 📊 Probabilidades → 💬 Gemini → ✅ Recomendación

🔗 **App publicada:** https://certamen1iagenerativa-nlutspz8a8jnmo2zepuy59.streamlit.app/

---

## 1. Problema y motivación

La acumulación de polvo, nieve o excremento de aves sobre los paneles reduce la energía generada, y los daños eléctricos o físicos implican riesgo y pérdida de módulos. Inspeccionar visualmente miles de paneles es lento y costoso. SolarCheck IA apoya al personal de operación y mantenimiento (O&M) de plantas fotovoltaicas priorizando qué paneles limpiar o revisar.

## 2. Datos

| Campo | Detalle |
|---|---|
| Dataset | *Solar Panel Images Clean and Faulty Images* |
| Autor | Afroz (Kaggle: `pythonafroz`) |
| Enlace | https://www.kaggle.com/datasets/pythonafroz/solar-panel-images |
| Licencia | No especificada ("Unknown") – uso exclusivamente académico |
| Clases (6) | Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered |
| Clase excluida | `New` (solo 16 imágenes, insuficiente para entrenar) |

**Imágenes por clase y división (estratificada 70 / 15 / 15, semilla 42):**

| Clase | Train | Val | Test | Total |
|---|---|---|---|---|
| Bird-drop | 134 | 28 | 29 | 191 |
| Clean | 135 | 29 | 29 | 193 |
| Dusty | 133 | 28 | 29 | 190 |
| Electrical-damage | 72 | 16 | 15 | 103 |
| Physical-Damage | 48 | 11 | 10 | 69 |
| Snow-Covered | 86 | 18 | 19 | 123 |
| **Total** | **608** | **130** | **131** | **869** |

**Limitaciones:** licencia y origen de las fotos no documentados; desbalance 2,8:1 (Clean vs Physical-Damage); pocas imágenes en total; fotos heterogéneas (ángulo, iluminación, resolución); el test de Physical-Damage tiene solo 10 imágenes, por lo que su recall es poco estable. El dataset no se sube al repositorio: se descarga desde el enlace.

## 3. Modelo y resultados

Se compararon dos enfoques sobre el **mismo conjunto de prueba (131 imágenes)**:

| Métrica (test) | CLIP zero-shot (línea base) | **EfficientNet-B0 fine-tuned** |
|---|---|---|
| Accuracy | 0,481 | **0,740** |
| Balanced accuracy | 0,514 | **0,768** |

- **EfficientNet-B0** (timm, preentrenado en ImageNet) ajustado 15 épocas con AdamW, *data augmentation* y pesos por clase para compensar el desbalance. Se guardó la época con mejor balanced accuracy en validación.
- **Error frecuente:** clasificar un panel con polvo como limpio puede retrasar su limpieza y reducir la generación. CLIP cometió este error en 14 de 29 casos; EfficientNet-B0 en 5 de 29.
- **Error de mayor impacto potencial:** no detectar daños eléctricos o físicos puede retrasar una inspección necesaria y generar riesgos de seguridad. Por ello, las predicciones deben complementarse con revisión de personal calificado.
- **Clase más difícil:** Dusty (recall 0,48), que se confunde con Clean y Physical-Damage.

![Comparación](modelo/comparacion.png)

## 4. App

Hecha con **Streamlit** y publicada en **Streamlit Community Cloud**. Permite cargar fotografías JPG, JPEG o PNG, utilizar imágenes de ejemplo y obtener la clase predicha, nivel de confianza y probabilidades de las seis categorías.

También muestra las métricas del modelo en el conjunto de prueba y permite generar recomendaciones preventivas mediante Google Gemini. Incluye advertencias de uso educativo y recomienda inspección humana cuando la confianza es inferior al 60%.

**IA generativa:** Google Gemini, integrado mediante su API. El modelo utilizado puede configurarse mediante la variable `GEMINI_MODEL` de Streamlit Secrets; si no se configura, se utiliza el identificador definido por defecto en `app.py`.

La aplicación envía únicamente resultados textuales de la clasificación (clase, confianza y probabilidades), sin enviar la fotografía original.

El prompt limita las respuestas al mantenimiento de paneles fotovoltaicos, solicita revisión humana cuando la confianza es baja y advierte sobre posibles daños eléctricos o físicos. La clave API se configura mediante `GEMINI_API_KEY` en los Secrets de Streamlit Community Cloud y nunca debe publicarse en GitHub.

## 5. Estructura

```text
├── app.py                  # aplicación Streamlit
├── requirements.txt        # librerías
├── README.md
├── notebooks/
│   └── entrenamiento.ipynb # datos, CLIP vs EfficientNet, métricas
├── modelo/
│   ├── modelo.pt           # pesos EfficientNet-B0
│   ├── clases.json         # clases y métricas de test
│   └── comparacion.png
└── ejemplos/               # imágenes de prueba
```

## 6. Cómo ejecutar localmente

```bash
pip install -r requirements.txt

# Configurar GEMINI_API_KEY como variable de entorno
# Windows CMD: set GEMINI_API_KEY=tu_clave
# Windows PowerShell: $env:GEMINI_API_KEY="tu_clave"
# Mac/Linux: export GEMINI_API_KEY=tu_clave

streamlit run app.py
```

## 7. Integrantes

- Benjamin Araya
- Guillermo Barañao
- Fernando Rodríguez

## 8. Declaración de uso de IA

**Claude (Anthropic)**: apoyo en la selección del dataset, generación y explicación del código base del notebook (división de datos, CLIP y fine-tuning de EfficientNet-B0), desarrollo inicial de app.py y redacción del README.

**ChatGPT (OpenAI)**: herramienta de gran ayuda en la adaptación y revisión de app.py para Streamlit, resolución de errores, análisis de métricas, revisión del repositorio GitHub mediante conectores, integración de Gemini y elaboración de la presentación y documentación del certamen.

**Google Gemini**: integrado en la aplicación para generar recomendaciones preventivas de mantenimiento según la clasificación de imágenes y el nivel de confianza del modelo.

## 9. Consideraciones éticas

Prototipo educativo. No reemplaza la inspección de un técnico calificado. Ante daño eléctrico no se debe manipular el panel. El modelo se entrenó con una sola fuente de imágenes y puede fallar con otros tipos de panel, ángulos o condiciones de luz.
