"""SolarCheck IA: clasificador visual de paneles fotovoltaicos en Streamlit.
Imagen -> EfficientNet-B0 -> probabilidades -> recomendación Gemini (solo texto).
"""
from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import timm
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
ARCHIVO_CLASES = ROOT / "modelo" / "clases.json"
ARCHIVO_PESOS = ROOT / "modelo" / "modelo.pt"
UMBRAL_CONFIANZA = 0.60

NOMBRES = {
    "Bird-drop": "Excremento de aves",
    "Clean": "Limpio",
    "Dusty": "Polvo acumulado",
    "Electrical-damage": "Daño eléctrico",
    "Physical-Damage": "Daño físico",
    "Snow-Covered": "Cubierto de nieve",
}

PROMPT_SISTEMA = """Eres un asistente de operación y mantenimiento de paneles solares.
Responde únicamente sobre el mantenimiento de instalaciones fotovoltaicas.
En español y en máximo 120 palabras explica: (1) qué significa el hallazgo,
(2) impacto probable, (3) dos o tres acciones preventivas priorizadas.
Si la confianza es menor del 60%, comienza pidiendo inspección humana.
Ante posibles daños físicos o eléctricos, advierte del riesgo y pide acudir
al personal calificado. No des instrucciones operativas peligrosas.
No inventes porcentajes de pérdida de producción ni asegures un diagnóstico.
Finaliza aclarando que es una orientación educativa, no un informe técnico.
"""

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


@st.cache_resource(show_spinner="Cargando modelo EfficientNet-B0...")
def cargar_modelo():
    if not ARCHIVO_CLASES.is_file():
        raise FileNotFoundError(f"Falta {ARCHIVO_CLASES.relative_to(ROOT)} en GitHub")
    if not ARCHIVO_PESOS.is_file():
        raise FileNotFoundError(f"Falta {ARCHIVO_PESOS.relative_to(ROOT)} en GitHub")
    with ARCHIVO_CLASES.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    clases = meta["clases"]
    modelo = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(clases))
    pesos = torch.load(ARCHIVO_PESOS, map_location="cpu", weights_only=True)
    modelo.load_state_dict(pesos)
    modelo.eval()
    return modelo, meta


def clasificar(img: Image.Image, modelo, clases):
    tensor = TF(img.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilidades = modelo(tensor).softmax(dim=1)[0].cpu().tolist()
    pares = list(zip(clases, probabilidades))
    mejor_clase, mejor_confianza = max(pares, key=lambda item: item[1])
    resultado = {
        "clase": mejor_clase,
        "nombre": NOMBRES.get(mejor_clase, mejor_clase),
        "confianza": float(mejor_confianza),
        "probabilidades": {
            NOMBRES.get(clase, clase): float(probabilidad)
            for clase, probabilidad in pares
        },
    }
    return resultado


def secreto(nombre: str):
    valor = os.environ.get(nombre)
    if valor:
        return valor
    try:
        return st.secrets.get(nombre)
    except Exception:
        return None


def recomendar(resultado: dict):
    clave = secreto("GEMINI_API_KEY")
    if not clave:
        raise ValueError("Falta GEMINI_API_KEY en Streamlit → App settings → Secrets.")
    from google import genai
    cliente = genai.Client(api_key=clave)
    probabilidades = ", ".join(
        f"{clase}: {p:.1%}" for clase, p in
        sorted(resultado["probabilidades"].items(), key=lambda item: item[1], reverse=True)
    )
    contenido = (
        f"{PROMPT_SISTEMA}\n\n"
        f"Condición detectada: {resultado['nombre']}\n"
        f"Confianza: {resultado['confianza']:.1%}\n"
        f"Probabilidades: {probabilidades}\n"
        "Escribe la recomendación preventiva."
    )
    respuesta = cliente.models.generate_content(
        model=secreto("GEMINI_MODEL") or "gemini-3.8-flash",
        contents=contenido,
    )
    if not respuesta.text:
        raise RuntimeError("Gemini no devolvió texto. Intenta de nuevo.")
    return respuesta.text


st.set_page_config(page_title="SolarCheck IA", page_icon="☀️", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0b1220; }
section[data-testid="stSidebar"] { background: #102033; }
h1, h2, h3 { letter-spacing: -0.035em; }
.banner {background: linear-gradient(105deg,#0e4546,#136c5c);padding:30px 32px;
border-radius:18px;margin-bottom:24px;color:white;}
.banner h1 {color:white;font-size:2.7rem;margin:0 0 8px;}
.banner p {color:#e0fff6;margin:0;line-height:1.7;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="banner"><div style="letter-spacing:.15em;font-size:.8rem;
font-weight:700;color:#acfce0">ENERGÍA SOLAR · VISIÓN ARTIFICIAL · IA GENERATIVA</div>
<h1>☀️ SolarCheck IA</h1><p>Diagnóstico visual educativo de paneles solares.
Clasifica imágenes con EfficientNet-B0 y genera recomendaciones preventivas con Gemini.</p>
</div>""", unsafe_allow_html=True)

try:
    modelo, META = cargar_modelo()
    CLASES = META["clases"]
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.info("En GitHub deben existir `modelo/modelo.pt` y `modelo/clases.json`.")
    st.stop()

with st.sidebar:
    st.header("SolarCheck IA")
    st.write("**Modelo:** EfficientNet-B0 ajustado para 6 clases")
    st.write("**Umbral orientativo:** 60% de confianza")
    st.warning("Prototipo académico. No reemplaza una revisión profesional.")

pestana_analisis, pestana_metricas, pestana_acerca = st.tabs(
    ["🔎 Diagnóstico", "📊 Desempeño del modelo", "ℹ️ Acerca de"]
)

with pestana_analisis:
    st.subheader("1. Carga una fotografía")
    col_imagen, col_resultado = st.columns([1, 1], gap="large")
    with col_imagen:
        archivo = st.file_uploader("Fotografía de panel solar", type=["jpg", "jpeg", "png"])
        ejemplos = sorted(
            p for p in (ROOT / "ejemplos").glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ) if (ROOT / "ejemplos").is_dir() else []
        opcion = "Ninguna"
        if ejemplos:
            opcion = st.selectbox("O utiliza una imagen de ejemplo", ["Ninguna"] + [p.name for p in ejemplos])
        img = None
        huella = None
        try:
            if archivo is not None:
                contenido_img = archivo.getvalue()
                img = Image.open(BytesIO(contenido_img)).convert("RGB")
                huella = hashlib.sha256(contenido_img).hexdigest()
            elif opcion != "Ninguna":
                ruta = next(p for p in ejemplos if p.name == opcion)
                contenido_img = ruta.read_bytes()
                img = Image.open(BytesIO(contenido_img)).convert("RGB")
                huella = hashlib.sha256(contenido_img).hexdigest()
        except Exception as e:
            st.error(f"No se pudo abrir la imagen: {e}")
        if img is not None:
            st.image(img, caption="Imagen enviada", use_container_width=True)
        analizar = st.button("🔎 Analizar panel", type="primary", disabled=(img is None))
        if analizar and img is not None:
            with st.spinner("Analizando fotografía..."):
                try:
                    st.session_state["resultado_solarcheck"] = clasificar(img, modelo, CLASES)
                    st.session_state["huella_solarcheck"] = huella
                    st.session_state.pop("recomendacion_solarcheck", None)
                except Exception as e:
                    st.error(f"No se pudo clasificar la imagen: {e}")
    with col_resultado:
        st.subheader("2. Resultado de la clasificación")
        resultado = st.session_state.get("resultado_solarcheck")
        valido = (resultado is not None and huella == st.session_state.get("huella_solarcheck"))
        if valido:
            st.success(f"**Condición detectada:** {resultado['nombre']}")
            st.metric("Confianza de esta predicción", f"{resultado['confianza']:.1%}")
            if resultado["confianza"] < UMBRAL_CONFIANZA:
                st.warning("Confianza baja: se recomienda inspección humana antes de actuar.")
            df = pd.DataFrame(
                [{"Clase": clase, "Probabilidad (%)": prob * 100}
                 for clase, prob in resultado["probabilidades"].items()]
            ).sort_values("Probabilidad (%)", ascending=False)
            st.bar_chart(df, x="Clase", y="Probabilidad (%)", horizontal=True)
        else:
            st.info("Selecciona una imagen y presiona «Analizar panel».")

    st.divider()
    st.subheader("3. Recomendación generada por IA")
    st.caption("Gemini recibe solo resultados textuales del clasificador; no recibe la fotografía.")
    habilitado = bool(
        st.session_state.get("resultado_solarcheck")
        and huella == st.session_state.get("huella_solarcheck")
    )
    if st.button("✨ Generar recomendación con Gemini", disabled=not habilitado):
        with st.spinner("Consultando Gemini..."):
            try:
                st.session_state["recomendacion_solarcheck"] = recomendar(
                    st.session_state["resultado_solarcheck"]
                )
            except Exception as e:
                mensaje = str(e)
                if "503" in mensaje or "UNAVAILABLE" in mensaje:
                    st.warning("Gemini está temporalmente saturado (503). Espera y vuelve a intentar.")
                else:
                    st.error(f"No se pudo generar la recomendación: {mensaje}")
    if habilitado and st.session_state.get("recomendacion_solarcheck"):
        st.markdown(st.session_state["recomendacion_solarcheck"])

with pestana_metricas:
    st.subheader("Evaluación del modelo")
    st.write("Estas son métricas globales del conjunto de prueba; no son la confianza de una fotografía.")
    c1, c2 = st.columns(2)
    try:
        c1.metric("Accuracy en prueba", f"{float(META['test_accuracy']):.1%}")
        c2.metric("Balanced accuracy en prueba", f"{float(META['test_balanced_accuracy']):.1%}")
    except (KeyError, TypeError, ValueError):
        st.info("El archivo de clases no registra ambas métricas.")
    st.caption("La exportación puede corresponder a una ejecución distinta de la registrada en gráficos anteriores.")
    for ruta in [ROOT / "modelo" / "comparacion.png", ROOT / "resultados" / "comparacion_recall.png"]:
        if ruta.is_file():
            st.image(str(ruta), caption="Comparación de modelos (verificar ejecución de origen)")
            break
    st.markdown("**Limitación:** los datos provienen de una única fuente; las clases están desbalanceadas y las probabilidades no garantizan un diagnóstico.")

with pestana_acerca:
    st.subheader("Acerca del proyecto")
    st.markdown("""
**Datos:** [Solar Panel Images – Clean and Faulty Images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images) (Kaggle; autor: Afroz).

**Modelo:** EfficientNet-B0, red preentrenada en ImageNet y adaptada a seis categorías.

**Uso de IA:** Gemini recibe la etiqueta y probabilidades en texto para ofrecer recomendaciones educativas; no recibe fotografías.

**Seguridad:** ante daño eléctrico o físico, no manipular los paneles y solicitar inspección de personal técnico calificado.

**Aviso:** prototipo académico; los resultados no son un diagnóstico técnico certificado.
""")
