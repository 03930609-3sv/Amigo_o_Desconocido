import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
from datetime import datetime
import unicodedata
import time
import sys

# --- 1. CONFIGURACIÓN DEL SISTEMA ---
PALABRA_CLAVE_SECRETA = "ITSI2026"
FOLDER_BASE = "base_datos_estudiantes"

st.set_page_config(
    page_title="IA Control - Sistemas ITSI", 
    page_icon="🛡️", 
    layout="centered"
)

# Inicialización de estado para la cámara
if 'cam_key' not in st.session_state:
    st.session_state.cam_key = 0

def reiniciar_interfaz():
    """Limpia la captura actual y reinicia el flujo del programa"""
    st.session_state.cam_key += 1
    st.rerun()

def limpiar_nombre(texto):
    """Elimina acentos y caracteres especiales para crear carpetas seguras"""
    texto = texto.strip()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    return texto

if not os.path.exists(FOLDER_BASE):
    os.makedirs(FOLDER_BASE)

@st.cache_resource
def cargar_ia():
    """Carga el modelo de TensorFlow de forma segura"""
    try:
        # Intento de importación flexible
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            try:
                from tf_keras.models import load_model
            except ImportError:
                return "Error: No se detecta TensorFlow en el servidor. Revisa tu requirements.txt", None
            
        if not os.path.exists("keras_model.h5"):
            return "Error: No se encontró 'keras_model.h5'. Verifica que el archivo esté en la raíz de GitHub.", None
            
        model = load_model("keras_model.h5", compile=False)
        
        # Búsqueda de etiquetas
        etiquetas_path = "labels.txt"
        if os.path.exists("etiquetas.txt"):
            etiquetas_path = "etiquetas.txt"
        
        if not os.path.exists(etiquetas_path):
            return f"Error: No se encontró el archivo de etiquetas ({etiquetas_path}).", None
            
        with open(etiquetas_path, "r", encoding="utf-8") as f:
            labels = [line.strip()[2:] for line in f.readlines()]
        return model, labels
    except Exception as e:
        return f"Error de compatibilidad: {str(e)}", None

# --- 2. PANEL LATERAL DE DIAGNÓSTICO ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
    
    st.title("🛠️ Panel de Control")
    st.divider()
    
    # MONITOR DE SISTEMA
    st.subheader("🖥️ Monitor del Servidor")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    if sys.version_info.minor == 11:
        st.success(f"Python {py_ver} ✅ (Óptimo)")
    else:
        st.warning(f"Python {py_ver} ⚠️ (No ideal)")
        st.caption("Si la IA no carga, borra la app y créala de nuevo para forzar 3.11")
    
    st.divider()
    if st.button("🔄 Reiniciar Interfaz"):
        reiniciar_interfaz()

st.title("🛡️ Sistema de Control de Acceso IA")

# --- 3. INTENTO DE CARGA Y GESTIÓN DE ERRORES ---
resultado_carga = cargar_ia()
error_carga, datos_ia = resultado_carga if isinstance(resultado_carga[0], str) else (None, resultado_carga)

if error_carga:
    # Si la carga falla Y la versión es incorrecta, mostramos el mensaje de error crítico
    if sys.version_info.minor != 11:
        st.error(f"### ❌ Error de Sistema: Versión Incompatible ({py_ver})")
        st.write(f"""
        El servidor está usando una versión de Python que no soporta tu modelo de IA.
        
        **Detalle técnico:** `{error_carga}`
        
        **Cómo solucionarlo YA:**
        1. Ve a tu panel de **Streamlit Cloud**.
        2. Busca esta aplicación, haz clic en los tres puntos **(...)** y selecciona **Delete**.
        3. Haz clic en **Create App** de nuevo.
        4. Al crearla, el servidor leerá tu archivo `.python-version` y usará la versión **3.11** automáticamente.
        """)
    else:
        st.error(f"### ❌ Error al cargar archivos: {error_carga}")
    st.stop()

model, class_names = datos_ia

# --- 4. CONTENEDORES ESTÁTICOS ---
c_camara = st.container()
c_mensajes = st.container()
c_formulario = st.container()

# --- 5. LÓGICA DE IDENTIFICACIÓN ---
with c_camara:
    foto = st.camera_input(
        "Posiciona tu rostro para la validación", 
        key=f"input_{st.session_state.cam_key}"
    )

if foto:
    img = Image.open(foto)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Pre-procesamiento
    img_ia = cv2.resize(img_cv, (224, 224))
    img_ia = np.asarray(img_ia, dtype=np.float32).reshape(1, 224, 224, 3)
    img_ia = (img_ia / 127.5) - 1
    
    # Predicción
    pred = model.predict(img_ia, verbose=0)
    idx = np.argmax(pred)
    nombre = class_names[idx]
    conf = pred[0][idx]
    
    with c_mensajes:
        if conf > 0.65 and "Desconocido" not in nombre:
            st.success(f"✅ BIENVENIDO: {nombre.upper()} ({int(conf*100)}%)")
            st.balloons()
            if st.button("Finalizar y Siguiente ➡️"):
                reiniciar_interfaz()
        else:
            st.error(f"🚨 ROSTRO NO RECONOCIDO ({int(conf*100)}%)")
            
            with c_formulario:
                st.markdown("---")
                st.subheader("📝 Registro de Nuevo Estudiante")
                nombre_n = st.text_input("Nombre Completo:")
                clave_n = st.text_input("Clave Secreta:", type="password")
                
                col1, col2 = st.columns(2)
                if col1.button("💾 Guardar Datos"):
                    if clave_n == PALABRA_CLAVE_SECRETA and nombre_n:
                        nom_limpio = limpiar_nombre(nombre_n).replace(" ", "_")
                        ruta = os.path.join(FOLDER_BASE, nom_limpio)
                        if not os.path.exists(ruta): os.makedirs(ruta)
                        
                        cv2.imwrite(os.path.join(ruta, f"rec_{int(time.time())}.jpg"), img_cv)
                        st.toast(f"¡Imagen de {nombre_n} guardada!")
                        time.sleep(1.5)
                        reiniciar_interfaz()
                    else:
                        st.error("Clave incorrecta o datos incompletos.")
                
                if col2.button("🔄 Reintentar"):
                    reiniciar_interfaz()

st.divider()
st.caption("Sistemas 2026 | Desarrollado por Eugenio Mejía")