import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
from datetime import datetime
import unicodedata
import time
import sys

# --- 1. CONFIGURACIÓN Y ESTADO ---
PALABRA_CLAVE_SECRETA = "ITSI2026"
FOLDER_BASE = "base_datos_estudiantes"

st.set_page_config(page_title="IA Control - Sistemas ITSI", page_icon="🛡️", layout="centered")

# Inicializamos el ID de la cámara si no existe
if 'id_camara' not in st.session_state:
    st.session_state.id_camara = 0

def reiniciar_interfaz():
    st.session_state.id_camara += 1
    st.rerun()

def limpiar_nombre(texto):
    texto = texto.strip()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    return texto

if not os.path.exists(FOLDER_BASE):
    os.makedirs(FOLDER_BASE)

@st.cache_resource
def cargar_modelo_ia():
    try:
        # Intento de importación flexible
        try:
            from tf_keras.models import load_model
        except ImportError:
            try:
                from tensorflow.keras.models import load_model
            except ImportError:
                return "Error: Librerías de IA no instaladas correctamente en el servidor.", None
            
        if not os.path.exists("keras_model.h5"):
            return "Error: Archivo 'keras_model.h5' no encontrado en GitHub.", None
            
        model = load_model("keras_model.h5", compile=False)
        
        # Búsqueda flexible de archivos de etiquetas
        archivo_etiquetas = None
        for f in ["labels.txt", "etiquetas.txt", "Labels.txt", "Etiquetas.txt"]:
            if os.path.exists(f):
                archivo_etiquetas = f
                break
        
        if not archivo_etiquetas:
            return "Error: Archivo de etiquetas (labels.txt/etiquetas.txt) no encontrado.", None
            
        with open(archivo_etiquetas, "r", encoding="utf-8") as f:
            labels = [line.strip()[2:] for line in f.readlines()]
        return model, labels
    except Exception as e:
        return f"Error inesperado: {str(e)}", None

# --- 2. INTERFAZ FIJA (SIDEBAR) ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
    else:
        st.warning("⚠️ logo.png no encontrado")
        
    st.title("🛠️ Administración")
    
    st.divider()
    st.subheader("🖥️ Monitor del Servidor")
    v_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Verificación de versión para el usuario
    if sys.version_info.minor == 11:
        st.success(f"Python: {v_python} ✅")
    else:
        st.error(f"Python: {v_python} ❌")
        st.write("TensorFlow requiere Python 3.11. Borra la app en Streamlit Cloud y créala de nuevo.")
    
    st.divider()
    if st.button("🔄 Limpiar Cámara"):
        reiniciar_interfaz()

st.title("🛡️ Sistema de Control de Acceso IA")

# --- 3. CONTENEDORES ESTÁTICOS ---
espacio_camara = st.container()
espacio_mensajes = st.container()
espacio_formulario = st.container()

# Intento de carga
resultado_carga = cargar_modelo_ia()

if isinstance(resultado_carga[0], str):
    st.error(resultado_carga[0])
    st.info("Revisa que los nombres de los archivos en tu GitHub coincidan exactamente y que no tengan espacios.")
    st.stop()

model, class_names = resultado_carga

# --- 4. LÓGICA DE LA CÁMARA ---
with espacio_camara:
    foto = st.camera_input("Captura tu rostro para validación", key=f"cam_{st.session_state.id_camara}")

if foto is not None:
    img = Image.open(foto)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    img_ia = cv2.resize(img_cv, (224, 224))
    img_ia = np.asarray(img_ia, dtype=np.float32).reshape(1, 224, 224, 3)
    img_ia = (img_ia / 127.5) - 1
    
    pred = model.predict(img_ia, verbose=0)
    idx = np.argmax(pred)
    nombre = class_names[idx]
    conf = pred[0][idx]
    
    with espacio_mensajes:
        if conf > 0.65 and "Desconocido" not in nombre:
            st.success(f"✅ BIENVENIDO: {nombre.upper()} ({int(conf*100)}%)")
            st.balloons()
            if st.button("Siguiente persona ➡️"):
                reiniciar_interfaz()
        else:
            st.error(f"🚨 ROSTRO NO RECONOCIDO ({int(conf*100)}%)")
            
            with espacio_formulario:
                st.markdown("### 📝 ¿Eres nuevo? Regístrate")
                n_nuevo = st.text_input("Nombre Completo:")
                c_secreta = st.text_input("Clave Secreta:", type="password")
                
                col1, col2 = st.columns(2)
                if col1.button("💾 Guardar Datos"):
                    if c_secreta == PALABRA_CLAVE_SECRETA and n_nuevo:
                        n_limpio = limpiar_nombre(n_nuevo).replace(" ", "_")
                        path = os.path.join(FOLDER_BASE, n_limpio)
                        if not os.path.exists(path): os.makedirs(path)
                        
                        cv2.imwrite(os.path.join(path, f"f_{int(time.time())}.jpg"), img_cv)
                        st.toast("¡Imagen guardada con éxito!")
                        time.sleep(1.5)
                        reiniciar_interfaz()
                    else:
                        st.error("Clave incorrecta o falta nombre.")
                
                if col2.button("🔄 Intentar de nuevo"):
                    reiniciar_interfaz()

st.divider()
st.caption("ITSI Sistemas 2026 | Eugenio Mejía")