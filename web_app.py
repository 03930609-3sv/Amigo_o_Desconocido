import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
from datetime import datetime
import unicodedata
import time
import sys
import shutil  # Para comprimir las fotos en un archivo ZIP

# --- 1. CONFIGURACIÓN INICIAL ---
PALABRA_CLAVE_SECRETA = "ITSI2026"
FOLDER_BASE = "base_datos_estudiantes"

st.set_page_config(
    page_title="IA Control - Sistemas ITSI", 
    page_icon="🛡️", 
    layout="centered"
)

# Control de estado de la cámara (Evita errores de carga en el navegador)
if 'id_camara' not in st.session_state:
    st.session_state.id_camara = 0

def reiniciar_interfaz():
    """Incrementa el ID para refrescar el componente de cámara de Streamlit"""
    st.session_state.id_camara += 1
    st.rerun()

def limpiar_nombre(texto):
    """Normaliza el texto para crear nombres de carpeta válidos"""
    texto = texto.strip()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    return texto

if not os.path.exists(FOLDER_BASE):
    os.makedirs(FOLDER_BASE)

@st.cache_resource
def cargar_cerebro_ia():
    """Carga el modelo h5 de Teachable Machine con soporte para Keras antiguo"""
    try:
        # Intentar cargar usando tf_keras para máxima compatibilidad
        try:
            from tf_keras.models import load_model
        except ImportError:
            from tensorflow.keras.models import load_model
            
        if not os.path.exists("keras_model.h5"):
            return "Error: No se encontró 'keras_model.h5' en el repositorio.", None
            
        model = load_model("keras_model.h5", compile=False)
        
        # Detección automática de archivo de etiquetas
        etiquetas_path = "labels.txt" if os.path.exists("labels.txt") else "etiquetas.txt"
        
        if not os.path.exists(etiquetas_path):
            return "Error: Archivo de etiquetas no encontrado.", None

        with open(etiquetas_path, "r", encoding="utf-8") as f:
            labels = [line.strip()[2:] for line in f.readlines()]
        return model, labels
    except Exception as e:
        return f"Fallo al cargar modelo: {str(e)}", None

# --- 2. PANEL LATERAL (INSTRUCCIONES Y GESTIÓN) ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
    
    st.title("📝 Guía de Registro")
    st.info("""
    **Pasos para el usuario:**
    1. **Captura:** Haz clic en el botón de la cámara.
    2. **Validación:** Si no te reconoce, aparecerá el formulario abajo.
    3. **Registro:** Ingresa tu nombre y la clave secreta.
    4. **Guardado:** El sistema guardará la foto y se reiniciará solo.
    """)

    st.divider()
    
    # SISTEMA DE DESCARGA DE RESPALDO
    st.subheader("📦 Descargar Fotos")
    st.write("Descarga las nuevas capturas a tu PC antes de cerrar la página.")
    
    if st.button("Generar Archivo .ZIP"):
        if os.path.exists(FOLDER_BASE) and len(os.listdir(FOLDER_BASE)) > 0:
            shutil.make_archive("respaldo_estudiantes", 'zip', FOLDER_BASE)
            with open("respaldo_estudiantes.zip", "rb") as f:
                st.download_button(
                    label="⬇️ Descargar ZIP de fotos",
                    data=f,
                    file_name=f"capturas_ia_{datetime.now().strftime('%d_%m')}.zip",
                    mime="application/zip"
                )
        else:
            st.warning("No hay fotos registradas aún.")

    st.divider()
    st.caption(f"Monitor: Python {sys.version_info.major}.{sys.version_info.minor} ✅")
    if st.button("🔄 Reiniciar Cámara"):
        reiniciar_interfaz()

st.title("🛡️ Sistema de Control de Acceso IA")

# --- 3. PROCESO DE CARGA ---
resultado_carga = cargar_cerebro_ia()
error_msg, datos_ia = resultado_carga if isinstance(resultado_carga[0], str) else (None, resultado_carga)

if error_msg:
    st.error(f"⚠️ Error de Sistema: {error_msg}")
    st.stop()

model, class_names = datos_ia

# --- 4. CONTENEDORES DE UI ---
cont_camara = st.container()
cont_mensajes = st.container()
cont_formulario = st.container()

# --- 5. LÓGICA DE VISIÓN ARTIFICIAL ---
with cont_camara:
    foto = st.camera_input(
        "Posiciona tu rostro para la validación", 
        key=f"cam_{st.session_state.id_camara}"
    )

if foto:
    img = Image.open(foto)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Adaptar imagen al formato de Teachable Machine (224x224)
    img_ia = cv2.resize(img_cv, (224, 224))
    img_ia = np.asarray(img_ia, dtype=np.float32).reshape(1, 224, 224, 3)
    img_ia = (img_ia / 127.5) - 1
    
    prediccion = model.predict(img_ia, verbose=0)
    indice = np.argmax(prediccion)
    nombre_ia = class_names[indice]
    confianza = prediccion[0][indice]
    
    with cont_mensajes:
        if confianza > 0.65 and "Desconocido" not in nombre_ia:
            st.success(f"✅ BIENVENIDO(A): {nombre_ia.upper()} ({int(confianza*100)}%)")
            st.balloons()
            if st.button("Finalizar / Siguiente persona ➡️"):
                reiniciar_interfaz()
        else:
            st.warning(f"🚨 IDENTIDAD NO VERIFICADA ({int(confianza*100)}%)")
            
            with cont_formulario:
                st.markdown("---")
                st.subheader("🆕 Registro de Estudiante")
                st.write("Ingresa tus datos para ser agregado a la base de datos de entrenamiento.")
                
                nombre_in = st.text_input("Nombre Completo:")
                clave_in = st.text_input("Palabra Clave Secreta:", type="password")
                
                col1, col2 = st.columns(2)
                
                if col1.button("💾 Guardar Datos"):
                    if clave_in == PALABRA_CLAVE_SECRETA and nombre_in:
                        n_limpio = limpiar_nombre(nombre_in).replace(" ", "_")
                        ruta_carpeta = os.path.join(FOLDER_BASE, n_limpio)
                        if not os.path.exists(ruta_carpeta): os.makedirs(ruta_carpeta)
                        
                        # Guardar la foto capturada en el servidor
                        archivo_path = os.path.join(ruta_carpeta, f"cap_{int(time.time())}.jpg")
                        cv2.imwrite(archivo_path, img_cv)
                        
                        st.success(f"✨ ¡Listo! Imagen guardada. Reiniciando en 2 segundos...")
                        time.sleep(2)
                        reiniciar_interfaz()
                    else:
                        st.error("Clave incorrecta o falta el nombre.")
                
                if col2.button("🔄 Intentar de nuevo"):
                    reiniciar_interfaz()

st.divider()
st.caption("ITSI Sistemas Informáticos 2026 | Desarrollado por Eugenio Mejía")