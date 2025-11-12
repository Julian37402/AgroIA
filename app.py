import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # oculta warnings de TensorFlow

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo
model = tf.keras.models.load_model("AgroIA_model_CNN.h5")

# Clases del modelo
clases = ["sigatoka", "cordonata", "pestalotiopsis", "healthy"]

# Función de predicción
def predecir(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    clase_pred = clases[np.argmax(pred)]
    prob = float(np.max(pred))
    return {clase_pred: prob}

# Interfaz profesional con fondo verde menta y layout moderno, compatible con cualquier versión
with gr.Blocks(css="""
    body {background-color: #d0f0c0;} /* verde menta claro */
    .gradio-container {padding: 2rem;}
    .gr-button {background-color: #4CAF50; color: white; font-weight: bold;}
    .gr-label {font-size: 1.2rem; font-weight: bold;}
    .gr-column {background-color: #ffffffaa; border-radius: 15px; padding: 1rem; margin:0.5rem;}
""") as demo:

    # Encabezado central
    gr.HTML("""
        <div style="text-align:center; margin-bottom:20px;">
            <h1 style="color:#2e7d32;">🌿 AgroIA</h1>
            <p style="font-size:1.2rem; color:#2e7d32;">
                Modelo CNN BananaLSD para detectar enfermedades en hojas de plátano 🍌
            </p>
        </div>
    """)

    # Layout en dos columnas: imagen a la izquierda, predicción a la derecha
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Sube una hoja de plátano")
            btn = gr.Button("Detectar enfermedad")
        with gr.Column():
            output_label = gr.Label(num_top_classes=4, label="Predicción")

    # Conectar botón
    btn.click(fn=predecir, inputs=input_image, outputs=output_label)

demo.launch()

