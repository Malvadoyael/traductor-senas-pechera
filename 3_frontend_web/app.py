import cv2
import mediapipe as mp
import pickle
import warnings
from flask import Flask, Response, jsonify

# Quitar advertencias molestas de la terminal
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# 1. CARGAR EL MODELO
try:
    with open('modelo_senas.pkl', 'rb') as f:
        modelo = pickle.load(f)
    print("✅ ¡Cerebro de SENSE AI cargado y listo!")
except FileNotFoundError:
    print("❌ ERROR: No encuentro el archivo 'modelo_senas.pkl'.")

# 2. CONFIGURAR MEDIAPIPE (Con filtro anti-fantasmas a 0.75)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

# ==========================================
# EL TRUCO MAGISTRAL: EL DICCIONARIO DE MACROS
# ==========================================
diccionario_palabras = {
    '1': 'Hola 👋 ',
    '2': 'Gracias 🙏 ',
    '3': 'Por favor ',
    '4': '¿Cómo estás? ',
    '5': 'Necesito ayuda 🆘 ',
    '6': 'Sí 👍 ',
    '7': 'No 👎 ',
    '8': 'Mucho gusto 🤝 '
}

# Variable global para guardar la última seña o palabra
ultima_sena = ""

def generate_frames():
    global ultima_sena
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Si hay una mano (y está 75% seguro de que es una mano real)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                coords = []
                for p in hand_landmarks.landmark:
                    coords.extend([p.x, p.y])
                
                # Le preguntamos a la IA qué es
                prediccion = modelo.predict([coords])
                letra_cruda = prediccion[0]
                
                # INTERCEPTOR DE PALABRAS
                if letra_cruda in diccionario_palabras:
                    ultima_sena = diccionario_palabras[letra_cruda]
                    texto_pantalla = f"PALABRA: {ultima_sena}"
                    color_texto = (0, 255, 255) # Color amarillo en pantalla para palabras
                else:
                    ultima_sena = letra_cruda
                    texto_pantalla = f"SENA: {ultima_sena}"
                    color_texto = (0, 255, 0) # Color verde en pantalla para letras
                
                # Dibujamos en el video
                cv2.putText(frame, texto_pantalla, (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_texto, 3)
        else:
            # Si no hay mano, vaciamos la memoria
            ultima_sena = ""

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    return jsonify({'letra': ultima_sena})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)