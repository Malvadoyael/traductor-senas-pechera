import cv2
import mediapipe as mp
import pickle
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for

app = Flask(__name__)

# Desactivar caché del navegador
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Cargar el modelo
with open('modelo_senas.pkl', 'rb') as f:
    modelo = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Variable global para guardar la última letra detectada
ultima_letra = ""

def generate_frames():
    global ultima_letra
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = []
                for p in hand_landmarks.landmark:
                    coords.extend([p.x, p.y])
                
                prediccion = modelo.predict([coords])
                ultima_letra = prediccion[0]
                
                cv2.putText(frame, f"SENA: {ultima_letra}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            ultima_letra = ""

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# --- RUTAS DE LA PÁGINA WEB (LOGIN Y DASHBOARD) ---

@app.route('/')
def index():
    # Carga la pantalla de inicio de sesión
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Si es GET, solo muestra el login
    if request.method == 'GET':
        return render_template('login.html')
    
    # Si es POST, valida los datos
    username = request.form.get('username')
    password = request.form.get('password')

    # Valida el acceso
    if username == "admin" and password == "1234":
        return redirect(url_for('dashboard'))
    else:
        # Si se equivoca, recarga el login mandando un error
        return render_template('login.html', error="Usuario o contraseña incorrectos")

@app.route('/dashboard')
def dashboard():
    # Carga la interfaz principal del traductor
    return render_template('dashboard.html')


# --- RUTAS DE LA CÁMARA Y DETECCIÓN ---

@app.route('/video_feed')
def video_feed():
    # Transmite el video en vivo
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    # Le manda a la web la letra detectada en formato JSON
    return jsonify({'letra': ultima_letra})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)