import cv2
import mediapipe as mp
import csv

# Inicializar MediaPipe para detectar la mano
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# El archivo Excel donde se guardarán las matemáticas de tus dedos
archivo_csv = 'dataset_senas.csv'

# Encender la cámara de tu lap
cap = cv2.VideoCapture(0)

print("=====================================================")
print(" RECOLECTOR DE DATOS - SENSE AI")
print("=====================================================")
print("1. Pon tu mano frente a la cámara (debe salir el esqueleto rojo).")
print("2. Presiona la letra de la seña en tu teclado (ej. 'A').")
print("3. Presiona 'ESC' para salir cuando termines.")
print("=====================================================")

while True:
    success, frame = cap.read()
    if not success:
        break
        
    # Efecto espejo para mayor comodidad
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Si detecta la mano, dibuja el esqueleto rojo
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar la ventana de video
    cv2.imshow("Captura de Datos - SENSE AI", frame)
    
    # Leer qué tecla presionaste
    key = cv2.waitKey(1) & 0xFF
    
    # Si presionas ESC (código 27), se cierra el programa
    if key == 27: 
        break
    # Si presionas una letra de la 'A' a la 'Z' en tu teclado
    elif 97 <= key <= 122: 
        letra_presionada = chr(key).upper()
        
        # Solo guarda si realmente está viendo el esqueleto de tu mano
        if results.multi_hand_landmarks:
            mano = results.multi_hand_landmarks[0] # Agarra la mano detectada
            coordenadas = [letra_presionada] # Inicia la fila con la letra ('A')
            
            # Extraer las posiciones X y Y de los 21 puntos
            for punto in mano.landmark:
                coordenadas.append(punto.x)
                coordenadas.append(punto.y)
            
            # Guardar todo en el archivo .csv
            with open(archivo_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(coordenadas)
                
            print(f"✅ ¡Guardado un ejemplo para la letra: {letra_presionada}!")
        else:
            print("⚠️ No veo el esqueleto. Abre bien la mano primero.")

cap.release()
cv2.destroyAllWindows()