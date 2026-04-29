import cv2
import mediapipe as mp
import csv
import os

print("=====================================================")
print(" DEBUGGER DE FOTOS - SENSE AI")
print("=====================================================")

# Bajamos la exigencia (min_detection_confidence=0.3) por si hay poca luz
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

carpeta_principal = "fotos_entrenamiento"
archivo_csv = "dataset_senas.csv"

# 1. Revisar si la carpeta principal existe
if not os.path.exists(carpeta_principal):
    print(f"❌ ERROR FATAL: No encuentro la carpeta '{carpeta_principal}'")
    print("Asegúrate de ejecutar esto desde la raíz del proyecto.")
    exit()

with open(archivo_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    carpetas_encontradas = os.listdir(carpeta_principal)
    print(f"📂 Carpetas encontradas adentro: {carpetas_encontradas}")
    
    for nombre_carpeta in carpetas_encontradas:
        ruta_carpeta = os.path.join(carpeta_principal, nombre_carpeta)
        
        if os.path.isdir(ruta_carpeta):
            letra = nombre_carpeta.replace("Fotos", "")
            print(f"\n👉 Entrando a la carpeta de la letra: {letra}")
            
            fotos_procesadas = 0
            fotos_sin_mano = 0
            
            for nombre_foto in os.listdir(ruta_carpeta):
                # Asegurarse de que lea jpg, png o jpeg
                if nombre_foto.lower().endswith((".jpg", ".png", ".jpeg")):
                    ruta_foto = os.path.join(ruta_carpeta, nombre_foto)
                    img = cv2.imread(ruta_foto)
                    
                    if img is None:
                        print(f"⚠️ No pude leer la imagen (tal vez está dañada): {nombre_foto}")
                        continue
                        
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)
                    
                    if results.multi_hand_landmarks:
                        mano = results.multi_hand_landmarks[0]
                        fila = [letra]
                        for punto in mano.landmark:
                            fila.append(punto.x)
                            fila.append(punto.y)
                        writer.writerow(fila)
                        fotos_procesadas += 1
                    else:
                        fotos_sin_mano += 1
                        
            print(f"✅ ¡ÉXITO! Coordenadas extraídas de: {fotos_procesadas} fotos.")
            if fotos_sin_mano > 0:
                print(f"🙈 OJO: MediaPipe no logró ver ninguna mano en {fotos_sin_mano} fotos.")

print("\n=====================================================")
print(" ¡PROCESO TERMINADO! Revisa tu Excel.")