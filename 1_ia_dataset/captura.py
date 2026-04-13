import cv2
import os

# 1. Configurar la carpeta donde se guardarán las fotos
carpeta_destino = 'fotos_entrenamiento'
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# 2. Iniciar la cámara web (el 0 es la cámara por defecto)
cap = cv2.VideoCapture(0)

# Contador para el nombre de las imágenes
contador = 0

print("=========================================")
print("Iniciando cámara...")
print("Pon tu mano dentro del cuadro verde.")
print("Presiona la tecla 'S' para tomar una foto.")
print("Presiona la tecla 'Q' para salir.")
print("=========================================")

while True:
    # Leer el cuadro actual de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    # Voltear la imagen en modo espejo (más intuitivo)
    frame = cv2.flip(frame, 1)

    # Definir el tamaño y posición del cuadro de enfoque (ROI)
    # Ajusta estos valores si quieres el cuadro más grande o en otro lado
    x, y, w, h = 200, 100, 240, 240 
    
    # Dibujar el rectángulo verde en la pantalla original
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar la ventana
    cv2.imshow('Captura de Dataset - Pechera ESP32', frame)

    # Escuchar las teclas
    tecla = cv2.waitKey(1) & 0xFF

    # Si presiona 's', tomamos la foto
    if tecla == ord('s'):
        # 1. Recortar solo lo que está dentro del cuadro verde
        recorte = frame[y:y+h, x:x+w]
        
        # 2. Convertir a blanco y negro (escala de grises)
        escala_grises = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        
        # 3. Redimensionar a 96x96 pixeles (Obligatorio para el ESP32)
        imagen_final = cv2.resize(escala_grises, (96, 96))
        
        # Guardar la imagen
        nombre_archivo = f"{carpeta_destino}/sena_{contador}.jpg"
        cv2.imwrite(nombre_archivo, imagen_final)
        print(f"Foto {contador} guardada con éxito -> {nombre_archivo}")
        contador += 1

    # Si presiona 'q', salimos del programa
    elif tecla == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()