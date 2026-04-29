import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("=====================================================")
print(" ENTRENAMIENTO DE IA - SENSE AI")
print("=====================================================")

archivo_csv = 'dataset_senas.csv'
archivo_modelo = 'modelo_senas.pkl'

try:
    # 1. Leer tu Excel
    print("📖 Leyendo tus ejemplos de la A y la B...")
    datos = pd.read_csv(archivo_csv, header=None)
    
    # Separar las letras (columna 0) de las coordenadas (el resto)
    X = datos.iloc[:, 1:].values 
    y = datos.iloc[:, 0].values  

    # 2. Separar datos para el "examen final" (80% para estudiar, 20% para examen)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. ENTRENAR EL CEREBRO
    print("🧠 Entrenando el algoritmo Random Forest...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # 4. Calificar a la IA
    predicciones = modelo.predict(X_test)
    precision = accuracy_score(y_test, predicciones)
    
    print(f"🎯 Precisión del modelo en el examen: {precision * 100:.2f}%")

    # 5. Exportar el cerebro para usarlo en la web o app
    print(f"💾 Guardando el cerebro matemático como '{archivo_modelo}'...")
    with open(archivo_modelo, 'wb') as f:
        pickle.dump(modelo, f)

    print("✅ ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")

except FileNotFoundError:
    print(f"❌ ERROR: No encuentro el archivo {archivo_csv}. ¿Seguro que estás en la ruta correcta?")
except Exception as e:
    print(f"❌ Ocurrió un error: {e}")