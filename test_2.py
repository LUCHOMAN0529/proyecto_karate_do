import cv2
import pytesseract
import re
import csv
# Esta es la ruta basada en tu captura de pantalla
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\luisd\Documents\tesseract\tesseract.exe'

# 1. Configuración de coordenadas y constantes
X_COORD, Y_COORD, W_WIDTH, H_HEIGHT = 600, 270, 350, 300
VIDEO_PATH = 'video_1.mp4'
OUTPUT_CSV = 'datos_fuerza_segmentados.csv'

# Lista maestra para almacenar todas las filas de datos
lista_datos_final = []
X_COORD, Y_COORD, W_WIDTH, H_HEIGHT = 600, 270, 350, 300

def extraer_numeros(roi_segmentada):
    if roi_segmentada.size == 0:
        return "0.0"
    
    # CORRECCIÓN AQUÍ: cv2.COLOR_BGR2GRAY (con un '2')
    gris = cv2.cvtColor(roi_segmentada, cv2.COLOR_BGR2GRAY)
    
    # Mejoramos el contraste para el fondo de video_1.mp4
    _, umbral = cv2.threshold(gris, 150, 255, cv2.THRESH_BINARY_INV)
    
    config_ocr = '--psm 6 -c tessedit_char_whitelist=-0123456789.'
    texto = pytesseract.image_to_string(umbral, config=config_ocr)
    
    encontrados = re.findall(r"[-+]?\d*\.\d+|\d+", texto)
    return encontrados[0] if encontrados else "0.0"

# 2. Procesamiento del Video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

print("Procesando video y extrayendo datos... Presiona 'q' para finalizar prematuramente.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Segmentar la tabla principal
    tabla_roi = frame[Y_COORD:Y_COORD+H_HEIGHT, X_COORD:X_COORD+W_WIDTH]

    # Subcategorizar por Ejes (Dividiendo la ROI en 3 franjas horizontales)
    h_tercio = H_HEIGHT // 3
    # Sub-ROI para X, Y, Z
    dato_x = extraer_numeros(tabla_roi[0:h_tercio, :])
    dato_y = extraer_numeros(tabla_roi[h_tercio:h_tercio*2, :])
    dato_z = extraer_numeros(tabla_roi[h_tercio*2:H_HEIGHT, :])

    # Almacenar en la lista (Frame, Eje X, Eje Y, Eje Z)
    # Convertimos a float para asegurar que son datos numéricos puros
    fila = [frame_count, float(dato_x), float(dato_y), float(dato_z)]
    lista_datos_final.append(fila)

    # Mostrar progreso visual
    cv2.rectangle(frame, (X_COORD, Y_COORD), (X_COORD+W_WIDTH, Y_COORD+H_HEIGHT), (0, 255, 0), 2)
    cv2.imshow('Analisis en Tiempo Real', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 3. Creación del archivo CSV al finalizar
print(f"\nProcesamiento terminado. Total de frames analizados: {len(lista_datos_final)}")

try:
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Cabeceras del CSV
        writer.writerow(['Frame', 'Fuerza_X', 'Fuerza_Y', 'Fuerza_Z'])
        # Escribir toda la lista de datos
        writer.writerows(lista_datos_final)
    print(f"Éxito: Los datos se han guardado en '{OUTPUT_CSV}'")
except Exception as e:
    print(f"Error al guardar el CSV: {e}")