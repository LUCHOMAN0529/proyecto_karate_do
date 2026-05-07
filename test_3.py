import cv2
import pytesseract
import re
import csv
import numpy as np

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\luisd\Documents\tesseract\tesseract.exe'

# 1. PARÁMETROS DE CONFIGURACIÓN
VIDEO_PATH = 'video_1.mp4'
OUTPUT_CSV = 'datos_fuerza_roja_limpios.csv'
X, Y, W, H = 600, 270, 350, 300  # Coordenadas de la tabla

def limpiar_texto_ocr(texto):
    """Extrae números y filtra errores comunes de lectura"""
    encontrados = re.findall(r"[-+]?\d*\.\d+|\d+", texto)
    # Convertir a float y filtrar valores absurdos (ej. > 5000 N si es humano)
    valores = []
    for val in encontrados:
        try:
            num = float(val)
            if abs(num) < 10000: # Filtro de seguridad para errores de OCR
                valores.append(num)
        except:
            continue
    
    # Asegurar que devolvemos 3 valores (X, Y, Z)
    while len(valores) < 3:
        valores.append(0.0)
    return valores[:3]

# 2. PROCESAMIENTO
cap = cv2.VideoCapture(VIDEO_PATH)
datos_rojos = []

print("Extrayendo datos de la LÍNEA ROJA (l gr)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Recortar ROI y convertir a HSV
    roi = frame[Y:Y+H, X:X+W]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Máscara específica para el ROJO (incluye ambos rangos del círculo HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    mask_roja = cv2.add(mask1, mask2)

    # Aplicar máscara y pre-procesar para OCR
    res = cv2.bitwise_and(roi, roi, mask=mask_roja)
    gris = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gris, 1, 255, cv2.THRESH_BINARY_INV)

    # OCR
    texto = pytesseract.image_to_string(inv, config='--psm 6')
    x, y, z = limpiar_texto_ocr(texto)

    # Guardar: Frame y los 3 ejes de la línea roja
    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    datos_rojos.append([frame_id, x, y, z])

    # Visualización (opcional)
    cv2.imshow("Solo Rojo", inv)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

# 3. GUARDAR RESULTADOS
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Rojo_X', 'Rojo_Y', 'Rojo_Z'])
    writer.writerows(datos_rojos)

print(f"Finalizado. Los datos de la línea roja están en: {OUTPUT_CSV}")