import cv2
import pytesseract
import re
import csv
import numpy as np
import os

# 1. CONFIGURACIÓN INICIAL
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\luisd\Documents\tesseract\tesseract.exe'

# Rutas de archivos
VIDEO_PATH = 'video_1.mp4'
OUTPUT_CSV = 'datos_fuerza_segmentados.csv'

# Coordenadas de la tabla (Asegúrate de que estas cubran toda la caja de datos)
X, Y, W, H = 600, 270, 350, 300

def limpiar_y_extraer(segmento_mask, roi_original):
    """Aplica la máscara y extrae los números X, Y, Z"""
    # Aplicar máscara al color original
    res = cv2.bitwise_and(roi_original, roi_original, mask=segmento_mask)
    gris = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    # Inversión para Tesseract (Texto negro sobre fondo blanco)
    _, inv = cv2.threshold(gris, 1, 255, cv2.THRESH_BINARY_INV)
    
    config = '--psm 6 -c tessedit_char_whitelist=-0123456789.'
    texto = pytesseract.image_to_string(inv, config=config)
    
    # Extraer todos los números encontrados
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", texto)
    # Rellenar con 0.0 si faltan datos
    while len(nums) < 3:
        nums.append("0.0")
    return [float(n) for n in nums[:3]]

# 2. PROCESAMIENTO
if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: No se encontró el archivo {VIDEO_PATH}. Verifica el nombre.")
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    datos_totales = []

    print("Analizando video... Presiona ESC para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extraer Región de Interés (ROI)
        roi = frame[Y:Y+H, X:X+W]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # MÁSCARAS DE COLOR
        # Verde
        mask_verde = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        # Rojo (dos rangos)
        mask_rojo = cv2.add(cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),
                            cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255])))
        # Azul
        mask_azul = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))

        # Extraer valores
        v_xyz = limpiar_y_extraer(mask_verde, roi)
        r_xyz = limpiar_y_extraer(mask_rojo, roi)
        a_xyz = limpiar_y_extraer(mask_azul, roi)

        # Guardar resultados
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        datos_totales.append([frame_id] + v_xyz + r_xyz + a_xyz)

        # Mostrar previsualización
        cv2.rectangle(frame, (X, Y), (X+W, Y+H), (255, 255, 0), 2)
        cv2.imshow("Analisis", frame)
        
        # Salida de emergencia con tecla ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # 3. GUARDAR CSV
    try:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Frame', 'Verde_X', 'Verde_Y', 'Verde_Z', 
                        'Rojo_X', 'Rojo_Y', 'Rojo_Z', 
                        'Azul_X', 'Azul_Y', 'Azul_Z'])
            w.writerows(datos_totales)
        print(f"PROCESO COMPLETADO. Datos guardados en: {OUTPUT_CSV}")
    except Exception as e:
        print(f"Error al escribir el CSV: {e}")