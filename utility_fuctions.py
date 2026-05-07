def limpiar_texto_ocr(texto):
    import cv2
    import pytesseract
    import re
    import csv
    import numpy as np
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