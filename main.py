import cv2
import numpy as np
from region_selector import RegionSelector

# Ruta a la imagen base
fondo_path = "recursos/fondo.png"

# Paso 1: Seleccionar región
selector = RegionSelector(fondo_path)
polygon = selector.run()

if polygon is None or len(polygon) < 3:
    print("No se seleccionó una región válida.")
    exit()

# Paso 2: Capturar video de la cámara
cap = cv2.VideoCapture(0)

# Leer imagen fondo original
fondo = cv2.imread(fondo_path)
altura, ancho = cv2.boundingRect(polygon)[3], cv2.boundingRect(polygon)[2]

mask = np.zeros_like(fondo[:, :, 0])
cv2.fillPoly(mask, [polygon], 255)
mask_inv = cv2.bitwise_not(mask)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar frame al tamaño de la región
    frame_resized = cv2.resize(frame, (ancho, altura))

    # Crear ROI de fondo
    fondo_copia = fondo.copy()

    # Crear una "ventana" del tamaño del bounding box del polígono
    x, y, w, h = cv2.boundingRect(polygon)

    # Área a reemplazar dentro del fondo
    roi = fondo_copia[y:y+h, x:x+w]

    # Fusionar video redimensionado con fondo
    mask_roi = mask[y:y+h, x:x+w]
    mask_inv_roi = mask_inv[y:y+h, x:x+w]

    fondo_roi = cv2.bitwise_and(roi, roi, mask=mask_inv_roi)
    video_roi = cv2.bitwise_and(frame_resized, frame_resized, mask=mask_roi)

    combinado = cv2.add(fondo_roi, video_roi)
    fondo_copia[y:y+h, x:x+w] = combinado

    cv2.imshow("Video Fusionado", fondo_copia)

    if cv2.waitKey(1) == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
