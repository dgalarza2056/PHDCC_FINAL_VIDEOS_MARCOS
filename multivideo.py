import cv2
import numpy as np
from region_selector import RegionSelector

# Cargar fondo
fondo_path = "recursos/fondo.png"
fondo = cv2.imread(fondo_path)

# Seleccionar múltiples regiones
selector = RegionSelector(fondo_path)
regiones = []
for i in range(2):  # Puedes cambiar el número de regiones
    print(f"Selecciona la región {i+1} y presiona ENTER")
    poligono = selector.run()
    if poligono is not None and len(poligono) >= 3:
        regiones.append(poligono)
    else:
        print("Región inválida.")
        exit()

# Inicializar fuentes de video
cam = cv2.VideoCapture(0)
video = cv2.VideoCapture("recursos/video.mp4")

# Crear máscaras para cada región
mascaras = []
mascaras_inv = []
bounding_boxes = []

for poly in regiones:
    mask = np.zeros_like(fondo[:, :, 0])
    cv2.fillPoly(mask, [poly], 255)
    mask_inv = cv2.bitwise_not(mask)
    x, y, w, h = cv2.boundingRect(poly)

    mascaras.append(mask[y:y+h, x:x+w])
    mascaras_inv.append(mask_inv[y:y+h, x:x+w])
    bounding_boxes.append((x, y, w, h))

while True:
    ret1, frame1 = cam.read()
    ret2, frame2 = video.read()

    if not ret1:
        print("No se pudo leer la cámara.")
        break

    if not ret2:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Repetir el video si se acaba
        ret2, frame2 = video.read()

    fondo_copia = fondo.copy()

    frames_fuente = [frame1, frame2]  # Cámara va a la región 1, video.mp4 a la región 2

    for i, (frame, mask, mask_inv, (x, y, w, h)) in enumerate(zip(frames_fuente, mascaras, mascaras_inv, bounding_boxes)):
        frame_resized = cv2.resize(frame, (w, h))
        roi = fondo_copia[y:y+h, x:x+w]

        fondo_roi = cv2.bitwise_and(roi, roi, mask=mask_inv)
        video_roi = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)

        combinado = cv2.add(fondo_roi, video_roi)
        fondo_copia[y:y+h, x:x+w] = combinado

    cv2.imshow("Video Multifuente", fondo_copia)

    if cv2.waitKey(1) == 27:  # ESC para salir
        break

cam.release()
video.release()
cv2.destroyAllWindows()
