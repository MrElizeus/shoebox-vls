ROI RAW frames
Resolución ROI: ancho=260 px, alto=46 px, canales=3
Formatos guardados: RAW uint8 BGR, PNG RGB
Archivos RAW: uint8 sin cabecera, orden fila a fila (row-major). PNG: RGB estándar.
Puedes reconstruir los RAW con numpy.fromfile(..., dtype=np.uint8).reshape(h, w, canales).
