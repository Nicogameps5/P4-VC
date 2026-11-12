# Práctica 4 VC

Ejercicios de la cuarta práctica de laboratorio de VC.

Nicolás Hernández Castro y Gabriel Godoy Navarro - EII ULPGC 13 de noviembre de 2026


## Entrenamiento

El conjunto de imagenes de vehículos que se utilizó en esta práctica se puede ver en el siguiente enlace: https://drive.google.com/drive/folders/1si7ISib-EFdn_7kkbebNzmaCSSybdQJM?usp=drive_link

Las imágenes se distribuyen de la siguiente manera:
* Entrenamiento: 151 imagenes (72%)
* Validación: 28 imagenes (13%)
* Test: 31 imágenes (15%)

Dado que los pesos preentrenados del conjunto COCO no contemplan la categoría de matrícula de vehículos, se procedió al entrenamiento supervisado de un detector especializado utilizando **YOLO** (en concreto la versión ligera `yolo11n.pt`). El proceso se llevó a cabo en Google Colab con GPU NVIDIA Tesla T4 (16 GB VRAM), empleando el dataset mostrado anteriormente con la siguiente configuración:

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='/content/matriculas.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)
```

![alt text](results_yolo.png)


Al observar las gráficas de resultado a lo largo de las 50 épocas de entrenamiento:

* Evolución de los errores (pérdidas): Los fallos al situar las cajas (box_loss y dfl_loss) van bajando de forma suave y constante, pasando de valores iniciales cercanos a 1.8 hasta estabilizarse en torno a 0.8 tanto en entrenamiento como en validación. Por su parte, el error al reconocer la clase (cls_loss) cae en picado a partir de la época 8 por debajo de 0.4, lo que demuestra que la red aprende rápido y sin problemas de memorización o sobreajuste (overfitting).

* Precisión y aciertos: La precisión y la capacidad para detectar todas las placas sin dejarse ninguna (recall) suben con mucha fuerza entre las épocas 10 y 18, consolidándose en valores muy altos, superiores al 90%.

* Rendimiento global ($mAP$): El porcentaje de acierto general ($mAP_{50}$) sube enseguida hasta situarse en un valor superior al  90%. A su vez, la métrica más exigente ($mAP_{50-95}$), que mide si la caja está milimétricamente ajustada a los bordes de la matrícula, progresa poco a poco hasta casi el 80%, confirmando que el modelo encuadra las placas con gran exactitud.

### Apartado Extra: Evaluación comparativa YOLO vs. RT-DETR
Como ampliación, se evalúa una alternativa moderna basada en Transformers (RT-DTER). 

Para llevar a cabo el experimento en igualdad de condiciones, se entrenó rtdetr-l.pt sobre el mismo conjunto de datos durante 50 épocas a resolución 640. Debido a la alta demanda de memoria de las matrices de atención del Transformer, fue necesario ajustar el tamaño de lote a batch=4 (frente a batch=16 de YOLO) para no desbordar los 16 GB de VRAM de la GPU T4:

```python
from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt')
results = model.train(
    data='/content/matriculas.yaml',
    epochs=50,
    imgsz=640,
    batch=4,
    device=0,
    name='rtdetr_matriculas'
)
```

![alt text](results_rtdetr.png)

Al comparar las gráficas de resultado de RT-DETR frente a las de YOLO:
* Evolución de pérdidas y convergencia: Al igual que en YOLO, las pérdidas de RT-DETR (giou_loss, cls_loss y l1_loss) caen de forma constante sin indicios de sobreajuste. Presenta algo más de oscilación en las primeras épocas de validación debido al ajuste de las capas de atención del Transformer, pero a partir de la época 20 se estabiliza con valores mínimos.

* Precisión y detección: Tanto la precisión como el recall crecen de forma rápida y pareja a YOLO, consolidándose ambas métricas en valores muy altos, sobre el 95% en precisión y superiores al 90% en capacidad de detección.

* Rendimiento de localización ($mAP$): En la detección estándar ($mAP_{50}$), ambos modelos muestran un comportamiento casi idéntico, saturando con rapidez en valores superiores al 95%. La diferencia principal aparece en la métrica estricta ($mAP_{50-95}$): mientras que YOLO se queda rondando el 80%, RT-DETR supera claramente el 80%, logrando delimitar las esquinas y proporciones de la matrícula con una fidelidad geométrica ligeramente superior. 

Además, destacar que el entrenamiento con YOLO fue notablemente más rápido (0.072 horas) que con RT-DETR (0.274 horas).

## Detección, seguimiento y lectura de matrículas en vídeo

En primer lugar, se crea dinámicamente el archivo `bytetrack_rapido.yaml`. En lugar de los 30 fotogramas habituales, se ajusta el búfer a `track_buffer: 8` para que, cuando un vehículo salga del encuadre, su caja delimitadora se destruya en apenas 0.25 segundos y no quede flotando en los bordes.   

Además, se define el selector booleano `USAR_RTDETR` para conmutar fácilmente entre el modelo convolucional `best.pt` y el Transformer `best_rtdetr.pt`, asignando nombres dinámicos a las salidas para que no se sobreescriban entre ejecuciones:

```python
USAR_RTDETR = False

TRACKER_CFG = 'bytetrack_rapido.yaml'
with open(TRACKER_CFG, 'w') as f:
    f.write("""
tracker_type: bytetrack
track_high_thresh: 0.35
track_low_thresh: 0.1
new_track_thresh: 0.4
track_buffer: 8
match_thresh: 0.8
fuse_score: True
""".strip())

VIDEO_ENTRADA = 'C0142.MP4'

SUFIJO_MODELO = '_rtdetr' if USAR_RTDETR else '_yolo'
VIDEO_SALIDA = f'resultado_C0142{SUFIJO_MODELO}.mp4'
CSV_SALIDA = f'detecciones_seguimiento{SUFIJO_MODELO}.csv'

MODELO_GENERAL_PATH = 'yolo11n.pt'
MODELO_MATRICULAS_PATH = 'best_rtdetr.pt' if USAR_RTDETR else 'best.pt'

CLASES_INTERES = [0, 2, 3, 5, 7]
CLASES_VEHICULOS = [2, 3, 5, 7]

DISPOSITIVO = 0 if torch.cuda.is_available() else 'cpu'

modelo_general = YOLO(MODELO_GENERAL_PATH)

if USAR_RTDETR:
    modelo_matriculas = RTDETR(MODELO_MATRICULAS_PATH)
else:
    modelo_matriculas = YOLO(MODELO_MATRICULAS_PATH)

reader_ocr = easyocr.Reader(['es', 'en'], gpu=torch.cuda.is_available())
```

Luego, se extraen las propiedades del vídeo de entrada (dimensiones y FPS) con `cv2.VideoCapture` y se configura el objeto `cv2.VideoWriter` con el códec `mp4v`. A la par, se inicializa el archivo CSV escribiendo la cabecera con las 15 columnas exigidas para almacenar fotograma, clases, coordenadas y texto:

```python
cap = cv2.VideoCapture(VIDEO_ENTRADA)
if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video: {VIDEO_ENTRADA}")

ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fps_salida = fps if fps > 0 else 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    VIDEO_SALIDA,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps_salida,
    (ancho, alto)
)

f_csv = open(CSV_SALIDA, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(f_csv)

csv_writer.writerow([
    'fotograma', 'tipo_objeto', 'confianza', 'identificador_tracking',
    'x1', 'y1', 'x2', 'y2',
    'matricula_en_su_caso', 'confianza_matricula',
    'mx1', 'my1', 'mx2', 'my2', 'texto_matricula'
])
```

Para evitar contar vehículos duplicados por parpadeos o vehículos estacionados, se fijan umbrales de tiempo y movimiento. Se definen diccionarios y conjuntos para registrar el historial de desplazamiento, la votación de categorías y una memoria temporal para no reejecutar el detector de matrículas fotograma a fotograma:

```python
MIN_FRAMES_VALIDO = 10
MIN_DESPLAZAMIENTO = 40

historial_tracks = defaultdict(lambda: {
    'inicio': None,
    'actual': None,
    'frames': 0,
    'clases': defaultdict(int)
})

conteo_unicos = {
    'person': set(),
    'car': set(),
    'motorcycle': set(),
    'bus': set(),
    'truck': set()
}

cache_matriculas = {}

COLORES = {
    'person': (255, 105, 65),
    'car': (40, 200, 40),
    'motorcycle': (0, 165, 255),
    'bus': (200, 200, 0),
    'truck': (180, 0, 180),
    'matricula': (0, 255, 255)
}
````

El vídeo se recorre fotograma a fotograma bajo `torch.no_grad()` para ahorrar memoria y acelerar la inferencia. En cada cuadro se ejecuta `modelo_general.track(..., persist=True)` para detectar objetos y mantener su identificador temporal, extrayendo sus coordenadas y descartando cajas menores a $25 \times 25$ píxeles situadas al fondo para evitar falsos positivos:

```python
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        resultados_track = modelo_general.track(
            source=frame,
            persist=True,
            classes=CLASES_INTERES,
            tracker=TRACKER_CFG,
            conf=0.35,
            imgsz=640,
            device=DISPOSITIVO,
            verbose=False
        )

        if resultados_track and len(resultados_track) > 0 and resultados_track[0].boxes is not None:
            cajas = resultados_track[0].boxes

            for box in cajas:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confianza = float(box.conf[0])
                clase_id = int(box.cls[0])
                nombre_clase = modelo_general.names[clase_id]
                track_id = int(box.id[0]) if box.id is not None else -1

                w_box = x2 - x1
                h_box = y2 - y1
                if w_box < 25 or h_box < 25:
                    continue

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
```

Para cada identificador activo se acumulan los fotogramas en los que aparece y se elige su categoría definitiva por votación mayoritaria `max`, evitando cambios bruscos de etiqueta entre cuadros. Cuando un objeto ha sido visto durante 10 fotogramas y se ha desplazado `math.hypot` al menos 40 píxeles desde su origen (20 en personas), se registra en su conjunto correspondiente de `conteo_unicos` tras descartarlo de las demás clases:

```python
                clase_dominante = nombre_clase
                if track_id != -1:
                    t = historial_tracks[track_id]
                    t['frames'] += 1
                    t['clases'][nombre_clase] += 1
                    if t['inicio'] is None:
                        t['inicio'] = (cx, cy)
                    t['actual'] = (cx, cy)

                    clase_dominante = max(t['clases'].items(), key=lambda x: x[1])[0]

                    if t['frames'] >= MIN_FRAMES_VALIDO:
                        disp = math.hypot(t['actual'][0] - t['inicio'][0], t['actual'][1] - t['inicio'][1])
                        es_vehiculo = clase_id in CLASES_VEHICULOS
                       
                        if (not es_vehiculo and disp >= 20) or (es_vehiculo and disp >= MIN_DESPLAZAMIENTO):
                            for c in conteo_unicos:
                                conteo_unicos[c].discard(track_id)
                            if clase_dominante in conteo_unicos:
                                conteo_unicos[clase_dominante].add(track_id)
```

Para optimizar el rendimiento, la búsqueda de matrículas se limita a vehículos de tamaño suficiente (mayor a 50 x 50 píxeles) que no estén tocando los bordes laterales `not esta_saliendo`. Si la matrícula ya se ubicó en los últimos 4 fotogramas, se proyecta su nueva posición utilizando las coordenadas relativas guardadas en caché `rel_coords`. Cuando toca volver a detectarla, se recorta la región del vehículo `roi_vehiculo`, se trasladan las coordenadas de la matrícula al fotograma completo, se amplía el recorte y se extraen los caracteres alfanuméricos con `EasyOCR`:

```python
caja_matricula = None
                conf_matricula = None
                texto_matricula = ""

                if clase_id in CLASES_VEHICULOS:
                    vx1, vy1 = max(0, x1), max(0, y1)
                    vx2, vy2 = min(ancho, x2), min(alto, y2)
                    w_veh = vx2 - vx1
                    h_veh = vy2 - vy1

                    # Ignorar vehículos que ya estén cortados saliendo de la pantalla
                    esta_saliendo = (vx1 <= 15) or (vx2 >= ancho - 15)

                    if w_veh > 50 and h_veh > 50 and not esta_saliendo:
                        usar_detector = True

                        if track_id != -1 and track_id in cache_matriculas:
                            frames_pasados = frame_idx - cache_matriculas[track_id]['frame']
                            if frames_pasados <= 4:
                                rx1, ry1, rx2, ry2 = cache_matriculas[track_id]['rel_coords']
                                gx1 = int(vx1 + rx1 * w_veh)
                                gy1 = int(vy1 + ry1 * h_veh)
                                gx2 = int(vx1 + rx2 * w_veh)
                                gy2 = int(vy1 + ry2 * h_veh)
                                caja_matricula = (gx1, gy1, gx2, gy2)
                               
                                conf_matricula = cache_matriculas[track_id].get('conf', 0.0)
                                texto_matricula = cache_matriculas[track_id].get('texto', "")
                                usar_detector = False

                        if usar_detector:
                            roi_vehiculo = frame[vy1:vy2, vx1:vx2]
                            res_placas = modelo_matriculas(
                                roi_vehiculo,
                                imgsz=320,
                                conf=0.35,
                                device=DISPOSITIVO,
                                verbose=False
                            )

                            if res_placas and len(res_placas) > 0 and len(res_placas[0].boxes) > 0:
                                mejor_placa = res_placas[0].boxes[0]
                                px1_r, py1_r, px2_r, py2_r = map(int, mejor_placa.xyxy[0].tolist())
                               
                                gx1 = vx1 + px1_r
                                gy1 = vy1 + py1_r
                                gx2 = vx1 + px2_r
                                gy2 = vy1 + py2_r
                                caja_matricula = (gx1, gy1, gx2, gy2)
                                conf_matricula = float(mejor_placa.conf[0])

                                img_placa = frame[gy1:gy2, gx1:gx2]
                                if img_placa.shape[0] > 10 and img_placa.shape[1] > 20:
                                   
                                    img_ampliada = cv2.resize(img_placa, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                                    img_gris = cv2.cvtColor(img_ampliada, cv2.COLOR_BGR2GRAY)

                                    resultados_ocr = reader_ocr.readtext(img_gris, detail=0, paragraph=True)
                                   
                                    if resultados_ocr:
                                        texto_crudo = "".join(resultados_ocr).upper()
                                        texto_limpio = "".join(filter(str.isalnum, texto_crudo))
                                       
                                        if len(texto_limpio) >= 4:
                                            texto_matricula = texto_limpio

                                if track_id != -1:
                                    cache_matriculas[track_id] = {
                                        'rel_coords': (px1_r / w_veh, py1_r / h_veh, px2_r / w_veh, py2_r / h_veh),
                                        'frame': frame_idx,
                                        'conf': conf_matricula,
                                        'texto': texto_matricula
                                    }
                    else:
                        if track_id != -1:
                            cache_matriculas.pop(track_id, None)
```

Al finalizar cada fotograma, se registran las coordenadas y datos en el CSV. Luego se dibujan las cajas con su color correspondiente para cada clase, el recuadro amarillo para la matrícula y un panel HUD en la esquina superior izquierda con el contador acumulado de cada tipo de objetos:

```python

                tiene_mat = "Si" if caja_matricula is not None else "No"
                conf_mat_str = f"{conf_matricula:.2f}" if caja_matricula is not None and conf_matricula is not None else ""
                mx1, my1, mx2, my2 = caja_matricula if caja_matricula is not None else ("", "", "", "")

                csv_writer.writerow([
                    frame_idx, clase_dominante, f"{confianza:.2f}", track_id,
                    x1, y1, x2, y2, tiene_mat, conf_mat_str,
                    mx1, my1, mx2, my2, texto_matricula
                ])

                color_obj = COLORES.get(clase_dominante, (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_obj, 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id} {clase_dominante}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_obj, 2
                )

                if caja_matricula is not None:
                    cv2.rectangle(frame, (mx1, my1), (mx2, my2), COLORES['matricula'], 2)
                    etiqueta_placa = f"{texto_matricula} ({conf_mat_str})" if texto_matricula else "Matricula"
                    cv2.putText(
                        frame, etiqueta_placa,
                        (mx1, max(15, my1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORES['matricula'], 2
                    )

        cv2.rectangle(frame, (15, 15), (270, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (15, 15), (270, 160), (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        y_offset = 62
        for clase, ids in conteo_unicos.items():
            color_txt = COLORES.get(clase, (200, 200, 200))
            cv2.putText(frame, f"Total {clase}s: {len(ids)}", (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_txt, 1)
            y_offset += 22

        out.write(frame)
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print(f"Progreso: {frame_idx}/{total_frames} fotogramas procesados...")

cap.release()
out.release()
f_csv.close()
```

Video resultado procesado con YOLO: https://www.youtube.com/watch?v=VU8vAfFiS4U

Video resultado procesado con RT-DETR: https://www.youtube.com/watch?v=x1TQoax1NNQ

