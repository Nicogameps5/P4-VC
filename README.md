# Práctica 4 VC

Ejercicios de la cuarta práctica de laboratorio de VC.

Nicolás Hernández Castro y Gabriel Godoy Navarro - EII ULPGC 13 de noviembre de 2026


## Entrenamiento

Dado que los pesos preentrenados del conjunto COCO no contemplan la categoría de matrícula de vehículos, se procedió al entrenamiento supervisado de un detector especializado. El proceso se llevó a cabo en Google Colab con GPU NVIDIA Tesla T4 (16 GB VRAM), empleando un conjunto de datos particionado en subdirectorios normalizados (`train`, `valid`, `test`) con una única clase (`names: ['matricula']`).

El entrenamiento del modelo convolucional base se configuró de la siguiente forma:

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

* Rendimiento global ($mAP$): El porcentaje de acierto general ($mAP_{50}$) sube enseguida hasta situarse en un superior al  90%. A su vez, la métrica más exigente ($mAP_{50-95}$), que mide si la caja está milimétricamente ajustada a los bordes de la matrícula, progresa poco a poco hasta casi el 80%, confirmando que el modelo encuadra las placas con gran exactitud.

### Apartado Extra: Evaluación comparativa YOLO vs. RT-DETR
Como ampliación, se evalua una alternativa moderna basada en Transformers (RT-DTER). 

Para llevar a cabo el experimento en igualdad de condiciones, se entrenó rtdetr-l.pt sobre el mismo fichero conjunto de datos durante 50 épocas a resolución 640. Debido a la alta demanda de memoria de las matrices de atención del Transformer, fue necesario ajustar el tamaño de lote a batch=4 (frente a batch=16 de YOLO) para no desbordar los 16 GB de VRAM de la GPU T4:

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

Además, destacar que el entrenamiento con YOLO fue notablemente más rapido (0.072 horas) que con RT-DETR (0.274 horas).

