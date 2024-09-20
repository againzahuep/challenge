import cv2
import numpy as np


# Cargar el modelo YOLO
def load_yolo_model():
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers


# Detectar objetos en una imagen
def detect_objects(image, net, output_layers):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Umbral de confianza
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_boxes = [boxes[i] for i in indices.flatten()]

    return detected_boxes, class_ids


# Sustituir gato por perro
def replace_cat_with_dog(cat_image_path, dog_image_path):
    # Cargar imágenes
    cat_image = cv2.imread(cat_image_path)
    dog_image = cv2.imread(dog_image_path)

    # Cargar modelo YOLO
    net, output_layers = load_yolo_model()

    # Detectar gatos en la imagen del gato
    cat_boxes, class_ids = detect_objects(cat_image, net, output_layers)

    # Filtrar solo las detecciones de gatos (ID de clase para gato es 16 en COCO)
    cat_box = [box for box, cls_id in zip(cat_boxes, class_ids) if cls_id == 16]

    if not cat_box:
        print("No se detectó ningún gato.")
        return

    # Obtener la primera caja delimitadora del gato
    x, y, w, h = cat_box[0]

    # Recortar la región del gato
    cat_region = cat_image[y:y + h, x:x + w]

    # Detectar perros en la imagen del perro
    dog_boxes, dog_class_ids = detect_objects(dog_image, net, output_layers)

    # Filtrar solo las detecciones de perros (ID de clase para perro es 16 en COCO)
    dog_box = [box for box, cls_id in zip(dog_boxes, dog_class_ids) if cls_id == 16]

    if not dog_box:
        print("No se detectó ningún perro.")
        return

    # Obtener la primera caja delimitadora del perro
    dx, dy, dw, dh = dog_box[0]

    # Recortar la región del perro
    dog_region = dog_image[dy:dy + dh, dx:dx + dw]

    # Redimensionar la región del perro para que encaje en la región del gato
    dog_resized = cv2.resize(dog_region, (w, h))

    # Sustituir la región del gato con la región redimensionada del perro
    cat_image[y:y + h, x:x + w] = dog_resized

    # Guardar o mostrar la imagen modificada
    cv2.imwrite('imagen_modificada.jpg', cat_image)
    cv2.imshow('Imagen Modificada', cat_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    replace_cat_with_dog('images/cat.jpg', 'images/dog.jpg')
