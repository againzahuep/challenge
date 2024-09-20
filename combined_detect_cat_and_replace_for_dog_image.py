import cv2
import numpy as np


def load_yolo():
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers


def overlay_image(background, overlay, position):
    x, y = position
    h, w = overlay.shape[:2]

    # Ensure the overlay fits within the background
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background

    # Create a mask of the overlay and create its inverse mask
    overlay_mask = overlay[:, :, 3] if overlay.shape[2] == 4 else np.ones((h, w), dtype=np.uint8) * 255
    overlay_mask_inv = cv2.bitwise_not(overlay_mask)

    # Black-out the area of the overlay in the background
    background_area = background[y:y + h, x:x + w]
    background_area = cv2.bitwise_and(background_area, background_area, mask=overlay_mask_inv)

    # Take only the region of the overlay from the overlay image
    overlay_area = cv2.bitwise_and(overlay[:, :, :3], overlay[:, :, :3], mask=overlay_mask)

    # Put the overlay in the correct place
    combined_area = cv2.add(background_area, overlay_area)
    background[y:y + h, x:x + w] = combined_area

    return background


def detect_and_replace_cats(image_path, dog_image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    net, output_layers = load_yolo()

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load the dog image
    dog_image = cv2.imread(dog_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "cat":
                # Resize dog image to fit the bounding box of the cat
                dog_resized = cv2.resize(dog_image, (w, h), interpolation=cv2.INTER_AREA)

                # Overlay the dog image on the detected cat area
                image = overlay_image(image, dog_resized, (x, y))

    cv2.imshow("Combined - Replaced Cat for Dog", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_and_replace_cats("images/cat.jpg", "images/dog.png")