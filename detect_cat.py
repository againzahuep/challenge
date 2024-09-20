import cv2
import numpy as np

def load_yolo():
    # Load YOLO
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def detect_cats(image_path):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Load YOLO
    net, output_layers = load_yolo()

    # Prepare the image for detection
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Perform detection
    detections = net.forward(output_layers)

    # Initialize lists to hold detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop through detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Load class labels
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Draw bounding boxes for detected cats
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "cat":  # Check if the detected object is a cat
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label + " " + str(round(confidences[i], 2)),
                            (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_cats("images/cat.jpg")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
