import cv2
import numpy as np
from PlayerTracking import trackPlayer

# Load Yolo
net = cv2.dnn.readNet("files/yolov3.weights", "files/yolov3.cfg")
# classes = []
with open("files/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(r'data/AtelticoU17vsIndiaU172nd5min.mp4')

ret, frame = cap.read()

# frame = imutils.resize(frame, width=1000,height=1000)
# print(frame.shape)
cv2.imshow('Frame', frame)
box = cv2.selectROI('Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(box)
(x, y, w, h) = [int(a) for a in box]

while cap.isOpened():
    ret, frame = cap.read()

    if ret:

        copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)
        # outs = get_players(outs, height, width)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
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

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # print(indexes)

        # Tracking
        output = trackPlayer(frame, box)

        font = cv2.FONT_HERSHEY_PLAIN
        # print(len(boxes))
        max_ = 0
        players = []
        myplayer = boxes[0]
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                if label == 'person':
                    players.append(boxes[i])
                    player_box = output[y:y + h, x:x + w]
                    total = sum(sum(player_box))
                    # print(total)
                    if max_ < total:
                        max_ = total
                        myplayer = boxes[i]
        # print('players : ',len(players))
        for player in players:
            x, y, w, h = player
            if player == myplayer:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                box = player
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
