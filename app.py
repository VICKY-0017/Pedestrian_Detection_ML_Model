import gradio as gr
import numpy as np
import cv2
import imutils

NMS_Threshold = 0.3
MIN_Confidence = 0.2

def pedestrian_detection(image, personidz=0, label_names=None):
    (H, W) = image.shape[:2]
    results = []
    
    blb = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blb)
    layerOutputs = model.forward(layer_name)

    boxes = []
    centroids = []
    confidences = []

    for result in layerOutputs:
        for detection in result:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_Confidence:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_Confidence, NMS_Threshold)
    if len(idzs) > 0:
        for i in idzs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)
    return results

def process_video(video_path, label_names):
    capture = cv2.VideoCapture(video_path)
    while True:
        (grabbed, image) = capture.read()
        if not grabbed:
            capture.release()
            break
        image = imutils.resize(image, width=700)
        
        # Check if "person" label exists in label_names
        if "person" in label_names:
            results = pedestrian_detection(image, personidz=label_names.index("person"))
            for res in results:
                cv2.rectangle(image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)
        else:
            print("Warning: 'person' label not found in label_names.")
            
        yield image

# Gradio Interface
iface = gr.Interface(
    fn=process_video,
    inputs=["file", "text"],
    outputs="image",
    live=True,
)

# Launch the Gradio interface
iface.launch(share=True, debug=True)
