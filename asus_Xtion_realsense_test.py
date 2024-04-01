import cv2
#import openai

from openai import OpenAI


# Initialize the Asus Xtion Live Depth Camera
capture = cv2.VideoCapture(cv2.CAP_OPENNI2 + cv2.CAP_OPENNI2_ASUS)

# Load a pre-trained object detection model (for example, YOLO)
#net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net = cv2.dnn.readNet("/mnt/d/Project/sandbox/darknet/yolov3.weights", "/mnt/d/Project/sandbox/darknet/yolov3.cfg")

# Load the COCO class labels (common objects in context)
#with open("coco.names", "r") as f:
with open("/mnt/d/Project/sandbox/darknet/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

'''
# Function to interact with GPT-3.5 and get the object of interest
def get_object_from_gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()
'''
# Update the code to use openai.ChatCompletion instead of openai.Completion
def get_object_from_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message['content'].strip()

# Function to perform object detection and highlight the specified object
def highlight_object(frame, object_name):
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > 0.5 and classes[class_id] == object_name:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(confidence)
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

try:
    while True:
        # Use GPT-3.5 to ask the user what they are looking for in the picture
        user_query = input("What are you looking for in the picture? ")
        
        # Get the object from GPT-3.5 based on the user's query
        object_name = get_object_from_gpt(user_query)

        # Read a new frame from the Asus Xtion Live Depth Camera
        ret, frame = capture.read()
        if not ret:
            break

        # Perform object detection and highlight the specified object
        highlighted_frame = highlight_object(frame, object_name)

        # Display the original and highlighted frames
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Highlighted Frame", highlighted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the Asus Xtion Live Depth Camera
    capture.release()
    cv2.destroyAllWindows()
