import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import json

tf.keras.backend.clear_session()
# Load MobileNet SSD v2 model from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")




def LoadImage(image_ip):

    image = cv2.cvtColor(image_ip, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (320, 320))  # Resize to match model's input shape
    image = np.expand_dims(image, 0)  # Add batch dimension
    image = image.astype(np.uint8)  # Convert to uint8

    # Convert numpy array to Tensor
    image = tf.convert_to_tensor(image)
    return image

# Perform inference using the loaded model



# Retrieve the predicted bounding boxes, classes, and scores



# Set a score threshold
score_threshold = 0.5


# Load the COCO Label Map
category_index = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
}
#def Draw(input_image):
# Iterate through all the bounding boxes and draw them on the image
#    for box, class_id, score in zip(bounding_boxes, classes, scores):
#        if score > score_threshold:
#            ymin, xmin, ymax, xmax = box.numpy()
#            xmin = int(xmin * input_image.shape[1])
##            xmax = int(xmax * input_image.shape[1])
 #           ymin = int(ymin * input_image.shape[0])
  #          ymax = int(ymax * input_image.shape[0])
#
            # Draw bounding box rectangle and label on the image
 #           cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
  ##         cv2.putText(input_image, class_name, (xmin, ymin - 10),
    #                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)



   # cv2.imshow("Object Detection", input_image)
  ## cv2.destroyAllWindows()

def JsonFile(inputimg):
    model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

    image = cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(inputimg, (320, 320))  # Resize to match model's input shape
    image = np.expand_dims(image, 0)  # Add batch dimension
    image = image.astype(np.uint8)  # Convert to uint8

    # Convert numpy array to Tensor
    image = tf.convert_to_tensor(image)
    detections = model(image)
    bounding_boxes = detections["detection_boxes"][0]
    classes = detections["detection_classes"][0]
    scores = detections["detection_scores"][0]


    # Set a score threshold
    score_threshold = 0.5

    # Get data
    data=[]
    for box, class_id, score in zip(bounding_boxes, classes, scores):
        if score > score_threshold:
            ymin, xmin, ymax, xmax = box.numpy()
            xmin = int(xmin * inputimg.shape[1])
            xmax = int(xmax * inputimg.shape[1])
            ymin = int(ymin * inputimg.shape[0])
            ymax = int(ymax * inputimg.shape[0])
            obj_info = {
                "Label": category_index[int(class_id)],
                "Score": float(score),
                "Boundingbox": [int((xmax-xmin)/2), int((ymax-ymin)/2)]
            }
            data.append(obj_info)
            # Convert data to JSON
            json_data = json.dumps(data, indent=4)

            # Write JSON to file
            with open("data.json", "w") as file:
                file.write(json_data)

    return json_data

