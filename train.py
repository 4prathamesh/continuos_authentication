#code 02:
import cv2
import os
import numpy as np

# Set the directory containing face images
dir_path = "database"

# Create an empty list to store face images and corresponding labels
faces = []
labels = []
names = {}
id = 0

# Loop through the images in the directory
for subdir in os.listdir(dir_path):
    names[id] = subdir
    subjectpath = os.path.join(dir_path, subdir)
    for img_name in os.listdir(subjectpath):
        path = os.path.join(subjectpath, img_name)
        label = id
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect the face in the image using a pre-trained Haar cascade classifier
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces_rect = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

        # Loop through the detected faces
        for (x, y, w, h) in faces_rect:
            # Extract the face region from the image
            face = img[y:y + h, x:x + w]

            # Resize the face image to a fixed size
            face = cv2.resize(face, (96, 96))

            # Append the face image and corresponding label to the lists
            faces.append(face[..., 0])
            labels.append(label)

    id += 1

# Convert the lists to NumPy arrays
faces = np.array(faces)
labels = np.array(labels)

# Load the neural network model
model_file = 'nn4.small2.v1.t7'
if not os.path.isfile(model_file):
    print('Error: The model file %s does not exist' % model_file)
    exit()
try:
    model = cv2.dnn.readNetFromTorch(model_file)
except cv2.error as e:
    print('Error: Could not load the model file %s' % model_file)
    print(e)
    exit()

# Generate embeddings for the face images
blob = cv2.dnn.blobFromImages(np.repeat(faces[..., np.newaxis], 3, -1).astype(np.float32) / 255.0, 1.0, (96, 96), (0, 0, 0), False)

model.setInput(blob)
embeddings = model.forward()

# Save the embeddings and labels to a file
np.savez('embeddings.npz', embeddings=embeddings, labels=labels)
