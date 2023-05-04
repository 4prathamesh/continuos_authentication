import cv2
import os
import numpy as np

# Load the embeddings and labels from the file
data = np.load('embeddings.npz')
embeddings = data['embeddings']
labels = data['labels']

# Load the pre-trained deep neural network for face recognition
model = cv2.dnn.readNetFromTorch('nn4.small2.v1.t7')

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
datasets='database'
# Set the minimum distance threshold for face recognition
threshold = 0.60


names = []

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names.append(subdir)
# Start the video capture
cap = cv2.VideoCapture(0)

# Loop through the video frames
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    frame = cv2.resize(frame, (900, 500))
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the frame
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through the detected faces
    for (x, y, w, h) in faces_rect:
        # Extract the face region from the frame
        face = gray[y:y + h, x:x + w]

        # Resize the face image to a fixed size
        face = cv2.resize(face, (96, 96))

        # Compute the embedding (i.e. feature vector) of the face using the pre-trained deep neural network
        blob = cv2.dnn.blobFromImages(
            np.repeat(np.array([face])[..., np.newaxis], 3, -1).astype(np.float32) / 255.0, 1.0, (96, 96),
            (0, 0, 0), False)

        model.setInput(blob)
        embedding = model.forward()

        # Compute the Euclidean distance between the embedding of the test face and the embeddings of the training faces
        distances = np.linalg.norm(embeddings - embedding, axis=1)

        # Find the label (i.e. ID) of the closest matching training face
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        label = labels[min_distance_idx]

        # If the minimum distance is below the threshold, authenticate the user
        if min_distance < threshold:
            # Draw a green rectangle around the detected face and label it with the ID of the closest matching training face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{names[label]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
        else:
            # Draw a red rectangle around the detected face and label it as unauthenticated
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "unauthenticated", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Face Authentication', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, quit the loop
    if key == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
