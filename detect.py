#code 03:
import cv2
import os

dir_path = "database"
# Load the image
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

# Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect the faces in the image
faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces_rect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with the detected faces
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
