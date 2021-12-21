import cv2
import sys

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

try:
    image = cv2.imread(str(sys.argv[1]))
except IndexError:
    image = cv2.imread("images/sample_image.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_image,
                                      scaleFactor=1.05,
                                      minNeighbors=5)

for x, y, w, h in faces:
    image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

resized_image = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))

cv2.imshow("Gray", image)
cv2.waitKey(7000)
cv2.destroyAllWindows()
