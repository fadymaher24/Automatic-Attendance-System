import cv2
import numpy as np
import face_recognition

imgfady = face_recognition.load_image_file('image basic/fady.jpg')
imgfady = cv2.cvtColor(imgfady, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('image basic/kiro.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
faceLoc = face_recognition.face_locations(imgfady)[0]
encodefady = face_recognition.face_encodings(imgfady)[0]
cv2.rectangle(imgfady, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (247, 846, 632, 461), 2)
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (247, 846, 632, 461), 2)
results = face_recognition.compare_faces([encodefady], encodeTest)
faceDis =face_recognition.face_distance([encodefady], encodeTest)

print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 )
cv2.imshow('fady', imgfady)
cv2.imshow('fadyTest', imgTest)
cv2.waitKey(0)