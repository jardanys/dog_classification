import cv2
import numpy as np
import time
import pandas as pd
import sys
from pyimagesearch.centroidtracker import CentroidTracker
import argparse
import imutils

vector = []
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture('videoprueba3.mp4')
scaling_factor = 1
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
ct = CentroidTracker()
(H, W) = (None, None)

while (cap.isOpened()):
	ret, frame = cap.read()
	frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
	if len(face_rects)==1:
		time_rects = time.ctime()
		vector.append(time_rects)
		for (x,y,w,h) in face_rects:
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
	lista_nueva = []
	for i in vector:
		if i not in lista_nueva:
				lista_nueva.append(i)
	df = pd.DataFrame(lista_nueva)
	df.to_csv('FechaHora.csv')
	out.write(frame)
	objects = ct.update(rects)
	cv2.imshow('Face Detector', frame)

	c = cv2.waitKey(1)
	if c == 27:
		break

cap.release()
out.release()
cv2.destroyAllWindows()
