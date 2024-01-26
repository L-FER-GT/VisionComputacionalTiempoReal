import cv2 #opencv

import numpy as np

#PROGRAMA DE CLASIFICACION DE OBJETOS PARA VIDEO EN DIRECCION IP 

url = 'http://192.168.1.6/cam-hi.jpg'
#url = 'http://192.168.1.6/'
cap = cv2.VideoCapture("http://192.168.0.107:8080/video")
winName = 'camera Recongnition'
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
#net.setInputSize(480,480)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

width = 800  # Ancho de la ventana
height = 480  # Altura de la ventana

while True:
    # Capturar un frame
    ret, frame = cap.read()

    # Redimensionar el frame
    resized_frame = cv2.resize(frame, (width, height))

    imgNp = np.array(resized_frame, dtype=np.uint8)
    img = resized_frame #decodificamos

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # vertical
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #black and white
    

    classIds, confs, bbox = net.detect(img,confThreshold=0.5)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness = 3) #mostramos en rectangulo lo que se encuentra
            cv2.putText(img, classNames[classId-1], (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)


    cv2.imshow(winName,img) # mostramos la imagen

    #esperamos a que se presione ESC para terminar el programa
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q') or tecla == 27:
        break
cv2.destroyAllWindows()