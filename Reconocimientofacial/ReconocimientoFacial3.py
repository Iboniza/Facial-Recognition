import cv2
import os
import pygame

dataPath = 'C:/Users/Ibons PC/Desktop/Python/FR/Reconocimientofacial/Data'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo
# face_recognizer.read('modeloEigenFace.xml')
# face_recognizer.read('modeloFisherFace.xml')
face_recognizer.read('modeloLBPHFace.xml')

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture('elibon.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar Pygame
pygame.mixer.init()

# Cargar el sonido "miau.wav"
miau_sound = pygame.mixer.Sound('C:/Users/Ibons PC/Desktop/Python/FR/Reconocimientofacial/Audios/JulioBorracho.wav')

# Variable para controlar la reproducción del audio
audio_reproduciendo = False

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 70:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if imagePaths[result[0]] == 'Ibon' and not audio_reproduciendo:  # Verificar si la cara reconocida es la de Ibon y el audio no está en reproducción
                miau_sound.play()  # Reproducir el sonido "Miau" directamente
                audio_reproduciendo = True  # Marcar que el audio está en reproducción
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    if not pygame.mixer.get_busy() and audio_reproduciendo:
        audio_reproduciendo = False  # Reiniciar la variable cuando el audio termina

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()




