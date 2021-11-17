import cv2
import cvzone

face_detector = cv2.CascadeClassifier('pack/haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('pack/haarcascade_eye.xml')
lip_detector = cv2.CascadeClassifier('pack/haarcascade_smile.xml')

face_sticker = cv2.imread('emojy/face.png', cv2.IMREAD_UNCHANGED)
eye_sticker = cv2.imread('emojy/eye.png', cv2.IMREAD_UNCHANGED)
lip_sticker = cv2.imread('emojy/lip.png', cv2.IMREAD_UNCHANGED)

vedio_cap = cv2.VideoCapture(0)

flag = 0

def face_emojy(frame, f):
    faces = face_detector.detectMultiScale(frame, 1.3, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_sticker_resize = cv2.resize(face_sticker, (w, h))

        frame = cvzone.overlayPNG(frame, face_sticker_resize, [x, y])

    f = 1

    return frame,f

def lip_eye_emojy(frame, f):
    eyes = eye_detector.detectMultiScale(frame, 1.2, minSize=(50, 50))

    for (x, y, w, h) in eyes:
        eye_sticker_resize = cv2.resize(eye_sticker, (w, h))

        frame = cvzone.overlayPNG(frame, eye_sticker_resize, [x, y])
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

    lip = lip_detector.detectMultiScale(frame, 1.5, minSize=(30, 30), maxSize=(100, 100))

    for (x, y, w, h) in lip:
        lip_sticker_resize = cv2.resize(lip_sticker, (w, h))

        frame = cvzone.overlayPNG(frame, lip_sticker_resize, [x, y])

    f = 2

    return frame, f

def checkered_face(frame, f):
    faces = face_detector.detectMultiScale(frame, 1.3, minSize=(100, 100))

    for (x, y, w, h) in faces:
        chekered = cv2.resize(frame[y:y+h, x:x+w], (20, 20))
        result = cv2.resize(chekered, (w, h), interpolation= cv2.INTER_NEAREST)
        frame[y:y+h, x:x+w] = result

    f = 3

    return frame, f

def rotate_face(frame, f):
    faces = face_detector.detectMultiScale(frame, 1.3, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = cv2.flip(face, 0)

    f = 4

    return frame, f

while True:
    ret, frame = vedio_cap.read()

    if ret == False:
        break

    frame = cv2.resize(frame, (500, 500))

    key = cv2.waitKey(1)
    if key == ord('0'):
        flag = 0
    elif key == ord('5'):
        break
    elif key == ord('1') or flag == 1:
        frame,flag = face_emojy(frame, flag)
    elif key == ord('2') or flag == 2:
        frame, flag = lip_eye_emojy(frame, flag)
    elif key == ord('3') or flag == 3:
        frame, flag = checkered_face(frame, flag)
    elif key == ord('4') or flag == 4:
        frame, flag = rotate_face(frame, flag)
    
    cv2.imshow('output', frame)
