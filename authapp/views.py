import json
import time
import threading
import os
import cv2
import face_recognition
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.conf import settings
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Person

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
    else:
        form = AuthenticationForm()
    
    return render(request, 'authpage/login.html', {'form': form})

def home_view(request):
    return render(request, 'home.html')

recognized_person = None
known_face_encodings, known_face_names, known_persons = [], [], []

def load_known_faces():
    global known_face_encodings, known_face_names, known_persons
    known_face_encodings = []
    known_face_names = []
    known_persons = []

    persons = Person.objects.all()
    for person in persons:
        photo_path = os.path.join(settings.MEDIA_ROOT, person.photo.name)
        if os.path.exists(photo_path):
            image = face_recognition.load_image_file(photo_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(f"{person.first_name} {person.last_name}")
                known_persons.append(person)

load_known_faces()

def detect_faces(camera, timeout):
    global recognized_person
    start_time = time.time()
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                recognized_person = known_persons[first_match_index]
                camera.release()
                return
        
        if time.time() - start_time > timeout:
            camera.release()
            return

def face_recognition_view(request):
    global recognized_person
    recognized_person = None

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        # print("Error: Could not open camera.")
        return render(request, 'no_person.html', {'message': 'Error: Could not open camera.'})

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detection_thread = threading.Thread(target=detect_faces, args=(camera, 1))
    detection_thread.start()
    detection_thread.join()

    if recognized_person:
        return redirect('person_info', person_id=recognized_person.id)
    
    return render(request, 'no_person.html', {'message': 'No person recognized'})

def person_info_view(request, person_id):
    person = Person.objects.get(id=person_id)
    return render(request, 'person_info.html', {'person': person})



MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(2-6)', '(6-10)', '(10-15)', '(15-18)', '(18-22)', '(22-25)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Paths to your model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
faceProto = os.path.join(BASE_DIR, 'static/models/opencv_face_detector.pbtxt')
faceModel = os.path.join(BASE_DIR, 'static/models/opencv_face_detector_uint8.pb')
ageProto = os.path.join(BASE_DIR, 'static/models/age_deploy.prototxt')
ageModel = os.path.join(BASE_DIR, 'static/models/age_net.caffemodel')
genderProto = os.path.join(BASE_DIR, 'static/models/gender_deploy.prototxt')
genderModel = os.path.join(BASE_DIR, 'static/models/gender_net.caffemodel')

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob1 = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob1)
    detections = net.forward()
    faceBoxes1 = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes1.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes1

def gen_frames():
    padding = 20
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            resultImg, faceBoxes = highlightFace(faceNet, frame)
            if faceBoxes:
                for faceBox in faceBoxes:
                    face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                                 max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]

                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]

                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]

                    cv2.putText(resultImg, f'{gender}, {age[1:-1]} years', (faceBox[0], faceBox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', resultImg)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@csrf_exempt
def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

