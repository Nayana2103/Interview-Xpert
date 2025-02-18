from django.http import HttpResponse
from django.shortcuts import render
from django.shortcuts import redirect
# FILE UPLOAD AND VIEW
from  django.core.files.storage import FileSystemStorage
# SESSION
from django.conf import settings
from .models import *
from django.db.models import F
import os
#from ML import launcher
from django.http import StreamingHttpResponse, JsonResponse
from threading import Thread, Event
import pandas as pd
import cv2
import numpy as np
from keras.models import load_model
import pyaudio
import wave
from ML.utils.datasets import get_labels
from ML.utils.inference import draw_text, draw_bounding_box, apply_offsets
from ML.utils.preprocessor import preprocess_input
from keras import backend as K
from ML import audio_process, resume_analysis

# Global stop event
stop_event = Event()

def generate_frames():
    """Generator function for streaming video frames."""
    stop_event.clear()  # Reset stop flag when streaming starts
    emotion_model_path = './ML/models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')

    # Load models
    face_cascade = cv2.CascadeClassifier('./ML/models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)

    emotion_target_size = emotion_classifier.input_shape[1:3]

    emotion_data = []

    cap = cv2.VideoCapture(0)  # Webcam source

    while cap.isOpened():
        if stop_event.is_set():  # Stop streaming when stop event is triggered
            break

        ret, bgr_image = cap.read()
        if not ret:
            break

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, (20, 40))
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_prediction[0][2] = emotion_prediction[0][2] * 7
            emotion_prediction[0][4] = emotion_prediction[0][4] * 3
            emotion_prediction[0][3] = emotion_prediction[0][3] * 4
            emotion_prediction[0][5] = emotion_prediction[0][5] * 9
            #emotion_prediction[0][6] = emotion_prediction[0][6] / 2
            emotion_data.append(emotion_prediction[0])
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]

            # Define colors based on emotion
            emotion_colors = {
                'angry': (255, 0, 0),
                'sad': (0, 0, 255),
                'happy': (255, 255, 0),
                'surprise': (0, 255, 255),
                'fear': (238, 130, 238),
                'disgust': (255, 20, 147),
                'neutral': (0, 255, 0)
            }
            color = emotion_probability * np.asarray(emotion_colors.get(emotion_text, (0, 255, 0)))
            color = color.astype(int).tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        _, jpeg = cv2.imencode('.jpg', bgr_image)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    K.clear_session()

    # Save emotion data
    emotion_df = pd.DataFrame(emotion_data, columns=emotion_labels.values())
    emotion_df.to_csv("./ML/emotion_df.csv", index=False)

def start_audio_capture():
    """Function to capture audio in a separate thread."""
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    WAVE_OUTPUT_FILENAME = "./ML/file.wav"

    audio = pyaudio.PyAudio()
    print("Audio recording started")

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    while not stop_event.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    print("Finished recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save recorded audio
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def start_interview(request):
    """Starts the interview process with video streaming and audio recording."""
    # Start audio recording in a separate thread
    audio_thread = Thread(target=start_audio_capture)
    audio_thread.start()

    # Return video streaming response
    return StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace; boundary=frame")

def stop_interview(request):
    """Stops both the video and audio streams."""
    stop_event.set()
    return JsonResponse({"message": "Interview stopped successfully"})

def scale_to_rating(value, min_value = 0, max_value = 1):
    new_min = 0
    new_max = 5

    scaled_value = ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
    return scaled_value

def interview_analysis(request):
    text, duration = audio_process.audio_text()
    print("Text data:", text)
    no_of_words = len(text.split())
    word_per_second = no_of_words / duration

    print("Words per minute:",word_per_second)

    if word_per_second > 2:
        response_score = scale_to_rating(2, min_value =0, max_value = 2)
    else:
        response_score = scale_to_rating(word_per_second, min_value =0, max_value = 2)

    emotion_df = pd.read_csv("./ML/emotion_df.csv")
    count_row = emotion_df.shape[0]

    print("Number of rows:",count_row)
    if count_row != 0:

        angry = emotion_df["angry"].median()
        disgust = emotion_df["disgust"].median()
        fear = emotion_df["fear"].mean()
        happy = emotion_df["happy"].median()
        sad = emotion_df["sad"].mean()
        surprise = emotion_df["surprise"].median()
        neutral = emotion_df["neutral"].median()

        calm = scale_to_rating(neutral)
        excitement = scale_to_rating((surprise + happy)/2)
        stress = scale_to_rating((fear + sad)/2)

        confidence = (calm + response_score + excitement + (5 - stress))/4

        print(f"Neutral: {neutral}, Surprise: {surprise}, happy: {happy}, Fear: {fear}, Sad: {sad}, Angry: {angry}, Disgust: {disgust}, word_per_second: {word_per_second}")

        print(f"Clam: {calm}\n Excitement: {excitement}\n Stress: {stress}\n Response_score: {response_score}\n Confidence Score: {confidence}")

        userdetails = regtable.objects.get(id=request.session['userid'])

        result = resume_analysis.analyze_pdf_keywords(userdetails.resume)
        print("result:",result)


        InterviewAnalysis.objects.update_or_create(
                                                        user_id=request.session['userid'],
                                                        defaults={
                                                            'calm': calm,
                                                            'excitement': excitement,
                                                            'stress': stress,
                                                            'response_score': response_score,
                                                            'confidence_score': confidence,
                                                            'resume_analysis': str(result)
                                                        }
                                                    )
        return render(request,'interview_analysis.html',{'calm':calm, 'excitement':excitement, 
                                                         'stress':stress, 'response_score':response_score, 
                                                         'confidence_score':confidence, 'resume_analysis':result,
                                                         'userdata':userdetails})
    else:
        calm = excitement = stress = confidence = 0
        userdetails = regtable.objects.get(id=request.session['userid'])

        result = resume_analysis.analyze_pdf_keywords(userdetails.resume)
        print("result:",result)
        InterviewAnalysis.objects.update_or_create(
                                                        user_id=request.session['userid'],
                                                        defaults={
                                                            'calm': calm,
                                                            'excitement': excitement,
                                                            'stress': stress,
                                                            'response_score': response_score,
                                                            'confidence_score': confidence,
                                                            'resume_analysis': str(result)
                                                        }
                                                    )
        return render(request,'interview_analysis.html',{'calm':calm, 'excitement':excitement, 
                                                         'stress':stress, 'response_score':response_score, 
                                                         'confidence_score':confidence, 'resume_analysis':result,
                                                         'userdata':userdetails})


def view_results(request, id):
    results = InterviewAnalysis.objects.get(user_id=id)
    userdetails = regtable.objects.get(id=id)
    print("resume analysis:",results.resume_analysis)
    resume_analysis = eval(results.resume_analysis)
    return render(request,'interview_analysis.html',{'calm':float(results.calm), 'excitement':float(results.excitement), 
                                                     'stress':float(results.stress), 'response_score':float(results.response_score), 
                                                     'confidence_score':float(results.confidence_score), 
                                                     'resume_analysis':resume_analysis,
                                                     'userdata':userdetails})




def home(request):
    return render(request,'index.html')

def index(request):
    return render(request,'index.html')

def register(request):
    return render(request,'register.html')

def addregister(request):
    if request.method=="POST":
        a=request.POST.get('name') 
        b=request.POST.get('email')
        c=request.POST.get('password')
        d=request.POST.get('phone') 
        resume = request.FILES["resume"]
        fs = FileSystemStorage()
        filename = fs.save("resume", resume)
        file_url = fs.url(filename)  
        ins=regtable(name=a,email=b,password=c,phone=d,resume = file_url)
        ins.save()
    return render(request,'register.html')

def login(request):
    return render(request,'login.html')

def addlogin(request):
    email=request.POST.get('email')
    password=request.POST.get('password')
    if email=='admin@gmail.com'and password=='admin':
       request.session['admin@gmail.com']='admin@gmail.com'
       request.session['admin']='admin'
       ins=regtable.objects.all()
       return render(request,'index.html')

    elif regtable.objects.filter(email=email,password=password).exists():
        userdetails=regtable.objects.get(email=email,password=password)
        if userdetails.email==request.POST['email']:
            request.session['userid']=userdetails.id
            request.session['username']=userdetails.name 
            return render(request,'index.html')  
    else:
         return render(request,'login.html')
def logout(request):
    session_keys=list(request.session.keys())   
    for key in session_keys:
            del request.session[key] 
    return redirect(index)

def viewuser(request):
    user=regtable.objects.all()
    return render(request,'viewuser.html',{'result':user})

def speech_recognition(request):
    object_search = speech_to_text.predict()
    request.session["object_search"] = object_search
    return render(request,'navigation.html')

def attend_interview(request):
    return render(request,'attend_interview.html')



def navigation(request):
    object_det_cmd = 'python object_detect_v7/detect.py --weights object_detect_v7/yolov7x.pt --source 0 --view-img --object-search "{}"'.format(request.session["object_search"])
    print("Object det command:\n",object_det_cmd)
    os.system(object_det_cmd)
    return render(request,'navigation.html')


def upload(request):
    return render(request,'upload.html')

def addupload(request):
    if request.method == "POST":
        myfile=request.FILES['file'] 
        fs=FileSystemStorage()
        filename=fs.save(myfile.name,myfile)
        try:
            os.remove(os.path.join(settings.MEDIA_ROOT,'input/test/test.csv'))
        except:
            pass
        fs=FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT,'input/test/'))  
        fs.save("test.csv", myfile) 
        fs=FileSystemStorage()
        fs.save(myfile.name,myfile) 
        k.clear_session()

        result=test1.predict()
        ins=uploadtable(images=filename,user_id=request.session['userid'],result=result)
        ins.save()
    return render(request,'upload.html',{'result':result})

def viewupload(request):
    user=uploadtable.objects.all()
    return render(request,'viewupload.html',{'result':user})

def prediction(request):
    return render(request,'prediction.html')   

def predict(request):
    if request.method == "POST":
        # Extract form data
        age = request.POST.get('age', 0)
        sex = request.POST.get('sex', 0)
        Chestpaintype = request.POST.get('Chestpaintype', 0)
        RestingBP = request.POST.get('RestingBP', 0)
        Cholesterol = request.POST.get('Cholesterol', 0)
        FastingBS = request.POST.get('FastingBS', 0)
        RestingECG = request.POST.get('RestingECG', 0)
        MaxHR = request.POST.get('MaxHR', 0)
        ExerciseAngina = request.POST.get('ExerciseAngina', 0)
        Oldpeak = request.POST.get('Oldpeak', 0)
        ST_Slope = request.POST.get('ST_Slope', 0)

        # Create a dictionary with data
        data = {
            ' ': '528',
            'age': [age],
            'sex': [sex],
            'Chestpaintype': [Chestpaintype],
            'RestingBP': [RestingBP],
            'Cholesterol': [Cholesterol],
            'FastingBS': [FastingBS],
            'RestingECG': [RestingECG],
            'MaxHR': [MaxHR],
            'ExerciseAngina': [ExerciseAngina],
            'Oldpeak': [Oldpeak],
            'ST_Slope': [ST_Slope],
            'HeartDisease':' '
        }

        csv_file_path = os.path.join(settings.MEDIA_ROOT, 'input/test/test.csv')
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
        # Create DataFrame
        df = pd.DataFrame(data)

        # Remove blank column (if any)
        # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Write DataFrame to CSV
        df.to_csv(csv_file_path, index=False)
        result=test1.predict()
        un= request.session['username']
        ins=restable(result=result,user_id=un)
        ins.save()
    return render(request,'result.html',{'result':result})


def viewresult(request):
    rs=restable.objects.all()
    return render(request,'viewresult.html',{'result':rs}) 