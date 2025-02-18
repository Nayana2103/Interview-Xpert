from threading import Thread, Event
import pandas as pd
import cv2
import numpy as np
from keras.models import load_model
import pyaudio
import wave
from utils.datasets import get_labels
from utils.inference import draw_text, draw_bounding_box, apply_offsets
from utils.preprocessor import preprocess_input
from keras import backend as K
import audio_process


# Global stop event
stop_event = Event()


def start_realTimeVideo():
    # Parameters for loading data and images
    emotion_model_path = './ML/models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')

    # Hyper-parameters for bounding boxes shape
    emotion_offsets = (20, 40)

    # Loading models
    face_cascade = cv2.CascadeClassifier('./ML/models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)

    # Getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    emotion_data = []

    # Starting video streaming
    cv2.namedWindow('window_frame')
    cap = cv2.VideoCapture(0)  # Webcam source

    while cap.isOpened():
        ret, bgr_image = cap.read()
        if not ret:
            break

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
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
        # cv2.imshow('window_frame', bgr_image)

        # # If 'q' is pressed, stop both threads
        # if cv2.waitKey(1) == ord('q'):
        #     stop_event.set()  # Signal the audio thread to stop
        #     break

    cap.release()
    cv2.destroyAllWindows()
    K.clear_session()

    print(emotion_labels)

    # Save emotion data
    emotion_df = pd.DataFrame(emotion_data, columns=emotion_labels.values())
    emotion_df.to_csv("./ML/emotion_df.csv", index=False)

def start_audio_capture():
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    WAVE_OUTPUT_FILENAME = "./ML/file.wav"

    audio = pyaudio.PyAudio()
    print("Audio recording started")

    # Start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    while not stop_event.is_set():  # Keep recording until stop_event is set
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    print("Finished recording")

    # Stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


def scale_to_rating(value, min_value = 0, max_value = 1):
    new_min = 0
    new_max = 5

    scaled_value = ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
    return scaled_value


def start_interview():
    # Create and start threads
    realTimeVideo = Thread(target=start_realTimeVideo)
    audioCapture = Thread(target=start_audio_capture)

    print("flag1")

    realTimeVideo.start()
    print("flag2")
    audioCapture.start()
    print("flag3")

    realTimeVideo.join()
    print("flag4")
    audioCapture.join()
    print("flag4")

    print("Interview completed.")

    text, duration = audio_process.audio_text()

    no_of_words = len(text.split())

    #print("No of words:", no_of_words)
    #print("Audio duration:", duration)

    word_per_second = no_of_words / duration

    print("Words per minute:",word_per_second)

    if word_per_second > 2:
        response_score = scale_to_rating(2, min_value =0, max_value = 2)
    else:
        response_score = scale_to_rating(word_per_second, min_value =0, max_value = 2)

    emotion_df = pd.read_csv("./ML/emotion_df.csv")

    angry = emotion_df["angry"].median()
    disgust = emotion_df["disgust"].median()
    fear = emotion_df["fear"].median()
    happy = emotion_df["happy"].median()
    sad = emotion_df["sad"].median()
    surprise = emotion_df["surprise"].median()
    neutral = emotion_df["neutral"].median()

    calm = scale_to_rating(neutral)
    excitement = scale_to_rating((surprise + happy)/2)
    stress = scale_to_rating((fear + sad)/2)

    confidence = (calm + response_score + excitement + (5 - stress))/4

    print(f"Neutral: {neutral}, Surprise: {surprise}, happy: {happy}, Fear: {fear}, Sad: {sad}, word_per_second: {word_per_second}")

    print(f"Clam: {calm}\n Excitement: {excitement}\n Stress: {stress}\n Response_score: {response_score}\n Confidence Score: {confidence}")
    
start_interview()






    


