import speech_recognition as sr
import wave


def audio_text():

    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Load your .wav file
    audio_file = "./ML/file.wav"

    text = str()

    with wave.open(audio_file, 'rb') as waveFile:
        frames = waveFile.getnframes()  # Number of frames in the audio file
        rate = waveFile.getframerate()  # Frame rate (samples per second)
        duration = frames / float(rate)  # Duration in seconds

    # Open the audio file
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    # Recognize the speech in the audio file using Google Web Speech API
    try:
        text = recognizer.recognize_google(audio_data)
        print("Recognized text:", text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    return text, duration
