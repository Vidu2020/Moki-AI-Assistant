import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()


def speak(text):
    print(f"Bot: {text}\n")
    engine.say(text)
    engine.runAndWait()


def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You: {text}")
        return text
    except:
        print("❌ Could not understand audio")
        return ""
