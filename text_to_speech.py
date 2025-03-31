import pyttsx3

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Adjust speech rate
        self.engine.setProperty('volume', 1.0)  # Set volume to max

    def speak(self, text):
        """
        Converts given text into speech.

        :param text: String text to be spoken
        """
        if text.strip():  # Only speak if there is text
            self.engine.say(text)
            self.engine.runAndWait()
