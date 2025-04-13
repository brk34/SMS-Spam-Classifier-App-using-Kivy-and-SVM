import numpy as np
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
import pickle
import string
from nltk.corpus import stopwords
import nltk
import re
import sklearn
from nltk.stem import SnowballStemmer

print(nltk.data.path)
nltk.data.path.append('/Users/berk/nltk_data')
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))



class SMSClassifierApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tfidf = pickle.load(open('vectorizer2.pkl', 'rb'))
        self.model = pickle.load(open('svm2.pkl', 'rb'))

    def build(self):
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        with layout.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(1, 1, 1, 1)  # White
            self.rect = Rectangle(size=layout.size, pos=layout.pos)

        layout.bind(size=self._update_rect, pos=self._update_rect)

        label = Label(text="SMS Spam Classifier", font_size=24, color=(0.5, 0.5, 0.5, 1.0))

        self.input_sms = TextInput(font_size=16, size_hint=(1, None), height=60)
        self.input_sms.background_color = get_color_from_hex('#FFFFFF')  # Change the background color of the TextInput

        predict_button = Button(text="Predict", font_size=18, size_hint=(1, None), height=40)
        predict_button.background_color = get_color_from_hex('#00FFCC')  # Change the background color of the button to blue-green
        predict_button.color = get_color_from_hex('#FFFFFF')  # Change the text color of the button to white

        layout.add_widget(label)
        layout.add_widget(self.input_sms)
        layout.add_widget(predict_button)

        predict_button.bind(on_press=self.predict)
        
        self.result_label = Label(font_size=18, color=(0.5, 0.5, 0.5, 1.0))

        layout.add_widget(self.result_label)
        return layout

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def stemmer(self, text):
        text = text.split()
        words = ""
        for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i)) + " "
        return words

    def preprocess_data(self, text):
        # Clean puntuation, urls, and so on
        text = self.clean_text(text)
        # Remove stopwords
        text = ' '.join(word for word in text.split(' ') if word not in stopwords.words('english'))
        # Stemm all the words in the sentence
        text = self.stemmer(text)

        return text

    def count_chars(self, text):
        return len(text)

    def count_punctuations(self, text):
        punctuations = '#$%&)(*+-/:;<=>?@[\]^_`}|{~'
        d = 0
        for i in punctuations:
            if i in text:
                d = d + 1
        return d

    def num_val(self, text):
        d = 0
        if any(chr.isdigit() for chr in text):
            d = d + 1
        return d

    def exist_links(self, text):
        d = 0
        if 'http' in text or 'https' in text or 'www' in text or 'bit.ly/S' in text:
            d = d + 1
        return d

    def count_unique_words(self, text):
        return len(set(text.split()))

    def predict(self, instance):
        input_text = self.input_sms.text
        tfidf = pickle.load(open('vectorizer2.pkl', 'rb'))
        model = pickle.load(open('svm2.pkl', 'rb'))
        # Prétraitement du texte d'entrée
        transformed_sms = self.preprocess_data(input_text)
        char_count = self.count_chars(input_text)
        punct_count = self.count_punctuations(input_text)
        has_number = self.num_val(input_text)
        has_link = self.exist_links(input_text)
        unique_words = self.count_unique_words(input_text)

        other_featurs = [char_count, punct_count, has_number, has_link, unique_words]

        # Vectorisation
        vector_input = self.tfidf.transform([transformed_sms]).toarray()
        allfeaturestest = np.concatenate((vector_input, np.array([other_featurs])), axis=1)

        # Prédiction
        result = self.model.predict(allfeaturestest)[0]

        # Affichage du résultat
        result_label = self.result_label
        if result == 'spam':
            result_label.text = "Spam"
            result_label.color = get_color_from_hex('#FF0000')  
        else:
            result_label.text = "Not Spam"
            result_label.color = get_color_from_hex('#00FF00')  




if __name__ == '__main__':
    Window.size = (400, 300)
    SMSClassifierApp().run()
