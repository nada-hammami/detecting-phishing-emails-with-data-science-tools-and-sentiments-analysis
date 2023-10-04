import io
import os

import numpy as np
import openai
import mailparser
import sqlite3
import sklearn
import joblib
from flask import Flask, render_template, request, session, redirect, url_for
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pickle
import re
import speech_recognition as sr
import email
from Bert_Classifier import Bert_Classifier

from nltk.corpus import stopwords
from nltk import PorterStemmer, WordNetLemmatizer

from Fextraction import FeatureExtraction

app = Flask(__name__, template_folder='templates', static_folder='assets')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 Mo

app.secret_key = "secret"
emotions = ['anger', 'confusion', 'fear', 'joy', 'love', 'neutral', 'sadness', 'surprise']
# Charger le modèle pickle
with open('Data/modelemotion222.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():  # put application's code here
    return render_template('index.html')
@app.route('/service')
def service():  # put application's code here
    return render_template('services.html')
@app.route('/about')
def about():  # put application's code here
    return render_template('about.html')
@app.route('/contact')
def contact():  # put application's code here
    return render_template('contact.html')
@app.route('/home1')
def home1():
    return render_template('index1.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Récupération des données de formulaire
        username = request.form['username']
        password = request.form['password']

        # Connexion à la base de données
        conn = sqlite3.connect('database.db')
        c = conn.cursor()

        # Vérification si l'utilisateur existe déjà
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        if user:
            return "Cet utilisateur existe déjà!"

        # Ajout de l'utilisateur dans la base de données
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()

        # Enregistrement de l'utilisateur dans la session
        session['username'] = username

        return redirect(url_for('home1'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Récupération des données de formulaire
        username = request.form['username']
        password = request.form['password']

        # Connexion à la base de données
        conn = sqlite3.connect('database.db')
        c = conn.cursor()

        # Récupération de l'utilisateur de la base de données
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        if user:
            # Enregistrement de l'utilisateur dans la session
            session['username'] = username

            return redirect(url_for('home1'))
        else:
            return "Nom d'utilisateur ou mot de passe incorrect!"

    return render_template('login.html')

@app.route('/logout')
def logout():
    # Suppression de l'utilisateur de la session
    session.pop('username', None)
    return redirect(url_for('home'))

# Route pour la prédiction
# Route pour la page d'accueil
@app.route('/emotionindex')
def emotionindex():
    return render_template('emotionindex.html')
@app.route('/predictemotion', methods=['POST'])
def predictemotion():


    # Récupérer le texte à prédire depuis le formulaire
    text = request.form['text']
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]
    text = " ".join(words)
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([text])
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=300)
    emotions = ['anger', 'confusion', 'fear', 'joy', 'love', 'neutral', 'sadness', 'surprise']

    # Faire la prédiction
    prediction = model.predict([padded_sequence])[0]
    predicted_emotion = emotions[prediction.argmax()]




    # Afficher le résultat sur la page web
    return render_template('emotionresult.html', predicted_emotion=predicted_emotion, prediction=prediction, emotions=emotions, len=len)


@app.route('/predictcyber', methods=['GET', 'POST'])
def predictcyber():
    input_text = ""
    predictions = ""
    if request.method == 'POST':

        if 'text_data' in request.form:
            input_text = request.form['text_data']
        elif 'audio_data' in request.files:
            recognizer = sr.Recognizer()
            audio_data = request.files['audio_data'].read()
            audio_data = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
            input_text = recognizer.recognize_google(audio_data, language='en-US', key=None)

        model = joblib.load("bert.pkl")
        classifier = Bert_Classifier(model)

        predictions = classifier.predict([input_text])

    return render_template('cyberindex.html', input_text=input_text, predictions=predictions)

@app.route('/predictfakerealindex')
def predictfakerealindex():
    return render_template('indexfake.html')
@app.route('/predictfakereal', methods=['POST'])
def predictfakereal():
    # Load vectorizer and model
    model = joblib.load('Data/naive_bayes.pkl')
    vectorizer = joblib.load('Data/vectorizer.pkl')

    # Check if the request is made with email file or text input
    if 'eml_file' in request.files:
        # Read the uploaded file
        eml_file = request.files['eml_file']
        with io.StringIO(eml_file.read().decode("utf-8", errors="ignore")) as f:
            message = email.message_from_file(f)

        # Extract the payload and convert it to a string
        payload = message.get_payload()
        if isinstance(payload, list):
            payload = '\n'.join(map(str, payload))
        else:
            payload = str(payload)

        # Preprocess the body text
        body = payload.lower()
    else:
        # Get the text from the text input
        body = request.form['text_data'].lower()

    # Vectorize the text
    body_vect = vectorizer.transform([body])

    # Make prediction
    predictions = model.predict(body_vect)
    proba_nb = model.predict_proba(body_vect)

    # Return prediction result
    return render_template("indexfake.html", predictions=predictions,proba_nb=proba_nb)

file = open("soft.pkl","rb")
gbc = pickle.load(file)
file.close()
@app.route("/mail", methods=["GET", "POST"])
def mail():
    if request.method == "POST":

        if 'eml_file' in request.files:
            # Récupérer le fichier uploadé
            eml_file = request.files['eml_file']

            # Spécifier le dossier de destination
            dossier_destination = 'test'

            # Vérifier si le dossier de destination existe, sinon le créer
            if not os.path.exists(dossier_destination):
                os.makedirs(dossier_destination)

            # Sauvegarder le fichier dans le dossier de destination
            chemin_fichier = os.path.join(dossier_destination, eml_file.filename)
            eml_file.save(chemin_fichier)

            fe = FeatureExtraction.load_mails(dossier_destination)
            obj = FeatureExtraction.extract(fe[len(fe) - 1])
            x = np.array(obj).reshape(1, -1)

            y_pred = gbc.predict(x)[0]
            # 1 is safe
            # -1 is unsafe
            y_pro_phishing = gbc.predict_proba(x)[0, 0]
            y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
            # if(y_pred ==1 ):
            pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing * 100)
            # Supprimer le fichier eml traité
            os.remove(chemin_fichier)

            # Vider le dossier de test
            for fichier in os.listdir(dossier_destination):
                chemin_fichier = os.path.join(dossier_destination, fichier)
                os.remove(chemin_fichier)

            return render_template('mail.html', xx=round(y_pro_non_phishing, 2), mail=eml_file)


    return render_template("mail.html")
from feature import ExtractionUrl
@app.route("/urldetection", methods=["GET", "POST"])
def urldetection():
    with open('url.pkl', 'rb') as f:
        souhe = pickle.load(f)
    if request.method == "POST":
        url = request.form["url"]
        obj = ExtractionUrl(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = souhe.predict(x)[0]
        # 1 is safe
        # -1 is unsafe
        y_pro_phishing = souhe.predict_proba(x)[0, 0]
        y_pro_non_phishing = souhe.predict_proba(x)[0, 1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing * 100)

        return render_template('url.html', xx=round(y_pro_non_phishing, 2), url=url)
    return render_template("url.html", xx=-1)
@app.route('/predictfakephishindex')
def predictfakephishindex():
    return render_template('rrrr.html')
@app.route('/predictfakephish', methods=['POST'])
def predictfakephish():
    # Load vectorizer and models
    model_nb = joblib.load('Data/naive_bayes.pkl')
    vectorizer_nb = joblib.load('Data/vectorizer.pkl')
    file = open("rfc.pkl","rb")
    gbc = pickle.load(file)
    file.close()
    dossier_destination = 'test'

    # Check if the request is made with email file or text input
    if 'eml_file' in request.files:
        # Read the uploaded file
        eml_file = request.files['eml_file']
        with io.StringIO(eml_file.read().decode("utf-8", errors="ignore")) as f:
            message = email.message_from_file(f)

        # Extract the payload and convert it to a string
        payload = message.get_payload()
        if isinstance(payload, list):
            payload = '\n'.join(map(str, payload))
        else:
            payload = str(payload)

        # Preprocess the body text
        body = payload.lower()
        vv=FeatureExtraction.get_URLs(body)


        # Vectorize the text
        body_vect = vectorizer_nb.transform([body])

        # Make prediction using naive bayes model
        prediction_nb = model_nb.predict(body_vect)
        proba_nb = model_nb.predict_proba(body_vect)

        # Extract features using FeatureExtraction

        if not os.path.exists(dossier_destination):
            os.makedirs(dossier_destination)
        chemin_fichier = os.path.join(dossier_destination, eml_file.filename)
        eml_file.save(chemin_fichier)
        fe = FeatureExtraction.load_mails(dossier_destination)
        obj = FeatureExtraction.extract(fe[len(fe) - 1])
        x = np.array(obj).reshape(1, -1)


        # Make prediction using gradient boosting classifier model
        prediction_gbc = gbc.predict(x)
        proba_gbc = gbc.predict_proba(x)

        obj = ExtractionUrl(vv[1])
        x = np.array(obj.getFeaturesList()).reshape(1, 30)
        with open('url.pkl', 'rb') as f:
            souhe = pickle.load(f)
        y_pred = souhe.predict(x)[0]
        # 1 is safe
        # -1 is unsafe
        y_pro_phishing = souhe.predict_proba(x)[0, 0]
        y_pro_non_phishing = souhe.predict_proba(x)[0, 1]
        # if(y_pred ==1 ):


        # Return prediction results
        return render_template("rrrr.html",
                               prediction_nb=prediction_nb[0],
                               proba_nb=vv,
                               prediction_gbc=prediction_gbc[0],
                               proba_gbc=proba_gbc[0],preddd=y_pro_non_phishing)

    else:


        # Get the text from the text input
        body = request.form['text_data'].lower()
        vv=FeatureExtraction.get_URLs(body)

        # Vectorize the text
        body_vect = vectorizer_nb.transform([body])


        # Make prediction using naive bayes model
        prediction_nb = model_nb.predict(body_vect)
        proba_nb = model_nb.predict_proba(body_vect)
        if not os.path.exists(dossier_destination):
         os.makedirs(dossier_destination)



        # Extract features using FeatureExtraction
        fe = FeatureExtraction.load_mails(dossier_destination)
        print(fe)
        obj = FeatureExtraction.extract(fe[len(fe) - 1])
        x = np.array(obj).reshape(1, -1)

        obj = ExtractionUrl(vv[1])
        x = np.array(obj.getFeaturesList()).reshape(1, 30)
        with open('url.pkl', 'rb') as f:
            souhe = pickle.load(f)
        y_pred = souhe.predict(x)[0]
        # 1 is safe
        # -1 is unsafe
        y_pro_phishing = souhe.predict_proba(x)[0, 0]
        y_pro_non_phishing = souhe.predict_proba(x)[0, 1]
        # if(y_pred ==1 ):


        # Make prediction using gradient boosting classifier model
        prediction_gbc = gbc.predict(x)
        proba_gbc = gbc.predict_proba(x)

        # Return prediction results
        return render_template("rrrr.html",
                               prediction_nb=prediction_nb[0],
                               proba_nb=vv,
                               prediction_gbc=prediction_gbc[0],
                               proba_gbc=proba_gbc[0],preddd=y_pro_non_phishing)





if __name__ == '__main__':
    app.run()
