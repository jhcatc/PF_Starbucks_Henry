import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib
from joblib import load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st
import nltk
import ast
from groq import Groq 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.exceptions import NotFittedError

# Configurar la página
st.set_page_config(
    page_title="Machine Learning Project",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título
html_temp = """
<h2 style="color:#006847;text-align:center;">Machine Learning Project for Business Performance Optimization</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# CSS para estilizar el botón
button_style = """
<style>
.custom-button {
    background-color: #FFFFFF;
    color: #000000;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 12px;
    font-weight: bold;
}
</style>
"""
# Renderiza el CSS
st.markdown(button_style, unsafe_allow_html=True)

# Path de los modelos preentrenados
MANIPULATED_PATH = 'models/manipulated_reviews.pkl'
SENTIMENT_PATH = 'models/sentiment_analysis.pkl'
VECTORIZER_PATH = 'models/rating_vectorizer.pkl'
CLUSTER_PATH = 'models/cluster_analysis.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
TFIDF_PATH = 'models/tfidf_vectorizer.pkl'

# Cargar los modelos guardados
loaded_model_MANIPULATED = joblib.load(MANIPULATED_PATH)
loaded_model_SENTIMENT = joblib.load(SENTIMENT_PATH)
pipeline_VECTORIZER = joblib.load(VECTORIZER_PATH)
loaded_TFIDF_MANIPULATED = joblib.load(TFIDF_PATH)
api_key = "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"

# TOKENIZADO ______________________________________________________________________________________

# Verificar si los paquetes 'punkt' y 'stopwords' ya están descargados
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Definir la función para normalizar, tokenizar y capitalizar el texto
def process_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Tokenizar el texto
    tokens = word_tokenize(text)
    # Remover palabras vacías
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    # Capitalizar cada palabra
    tokens = [word.capitalize() for word in tokens]
    # Unir tokens en una cadena de texto
    processed_text = ' '.join(tokens)
    return processed_text    

# MANIPULATED_PATH _________________________________________________________________________________

def predict_review(text, stars):
    # Transformar el texto usando el vectorizador TF-IDF
    text_tfidf = loaded_TFIDF_MANIPULATED.transform([text])

    # Combinar las características de texto con la columna stars
    combined_features = np.hstack((text_tfidf.toarray(), [[stars]]))

    # Hacer una predicción
    prediction = loaded_model_MANIPULATED.predict(combined_features)

    # Devolver el resultado
    return 'False' if prediction[0] == 1 else 'Genuine'

# SENTIMENT_PATH ___________________________________________________________________________________

# Definir un diccionario que mapea los valores de sentimiento a emojis
sentiment_to_emoji = {
    0: "😞",  # negativo
    1: "😐",  # neutral
    2: "😊"   # positivo
}

# Función para predecir y devolver el emoji correspondiente
def predict_sentiment_SENTIMENT(text):
    prediction = loaded_model_SENTIMENT.predict([text])[0]
    return sentiment_to_emoji[prediction]

# VECTORIZER_PATH __________________________________________________________________________________

# Definir la función para predecir el rating del negocio
def predict_rating_VECTORIZER(text, stars):
    # Hacer una predicción del rating del negocio
    predicted_rating = pipeline_VECTORIZER.predict([text])[0]

    # Mostrar el resultado
    st.write(f'The predicted rating of the business based on the review is: {predicted_rating:.2f}')

    # Hacer recomendaciones según el rating predicho
    if predicted_rating >= 4.5:
        st.write("Business recommendation: Business Rating: Excellent service! Keep focused on maintaining high quality.")
    elif 3.5 <= predicted_rating < 4.5:
        st.write("Business recommendation: Here are some areas for improvement based on the reviews received:")
        # Puedes imprimir recomendaciones específicas basadas en el análisis de las revisiones aquí.
    else:
        st.write("Business recommendation:  We have identified critical areas that require urgent attention:")
        # Puedes imprimir recomendaciones específicas para abordar los problemas identificados aquí.

# CLUSTER_PATH _____________________________________________________________________________________

def cluster_and_tag(review):
    client = Groq(api_key=api_key)
    
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are a business specialist. You have to classify Starbucks client reviews into one of five clusters: \n1) Service.\n2) Place.\n3) Coffee.\n4) Food.\n5) Time\n\nAdditionally, add a positive, neutral or negative tag to the review.\n\nReturn your answer in the specified format, without any other messages."
            },
            {
                "role": "user",
                "content": "Review: The location is a franchise of sorts operated by Sodexo, the food service intermediary working on contract for the university. As a result, the service is not and will never be up to Starbucks's standards. This is less about the quality of the food & beverages, and more to do with the quality and friendliness of the servers. I personally have only had to deal with slow service, but that could happen at any time during a rush. The worst stories come from others; I have been around while the building was hosting an added-security event, which required the presence of LEOs; the baristas refused to provide him with a glass of water without charging him. At other foodservice locations in the same building and operated by the same company, not even students are charged for water glasses. That one event does a good job summarizing the apathy and disregard the employees have for their customers. Another critical issue is that this location doesn't accept Starbucks's own rewards program, which is extremely annoying after having signed up for the program just for the convenience of the location being in the same building that I work for. I've personally made a resolution to never purchase another item at this location again."
            },
            {
                "role": "assistant",
                "content": "{'cluster': 'Service', 'tag': 'negative'}"
            },
            {
                "role": "user",
                "content": "Go to the one in Sterne. This place is a mess. Wrong size coffee, stale croissant, long wait."
            },
            {
                "role": "assistant",
                "content": "[{'cluster': 'place', 'tag': 'negative'},{'cluster': 'food', 'tag': 'negative'}]"
            },
            {
                "role": "user",
                "content": f"Review: {review}"
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    answer = []
    for chunk in completion:
        sent = (chunk.choices[0].delta.content or "")
        answer.append(sent)

    response = "".join(answer).strip()
    
    # Intenta evaluar la respuesta, si falla, muestra el error
    try:
        result = ast.literal_eval(response)
        if isinstance(result, dict):
            result = [result]
        return pd.DataFrame(result)
    except Exception as e:
        print(f"Error al evaluar la respuesta: {e}")
        print("Respuesta recibida:", response)
        return pd.DataFrame()

# INTERFACE DE STREAMLIT _________________________________________________________________________

# Configuración de la barra lateral
with st.sidebar:
    st.title("Machine Learning Models")
    st.write("Select a model to use")

    # Selector de modelos
    model_option = st.selectbox(
        "Choose a model:",
        ["Sentiment Analysis", "Fake Review Detection", "Business Rating Prediction", "Cluster and Tag"]
    )

# Input del usuario
st.subheader("")
review_text = st.text_area("", "")

# Variables de estado de sesión
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = ""

if "predict_sentiment" not in st.session_state:
    st.session_state.predict_sentiment = ""

if "predicted_rating" not in st.session_state:
    st.session_state.predicted_rating = ""

if "cluster_and_tag_result" not in st.session_state:
    st.session_state.cluster_and_tag_result = ""

if model_option == "Sentiment Analysis":    
    if st.button("Analyze Sentiment", key="sentiment"):
        if review_text:
            st.session_state.predict_sentiment = predict_sentiment_SENTIMENT(review_text)
            st.write(f"Sentiment Analysis Result: {st.session_state.predict_sentiment}")
        else:
            st.write("Please enter a review text.")

elif model_option == "Fake Review Detection":
    stars = st.slider("Enter the star rating (1-5)", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
    if st.button("Detect Fake Review", key="fake_review"):
        if review_text:
            st.session_state.prediction_result = predict_review(review_text, stars)
            st.write(f"Fake Review Detection Result: {st.session_state.prediction_result}")
        else:
            st.write("Please enter a review text.")

elif model_option == "Business Rating Prediction":
    stars = st.slider("Enter the star rating (1-5)", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
    if st.button("Predict Rating", key="rating_prediction"):
        if review_text:
            predict_rating_VECTORIZER(review_text, stars)
        else:
            st.write("Please enter a review text.")

elif model_option == "Cluster and Tag":
    if st.button("Cluster and Tag Review", key="cluster_and_tag"):
        if review_text:
            result_df = cluster_and_tag(review_text)
            st.session_state.cluster_and_tag_result = result_df

            # Mostrar el DataFrame en Streamlit
            st.write("Cluster and Tag Result:")
            st.dataframe(result_df)
        else:
            st.write("Please enter a review text.")
