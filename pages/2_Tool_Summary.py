import streamlit as st

# Configurar la página
st.set_page_config(
    page_title="Machine Learning Project",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título
html_temp = """
<h2 style="color:#006847;text-align:center;">Description of the Models and tools used</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Normalized Text
st.markdown('#### Normalized Text')
st.write("<p>This code preprocesses text using NLTK in Python:</p>", unsafe_allow_html=True)
st.write("""
- Technologies: NLTK (Natural Language Toolkit).
- Libraries: nltk for tokenization and stopwords.
""")
st.write("<p>Function process_text:</p>", unsafe_allow_html=True)
st.write("""
- Input: Text string.
- Processes: Converts to lowercase, tokenizes, removes stopwords, capitalizes words, and joins tokens.
- Output: Processed text string.
""")

# Manipulated Review
st.markdown('#### Manipulated Review')
st.write("<p>Predicts whether a review is genuine or manipulated.</p>", unsafe_allow_html=True)
st.write("<p>Technologies and Libraries:</p>", unsafe_allow_html=True)
st.write("""
- Libraries: numpy, machine learning model (loaded_model_MANIPULATED), TF-IDF vectorizer (loaded_TFIDF_MANIPULATED).
""")
st.write("<p>How It Works:</p>", unsafe_allow_html=True)
st.write("""
- Input: text (review), stars (rating).
- Process: Transforms text with TF-IDF, combines with rating, and predicts with the model.
- Output: 'False' if manipulated, 'Genuine' if genuine.
""")

# Sentimental Analysis
st.markdown('#### Sentimental Analysis')
st.write("<p>This code predicts the sentiment of a text and displays a corresponding emoji. It uses a machine learning model for prediction and streamlit for the interface.</p>", unsafe_allow_html=True)
st.write("<p>Input: Processed text.</p>", unsafe_allow_html=True)
st.write("<p>Output: Emoji representing the sentiment (negative, neutral, or positive).</p>", unsafe_allow_html=True)
st.write("<p>Technology and Libraries:</p>", unsafe_allow_html=True)
st.write("""
- Libraries: Preloaded model (loaded_model_SENTIMENT), streamlit.
""")

# Business Rating based on Review
st.markdown('#### Business Rating based on Review')
st.write("<p>This code defines the function predict_rating_VECTORIZER, which predicts a business's rating from review text and provides recommendations based on the prediction.</p>", unsafe_allow_html=True)
st.write("<p>Input:</p>", unsafe_allow_html=True)
st.write("""
- text: The review text.
- stars: The current rating of the business.
""")
st.write("<p>Output:</p>", unsafe_allow_html=True)
st.write("""
- Streamlit for displaying results.
- CountVectorizer to convert text into count vectors.
- RandomForestRegressor for rating prediction.
- Pipeline to chain the vectorizer and model.
- train_test_split to split data into training and testing sets.
""")
st.write("<p>Process:</p>", unsafe_allow_html=True)
st.write("""
- Trains a model using review text and ratings.
- Uses a pipeline to predict the rating and display recommendations based on the result.
""")

# Classification and Labeling of Reviews
st.markdown('#### Classification and Labeling of Reviews')
st.write("<p>Classify and tag Starbucks reviews into five categories and visualize the results.</p>", unsafe_allow_html=True)
st.write("<p>Inputs: Text of a customer review.</p>", unsafe_allow_html=True)
st.write("<p>Technology and Libraries Used:</p>", unsafe_allow_html=True)
st.write("""
- Technology: Groq API with the LLaMA model for classification and tagging.
- Libraries: Groq (API), pandas (data manipulation), matplotlib.pyplot (visualization), streamlit (web interface).
""")
st.write("<p>Process:</p>", unsafe_allow_html=True)
st.write("""
- cluster_and_tag(review): Sends the review to the API for classification into categories and sentiment tagging.
- cap(el): Capitalizes text for uniformity.
- analyze_review(new_text, figsize=(10, 4)): Analyzes the review, aggregates and counts results, and creates a stacked bar chart.
""")
st.write("<p>Outputs:</p>", unsafe_allow_html=True)
st.write("""
- DataFrame with frequencies of categories and tags.
- Bar chart showing the distribution of tags by category.
""")