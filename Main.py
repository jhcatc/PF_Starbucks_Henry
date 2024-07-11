import streamlit as st

# streamlit run Main.py

# Configurar título de la página y favicon
st.set_page_config(
    page_title="[ProData] Consulting",
    page_icon=":teacup_without_handle:",
    layout="wide"
)

# Definir las rutas de las imágenes
image_path1 = "statics//pdc.png"

# Mostrar las imágenes en dos columnas
col1, col2 = st.columns(2)  # Dividir la fila en dos columnas

# Mostrar la primera imagen en la primera columna
col1.image(image_path1, width=600)

st.markdown('## Project Title: Machine Learning Project for Business Performance Optimization: Starbucks &reg;')  # Título del Dash
st.markdown('#### Client: Starbucks Corporation')
st.markdown('#### Date: July 19, 2024')
st.markdown('#### Project Team: Maria Teresa Laura, Alan Zimmermann, Conrado Torino, Maxi Lucchesi, Jhonny Catari')
st.markdown('___')
st.write("<p>We offer innovative and customized solutions to optimize operations and maximize revenue.</p>", unsafe_allow_html=True)
st.write("<p>We combine advanced data analytics and industry knowledge to provide strategic insights and tangible results.</p>", unsafe_allow_html=True)
st.write("<p>This application utilizes advanced Machine Learning techniques to provide real-time analysis of customer reviews for Starbucks. Historical reviews have been used to train the ML models. Through predictive modeling and data analysis, we enhance Starbucks' ability to forecast market trends, optimize operations, and improve customer satisfaction. Our findings and recommendations aim to strengthen Starbucks' leadership in the coffee industry by identifying key areas of opportunity and efficiency.</p>", unsafe_allow_html=True)