FROM python:3.10.8-slim

# Establecer un directorio de trabajo
WORKDIR /app

# Copiar los requisitos
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación
COPY app.py ./

# Copiar el directorio de imágenes estáticas y otros directorios necesarios
COPY statics/ ./statics/
COPY models/ ./models/
COPY pages/ ./pages/

# Exponer el puerto 8080
EXPOSE 8080

# Comando por defecto
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]


