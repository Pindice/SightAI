# Utilisez une image de base Python
FROM python:3.10.11

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez les fichiers du projet dans le conteneur
COPY . /app

# Installez les dépendances du projet
RUN pip install --no-cache-dir -r requirements.txt

# Installez la bibliothèque libgl1-mesa-glx
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Exposez le port utilisé par Streamlit
EXPOSE 8501

# Démarrez l'application Streamlit lorsque le conteneur démarre
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]
