# Utilisez une image de base Python
FROM python:3.8

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez les fichiers du projet dans le conteneur
COPY . /app

# Installez les dépendances du projet
RUN pip install -r requirements.txt

# Exposez le port utilisé par Streamlit
EXPOSE 8501

# Démarrez l'application Streamlit lorsque le conteneur démarre
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]
