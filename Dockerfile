# Image de base legere avec Python 3.10
FROM python:3.10-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# On copie d'abord uniquement requirements.txt pour profiter du cache Docker
# (si le code change mais pas les dependances, Docker ne les reinstalle pas)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# On copie le reste du projet (app/, model/, scraping/)
COPY app/ ./app/
COPY model/ ./model/
COPY scraping/ ./scraping/

# Port par defaut de Streamlit
EXPOSE 8501

# Verifie que l'app repond bien (utile pour docker-compose / orchestrateurs)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Commande de lancement
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
