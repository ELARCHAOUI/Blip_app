# Utiliser une image de base Debian récente pour Python 3.13
FROM python:3.11-slim

# Mettre à jour et installer FFmpeg (sera installé dans la couche d'image,
# évitant les problèmes de lecture seule au moment de l'exécution/du build Render)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail pour l'application
WORKDIR /app

# Copier les dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier votre code principal (votre app.py)
COPY app.py .

# Exposer le port par défaut de Gradio
EXPOSE 7860

# Commande de démarrage
CMD ["python", "app.py"]
