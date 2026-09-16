# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Empêcher Python d'écrire des fichiers .pyc et activer le flush stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Dépendances système minimales (libGL et libglib pour le traitement d'image OpenCV/PIL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .

# Installation de PyTorch CPU (image légère et universelle) puis des autres paquets
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code source et des artefacts nécessaires
COPY app/ ./app/
COPY notebooks/best_resnet18.pt ./notebooks/best_resnet18.pt

# Exposition du port par défaut de Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Lancement de l'interface Streamlit
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
