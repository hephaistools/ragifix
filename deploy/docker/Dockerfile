# A lancer depuis la racine de ce dépôt :
#   docker build -f deploy/docker/Dockerfile -t ragifix .
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system ragifix && useradd --system --gid ragifix --create-home --home-dir /home/ragifix ragifix

WORKDIR /opt/ragifix

COPY . .

RUN pip install --no-cache-dir .

RUN mkdir -p /var/lib/ragifix && chown -R ragifix:ragifix /var/lib/ragifix /opt/ragifix

USER ragifix

# Optionnel mais recommandé en environnement sans accès sortant à
# openaipublic.blob.core.windows.net / huggingface.co : pré-peupler les
# caches (tiktoken, fastembed) à la construction de l'image plutôt qu'au
# premier run — voir le README.
# ENV TIKTOKEN_CACHE_DIR=/opt/ragifix/.tiktoken_cache

EXPOSE 8421

# IMPORTANT : un seul container par fichier Milvus Lite. Ne jamais monter
# le même volume /var/lib/ragifix depuis deux containers ragifix lancés
# simultanément (voir README).
ENTRYPOINT ["ragifix"]
CMD ["--config", "/etc/ragifix/config.yaml"]
