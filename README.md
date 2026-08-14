# ragifix

Service RAG auto-porté : ingestion, suppression et interrogation de
documents via une API HTTP locale (`127.0.0.1` par défaut).

## Sommaire

- [Installation via Docker](#installation-via-docker)
- [Installation via paquet .deb](#installation-via-paquet-deb)
- [Installation en environnement de développement](#installation-en-environnement-de-développement)
- [Configuration](#configuration)
- [API](#api)
- [Exploitation](#exploitation)

---

## Installation via Docker

```bash
git clone <url-du-dépôt> ragifix && cd ragifix

cp config.example.yaml config.yaml        # puis l'adapter
cp deploy/ragifix.env.example ragifix.env # puis renseigner les secrets

docker build -f deploy/docker/Dockerfile -t ragifix .

docker run -d \
  --name ragifix \
  --network host \
  --env-file ragifix.env \
  -v "$(pwd)/config.yaml:/etc/ragifix/config.yaml:ro" \
  -v ragifix-data:/var/lib/ragifix \
  ragifix
```

`--network host` est nécessaire : `ragifix` doit rester joignable en
`127.0.0.1` par les autres services de la machine (`ragifix-collector`,
`ragifix-mcp`), et un mapping de port Docker classique ne préserve pas
cette sémantique (voir `api.host` dans `config.yaml`). Sous Windows/macOS
(Docker Desktop, pas de réseau host natif), publier explicitly le port sur
`127.0.0.1` avec `-p 127.0.0.1:8421:8421` à la place.

Vérifier que le service tourne :

```bash
curl http://127.0.0.1:8421/health
```

## Installation via paquet .deb

Le paquet se construit depuis ce dépôt (il n'est pas publié sur un dépôt
apt) :

```bash
git clone <url-du-dépôt> ragifix && cd ragifix

sudo apt-get install -y devscripts debhelper python3-venv python3-pip
dpkg-buildpackage -us -uc -b

sudo apt install -y ../ragifix_0.1.0-1_all.deb
```

L'installation (script `postinst`) construit un environnement virtuel
Python dans `/opt/ragifix/venv` et y installe les dépendances **depuis
PyPI** : **un accès réseau est nécessaire au moment de `apt install`**
(les dépendances sont volumineuses — docling embarque PyTorch — l'opération
peut prendre plusieurs minutes). Ensuite, le service fonctionne
entièrement hors-ligne.

Le paquet crée :
- un utilisateur système dédié `ragifix`
- `/etc/ragifix/config.yaml` et `/etc/ragifix/ragifix.env`, pré-remplis
  depuis les fichiers d'exemple (à adapter avant de démarrer)
- une unité systemd `ragifix.service` (activée mais pas démarrée)

```bash
sudo nano /etc/ragifix/config.yaml       # adapter la configuration
sudo nano /etc/ragifix/ragifix.env       # renseigner les secrets
sudo systemctl enable --now ragifix
sudo systemctl status ragifix
journalctl -u ragifix -f
```

Désinstallation :

```bash
sudo apt remove ragifix     # conserve /etc/ragifix et /var/lib/ragifix
sudo apt purge ragifix      # supprime tout, y compris la configuration
```

## Installation en environnement de développement

```bash
git clone <url-du-dépôt> ragifix && cd ragifix

python3 -m venv venv
source venv/bin/activate
pip install -e .

cp config.example.yaml config.yaml   # reste à côté du code, hors /etc

export RAGIFIX_API_TOKEN=dev-token
# export EMBEDDING_API_KEY=...       # si embedding.backend=openai_compatible

ragifix --config ./config.yaml
```

## Configuration

Un seul fichier YAML (voir `config.example.yaml` pour la référence
complète et commentée). Aucun secret n'y est jamais stocké en clair :
chaque secret est référencé via une variable d'environnement (champs
`*_env`).

| Variable d'environnement | Rôle |
|---|---|
| `RAGIFIX_API_TOKEN` | Token bearer requis sur toutes les routes de l'API sauf `/health`. |
| `EMBEDDING_API_KEY` | Clé de l'API d'embedding distante (si `embedding.backend: openai_compatible`). |

Points de configuration à connaître :
- `api.host` doit rester `127.0.0.1` (validé au chargement, refusé sinon).
- `vectorstore.milvus.mode: lite` (par défaut) : **un seul process
  `ragifix` à la fois** ne doit ouvrir ce fichier. Ne jamais lancer deux
  instances (ni plusieurs workers) pointant vers le même `lite_path`. Pour
  plusieurs clients concurrents, utiliser `mode: server` avec un vrai
  serveur Milvus (`host`/`port`).
- `embedding.backend: fastembed` télécharge le modèle au premier usage
  (mis en cache ensuite). En environnement sans accès sortant à
  `huggingface.co`, pré-peupler ce cache avant la mise en prod.
- Le chunking (`chunking.*`) utilise `tiktoken`, qui télécharge son
  fichier d'encodage au premier usage depuis
  `openaipublic.blob.core.windows.net` (mis en cache ensuite via
  `TIKTOKEN_CACHE_DIR`) — même remarque en environnement restreint.

## API

Toutes les routes sauf `/health` nécessitent l'en-tête
`Authorization: Bearer <RAGIFIX_API_TOKEN>`.

| Méthode | Route | Description |
|---|---|---|
| `PUT` | `/documents/{doc_id}?extension=...&metadata=...` | Ajoute ou met à jour un document. Corps de requête = contenu brut (`application/octet-stream`), jamais de multipart. |
| `DELETE` | `/documents/{doc_id}` | Supprime un document (204, ou 404 s'il n'existait pas). |
| `GET` | `/documents/{doc_id}` | Détail d'un document indexé. |
| `GET` | `/documents?prefix=...` | Liste les documents indexés. |
| `POST` | `/query` | `{"query": "...", "top_k": 5, "filters": {...}}` → chunks pertinents. |
| `GET` | `/health` | Sans authentification. |

`doc_id` accepte `/` et `:`.

Exemple :

```bash
curl -X PUT "http://127.0.0.1:8421/documents/notes:readme.md?extension=md" \
  -H "Authorization: Bearer $RAGIFIX_API_TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @README.md

curl -X POST http://127.0.0.1:8421/query \
  -H "Authorization: Bearer $RAGIFIX_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "comment installer ragifix ?", "top_k": 3}'
```

## Exploitation

- Logs : `journalctl -u ragifix -f` (systemd) ou `docker logs -f ragifix`.
- Sauvegarde : `/var/lib/ragifix` (registre SQLite + base Milvus Lite si
  `mode: lite`) — arrêter le service avant toute copie à froid.
- Mise à jour de version : reconstruire/réinstaller le paquet ou l'image ;
  le registre et la base vectorielle sont conservés (volume/`/var/lib`
  inchangés).
- Changement de modèle d'embedding : les vecteurs déjà indexés ne sont
  plus comparables aux nouvelles requêtes. Ré-appeler `PUT
  /documents/{doc_id}` pour chaque document (idéalement depuis un
  environnement de pré-production dédié) après un tel changement.
