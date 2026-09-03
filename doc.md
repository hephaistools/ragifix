# ragifix — Service RAG principal

## Vue d'ensemble

**ragifix** est le cœur du système. C'est un service FastAPI qui expose une API HTTP pour :
- Indexer des documents (parsing, chunking, embedding, écriture Milvus)
- Rechercher des passages pertinents via embedding sémantique
- Gérer les métadonnées des documents indexés

## Architecture interne

```
┌─────────────────────────────────────────────────────────┐
│                    ragifix (FastAPI)                     │
├─────────────────────────────────────────────────────────┤
│  routes.py          │  API HTTP (PUT, GET, DELETE, POST) │
├─────────────────────────────────────────────────────────┤
│  service.py         │  Orchestration métier              │
│                     │  - Chunking                        │
│                     │  - Embedding                       │
│                     │  - Upsert/Cleanup Milvus           │
├─────────────────────────────────────────────────────────┤
│  vectorstore/       │  Backend vectoriel                 │
│  └─ milvus_backend  │  - Milvus Lite (default)           │
│                     │  - Milvus Server (optionnel)       │
├─────────────────────────────────────────────────────────┤
│  embedding/         │  Backends d'embedding              │
│  ├─ fastembed       │  - Modèle local (BAAI/bge-small)  │
│  └─ openai_compat   │  - API OpenAI compatible           │
├─────────────────────────────────────────────────────────┤
│  registry.py        │  Registre SQLite                   │
│                     │  - doc_id → chunk_ids              │
│                     │  - Métadonnées (extension, date)   │
│                     │  - Sources (nom, description)      │
└─────────────────────────────────────────────────────────┘
```

## Points clés

### 1. Sérialisation des accès Milvus

Milvus Lite (mode par défaut) ne supporte **qu'un seul processus** ouvrant le fichier. Pour garantir cette contrainte :

- `RagifixService` utilise un `ThreadPoolExecutor(max_workers=1)`
- Toutes les opérations Milvus passent par ce thread unique
- C'est une défense en profondeur, pas la protection principale (qui est "un seul process ragifix")

### 2. Upsert idempotent

Au lieu de `delete + insert` (qui laisse une fenêtre où le document n'existe plus), ragifix utilise un **upsert** :

- `chunk_id` est déterministe : `sha256(doc_id + index)[:32]`
- Même document → mêmes chunk_ids → upsert plutôt qu'insert
- Pas de fenêtre de non-disponibilité

### 3. Cleanup des chunks orphelins

Quand un document est mis à jour :
1. Récupérer les anciens chunk_ids depuis le registre
2. Comparer avec les nouveaux chunk_ids
3. Supprimer les chunks orphelins (anciens non réutilisés)

Sans cette étape, chaque mise à jour créerait des chunks orphelins qui polluent la base.

### 4. Registre SQLite

Le registre stocke :
- `doc_id` → `chunk_ids`, `extension`, `metadata`, `updated_at`
- `sources` → `name`, `description`, `enabled`, `updated_at`

Usage principal : cleanup des chunks orphelins lors des mises à jour.

Convention `metadata.origin` (optionnelle, posée par le producteur du
document — typiquement ragifix-collector) : `{"kind": "https"|"file", "uri":
"...", "label": "..."}`, le lien ou chemin le plus rapide vers le document
source (ex: `webUrl` SharePoint, chemin local). ragifix la remonte telle
quelle, typée, en champ `origin` sur `DocumentResponse` et sur chaque
résultat de `POST /query` (`null` si absente ou invalide) — sans quoi elle
resterait invisible dans le blob `metadata` générique.

### 5. API HTTP

| Méthode | Route | Description |
|---------|-------|-------------|
| `PUT` | `/documents/{doc_id}` | Indexer/mettre à jour un document |
| `DELETE` | `/documents/{doc_id}` | Supprimer un document |
| `GET` | `/documents/{doc_id}` | Détail d'un document |
| `GET` | `/documents` | Liste des documents (avec prefix) |
| `POST` | `/query` | Recherche sémantique |
| `GET` | `/sources` | Lister les sources |
| `POST` | `/sources` | Mettre à jour les sources |
| `GET` | `/health` | Health check (pas d'auth) |

### 6. Authentification

- Token bearer via `Authorization: Bearer <token>`
- Comparaison en temps constant (`hmac.compare_digest`)
- Pas de distinction read/write pour le moment (TODO)

### 7. Configuration

| Champ | Description |
|-------|-------------|
| `vectorstore.milvus.mode` | `lite` (fichier local) ou `server` (Milvus distant) |
| `vectorstore.milvus.lite_path` | Chemin du fichier Milvus Lite |
| `embedding.backend` | `fastembed` (local) ou `openai_compatible` (API) |
| `embedding.fastembed.model` | Modèle à utiliser (ex: `BAAI/bge-small-en-v1.5`) |
| `registry.sqlite.path` | Chemin du registre SQLite |

---

## Déploiement

### Docker
```bash
docker build -f deploy/docker/Dockerfile -t ragifix .
docker run -d --name ragifix --network host -v ragifix-data:/var/lib/ragifix ragifix
```

### Développement
```bash
python3 -m venv venv && source venv/bin/activate
pip install -e .
ragifix --config ./config.yaml
```

---

## Limitations connues

- Milvus Lite : un seul process à la fois
- Pas de filtrage par source dans `POST /query` (TODO)
- Pas de pagination (TODO)
- Registry en SQLite (pas de backend distant pour le moment)
