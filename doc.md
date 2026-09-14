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
│                     │  - doc_id → chunk_ids, updated_at  │
│                     │  - Sources (nom, description)      │
└─────────────────────────────────────────────────────────┘
```

`metadata` (extension incluse) ne vit **que** dans Milvus, dupliquée par
chunk — voir point 4 ci-dessous.

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

### 4. Où vit quelle donnée : registre SQLite vs Milvus

Le registre SQLite ne stocke que ce qui sert au cleanup des chunks
orphelins et au listing des doc_id connus :
- `documents` : `doc_id` → `chunk_ids`, `updated_at`
- `sources` : `name`, `description`, `enabled`, `updated_at`

Tout le reste (`metadata`, `extension` incluse) ne vit que dans Milvus,
dupliqué sur chaque chunk du document. Pour répondre à `GET /documents` et
`GET /documents/{doc_id}` sans recherche vectorielle, `RagifixService`
combine la liste des `doc_id` du registre avec un point-lookup Milvus sur
le chunk d'index 0 de chaque document (`VectorStore.get_document_metadata`,
`chunk_id` étant déterministe — voir point 2).

`metadata` est un objet JSON à plat, formalisé côté producteur (voir
[`ragifix-collector/doc.md`](../ragifix-collector/doc.md) pour la liste des
clés). Seule contrainte imposée par `ragifix` : la clé `extension`
(extension du fichier, sans le point) est **obligatoire** — `PUT
/documents/{doc_id}` répond `400` si elle est absente. Le reste est libre
et renvoyé tel quel dans `DocumentResponse.metadata` et dans chaque
résultat de `POST /query`.

### 4bis. Filtres sur `POST /query` et `GET /documents`

`filters` (clés `source`, `extension`, `filename_glob`, `modified_after`,
`modified_before`, combinées en `AND` ; `source`/`extension`/`filename_glob`
acceptent une valeur unique ou une liste — `OR` intra-clé) est normalisé et
appliqué par `ragifix.vectorstore.base.matches_filters`, utilisé à deux
endroits :
- `RagifixService._list_documents_sync` : filtrage entièrement en Python
  (la metadata de tous les documents est déjà chargée pour construire la
  réponse).
- `MilvusVectorStore.search` : `source`/`extension`/`modified_after`/
  `modified_before` sont poussés dans l'expression de filtre Milvus
  (`source`/`extension` lus depuis le champ `metadata` de type `JSON` ;
  `modified_at` est **dupliqué en champ scalaire `VARCHAR`** au niveau de ce
  backend uniquement — un champ JSON ne permet pas de comparaisons
  `>=`/`<=` fiables. Ce détail est invisible du reste du programme :
  `VectorStore.get_document_metadata` ne renvoie jamais ce champ dupliqué).
  `filename_glob` n'est pas exprimable en filtre Milvus fiable : la
  recherche sur-échantillonne (`min(top_k * 5, 200)`) puis post-filtre en
  Python via `matches_filters`, ce qui peut renvoyer moins de `top_k`
  résultats si le(s) motif(s) sont très restrictifs.

### 5. API HTTP

| Méthode | Route | Description |
|---------|-------|-------------|
| `PUT` | `/documents/{doc_id}` | Indexer/mettre à jour un document (`metadata` doit inclure `extension`) |
| `DELETE` | `/documents/{doc_id}` | Supprimer un document |
| `GET` | `/documents/{doc_id}` | Détail d'un document |
| `GET` | `/documents` | Liste des documents, filtrable (voir point 4bis) — sans pagination pour le moment |
| `POST` | `/query` | Recherche sémantique, filtrable (voir point 4bis) |
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
- Pas de pagination sur `GET /documents` (TODO)
- Registry en SQLite (pas de backend distant pour le moment)
- `filters.filename_glob` sur `POST /query` peut renvoyer moins de `top_k` résultats (sur-échantillonnage + post-filtre Python, voir point 4bis)
