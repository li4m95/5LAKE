# modern-mlops — pipeline MLOps de prédiction de prix immobiliers

Projet SUPINFO 5LAKE : prédire le prix de vente d'une maison (jeu de données Ames Housing) et
exploiter ce modèle comme une équipe réelle : données validées, expériences tracées, registre de
modèles avec règles de promotion, API sécurisée, supervision, alertes et tableaux de bord.

**Outils** : PySpark, Feast, MLflow, Dagster, FastAPI, Postgres, Evidently, Prometheus, Grafana,
Docker Compose, uv.

## Architecture en un coup d'œil

```
OpenML ─► CSV brut ─► PySpark (validation + préparation) ─► Feast ─► entraînement + gate ─► MLflow
                                  (orchestré par Dagster)                                     │
                                                                               alias "champion"
client ──(X-API-Key)──► API FastAPI  /predict  /feedback ◄──────────────────────────────────┘
                              │  /metrics                  charge le champion au démarrage
                              ▼                  ▼
                        Prometheus ─► Grafana   Postgres (prédictions, feedback, contrôles)
```

**Cycle de vie d'un modèle** : l'entraînement produit une version ; la *gate* (R² ≥ 0,70, prix qui
augmente avec la surface et la qualité, pas pire que le champion) lui donne l'alias `challenger` ou
la refuse avec ses raisons ; **une personne** la promeut en `champion` ; un `rollback` revient à la
version précédente. Un modèle ne passe jamais en production tout seul.

## Adresses des services

Tout n'écoute que sur `127.0.0.1` : seul votre ordinateur y a accès.

| Service | URL | Accès |
|---|---|---|
| **API** : documentation interactive | http://localhost:8000/docs | clé API (ligne `API_KEYS` du `.env`), bouton *Authorize* |
| API : vivacité / disponibilité | http://localhost:8000/health · http://localhost:8000/ready | libre (`/ready` = 503 tant qu'aucun champion n'est chargé) |
| API : métriques Prometheus | http://localhost:8000/metrics | libre |
| **MLflow** : essais et registre des modèles | http://localhost:5000 | libre |
| **Dagster** : jobs et historique des exécutions | http://localhost:3000 | libre |
| **Prometheus** | http://localhost:9090 | libre ; cibles : `/targets`, alertes : `/alerts` |
| **Grafana** | http://localhost:3001 | utilisateur `admin`, mot de passe `GRAFANA_ADMIN_PASSWORD` du `.env` |
| Grafana : tableau de bord direct | http://localhost:3001/d/house-price-api | idem |
| **Postgres** | `localhost:5432` | pas une page web ; base `monitoring`, utilisateur `app` |
| Dagster daemon | — | tourne en arrière-plan, pas d'interface |

Les mots de passe et la clé d'API ne sont **jamais** écrits dans ce dépôt : ils sont dans `.env`
(ignoré par Git).

## Déploiement

### Prérequis

- Git, Python 3.11 (installé automatiquement par uv), [uv](https://docs.astral.sh/uv/), Java 17
  (uniquement pour les tests locaux PySpark)
- Docker Desktop (ou Docker Engine + Compose v2) **lancé**, avec au moins 4 Go de mémoire alloués
  (6 Go confortable)
- Ports libres sur `127.0.0.1` : 5432, 5000, 3000, 8000, 9090, 3001

### 1. Configuration

```bash
cp .env.example .env        # PowerShell : Copy-Item .env.example .env
uv sync --all-groups        # crée .venv avec les versions exactes de uv.lock
```

Remplacez chaque valeur `change-me-*` du `.env` par un secret aléatoire :

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

Utilisez la même valeur pour `APP_DB_PASSWORD` et pour le mot de passe dans `DATABASE_URL`.
Postgres ne lit ces mots de passe qu'à la **première** création de la base : les changer ensuite
demande de réinitialiser les volumes (`docker compose down -v`, qui efface les données).

Créez aussi les dossiers `data/` et `artifacts/` vous-même avant le premier lancement.

### 2. Démarrer la pile

```bash
docker compose config --quiet          # valide compose.yaml et le .env (aucune sortie = OK)
docker compose build mlflow api        # construit les 2 images, une seule fois chacune
docker compose up -d                   # démarre les 7 services
docker compose ps                      # postgres, mlflow et api doivent être "healthy"
```

> **Pourquoi `build` puis `up -d` plutôt que `up --build -d` ?** Avec Docker Desktop (magasin
> d'images containerd), `up --build` construit en parallèle trois services qui partagent la même image
> `modern-mlops-pipeline:local` et échoue avec « image already exists ». Construire les images une par
> une évite ce conflit.

Vérification : http://localhost:8000/health répond 200, et http://localhost:8000/ready répond **503**
(« No champion model is loaded ») : c'est normal, aucun modèle n'est encore entraîné.

### 3. Entraîner le premier modèle

Dans Dagster (http://localhost:3000) : *Jobs* → `training_job` → *Materialize all*. Les quatre étapes
(`raw_housing_data` → `processed_housing_data` → `feast_feature_view` → `candidate_model`) doivent
passer au vert. Dans MLflow, la version 1 apparaît avec ses paramètres, ses métriques et sa
`model_card.md` ; si la gate est passée, elle porte l'alias `challenger`.

> **Les données** : le CSV brut n'est téléchargé qu'une fois (`data/raw/housing.csv`). Relancer
> l'entraînement réutilise le même fichier (même empreinte SHA-256) ; une nouvelle version de modèle est
> tout de même créée à chaque exécution. Pour utiliser d'autres données, remplacez ce fichier.

### 4. Mettre le modèle en production (décision humaine)

```bash
docker compose exec api mlops-model status    # champion / challenger et leur RMSE
docker compose exec api mlops-model promote   # le challenger devient champion (refusé s'il a échoué à la gate)
docker compose restart api                    # l'API charge le nouveau champion
```

`/ready` répond alors 200 avec la `model_version` servie.

### 5. Vérifier le tout

```bash
uv run python scripts/smoke_test.py --api-key "<votre clé>"        # chaque ligne doit afficher PASS
uv run python scripts/replay_holdout.py --api-key "<votre clé>" --rows 200   # trafic réaliste + feedback
```

Puis lancer `monitoring_job` dans Dagster (contrôles de dérive et de précision réelle). Le tableau de
bord Grafana *House Price API* affiche les requêtes et la table « Latest monitoring checks ».

## Exploitation courante

| Besoin | Commande |
|---|---|
| Revenir au champion précédent | `docker compose exec api mlops-model rollback` puis `docker compose restart api` |
| Voir les logs d'un service | `docker compose logs -f api` (ou `dagster-daemon`, `mlflow`…) |
| Arrêter la pile (données conservées) | `docker compose down` |
| Tout effacer, données comprises | `docker compose down -v` ⚠️ supprime bases, modèles et historiques |
| Après une modification de `src/` | `docker compose build mlflow api` puis `docker compose up -d` |
| Tests, lint, format | `uv run ruff format . && uv run ruff check . && uv run pytest -q` |
| Test de charge | `$env:API_KEY="<clé>"; uv run locust -f loadtest/locustfile.py --headless -u 20 -r 5 -t 2m --host http://127.0.0.1:8000` |

Planification Dagster : `training_job` tous les lundis à 06:00 UTC (il ne produit qu'un *candidat*),
`monitoring_job` tous les jours à 07:00 UTC.

## Alertes Prometheus

`ApiDown`, `ModelNotLoaded`, `HighErrorRate` (> 5 % d'erreurs sur `/predict`), `HighLatency`
(p95 > 200 ms) et `PredictionLogFailures` (prédictions non enregistrées). Elles sont visibles sur
http://localhost:9090/alerts.

## Structure du projet

```
src/mlops/        code : settings, validation, data, features, evaluation, train, registry,
                  db, monitoring, orchestration (Dagster), api (FastAPI), logging_config
tests/            tests unitaires (pytest)
scripts/          smoke_test.py, replay_holdout.py
loadtest/         locustfile.py
docker/           Dockerfiles, configuration Dagster, script d'initialisation de Postgres
monitoring/       Prometheus (config + alertes) et Grafana (sources de données + tableau de bord)
feature_repo/     configuration Feast
compose.yaml      la pile complète (7 services)
```

## Dépannage

| Symptôme | Cause probable et solution |
|---|---|
| `docker compose` ne répond pas | Docker Desktop n'est pas lancé |
| `image ... already exists` au `up --build` | voir l'encadré de la section « Démarrer la pile » |
| `/ready` répond 503 | aucun champion : entraîner puis promouvoir (sections 3 et 4), puis redémarrer l'API |
| `variable is not set` au démarrage | une valeur manque dans `.env` ; comparez avec `.env.example` |
| Port déjà utilisé | changez `POSTGRES_HOST_PORT` dans `.env`, ou arrêtez le programme qui l'occupe |
| PySpark échoue en natif sous Windows | le pipeline de données s'exécute dans Docker (via Dagster), pas en natif |
