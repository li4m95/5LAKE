# House Price API : Runbook d'exploitation

Ce document dit **quoi faire quand…** pour exploiter le système sans lire le code. Les commandes sont
écrites pour PowerShell (Windows) ; elles fonctionnent aussi sous Linux/macOS, sauf mention contraire.
Adresses des services : voir le [README](../README.md).

## Objectifs de niveau de service (SLO, mesurés sur 7 jours)

| SLO | Objectif | Où le voir | Mesure de référence |
|---|---|---|---|
| Disponibilité | ≥ 99 % des requêtes `/predict` sans erreur 5xx | Grafana : requêtes par statut | 0 échec sur 1 748 requêtes (test de charge) |
| Latence | p95 de `/predict` < 200 ms | Grafana : latence p95 ; alerte `HighLatency` | **p95 = 23 ms**, p99 = 36 ms à 60 req/s |
| Fraîcheur du monitoring | `monitoring_job` réussi dans les dernières 26 h | Dagster : historique des runs | exécution quotidienne à 07:00 UTC |
| Qualité du modèle | RMSE réel ≤ 1,5 × RMSE de test | Grafana : « Latest monitoring checks » | 32 588 $ pour une limite de 47 735 $ |

*Mesure du 2026-11-05 : `locust -u 20 -r 5 -t 30s`, 20 utilisateurs simultanés, 1 748 requêtes, 0 échec,
moyenne 11,9 ms, p50 10 ms.*

## Opérations courantes

- **Démarrer** : `docker compose up -d` (sans `--build`). **Arrêter** : `docker compose down`.
  ⚠️ N'ajoutez **jamais** `-v` sauf si vous voulez effacer toutes les données (bases, modèles, historiques).
- **État** : `docker compose ps` et `docker compose exec api mlops-model status`
- **Journaux** : `docker compose logs -f api` (JSON ; chercher par `request_id`)
- **Après une modification du code `src/`** : `docker compose build mlflow api` **puis** `docker compose up -d`.
  Ne pas utiliser `up --build` : avec Docker Desktop, les trois services qui partagent l'image
  `modern-mlops-pipeline:local` se la disputent et l'opération échoue (« image already exists »).

## Mettre un nouveau modèle en production

1. Lancer `training_job` dans Dagster (ou attendre l'exécution planifiée du lundi 06:00 UTC).
2. Dans MLflow, lire la `model_card.md` et les tags de la nouvelle version (`gate_passed`, `gate_reasons`).
3. `docker compose exec api mlops-model promote`
4. `docker compose restart api`
5. `uv run python scripts/smoke_test.py --api-key <clé>` : chaque ligne doit afficher PASS.
6. Surveiller le tableau de bord pendant 15 minutes.

`promote` refuse toute version qui n'a pas passé la gate, même avec `--version`.

## Revenir à la version précédente

1. `docker compose exec api mlops-model rollback` (ou `--version <n>`)
2. `docker compose restart api`
3. Smoke test, puis vérifier que `/ready` affiche la version attendue.

Le redémarrage de l'API la coupe quelques secondes (l'alerte `ApiDown` exige 2 minutes).

## Alerte ou signal → action

| Alerte / signal | Premières vérifications | Correction habituelle |
|---|---|---|
| `ApiDown` | `docker compose ps`, `docker compose logs api` | Redémarrer l'API ; vérifier les limites de mémoire |
| `ModelNotLoaded` | `docker compose logs api` ; MLflow est-il sain ? existe-t-il un champion ? | Réparer MLflow, promouvoir une version, redémarrer l'API |
| `HighErrorRate` | Exceptions dans les journaux de l'API (`"level": "ERROR"`) | Revenir à la version précédente si le problème suit une mise en production |
| `HighLatency` | `docker stats --no-stream` ; un entraînement tourne-t-il ? | Attendre la fin de l'entraînement ; étudier les requêtes lentes |
| `PredictionLogFailures` | `docker compose ps postgres`, journaux de Postgres | Redémarrer Postgres ; vérifier l'espace disque. **Puis redémarrer Dagster** (voir ci-dessous). Les prédictions faites pendant la panne sont perdues. |
| `input_drift` en échec | Ouvrir le rapport HTML dans `artifacts/reports/` | Identifier les variables qui dérivent ; réentraîner si le changement est réel |
| `live_rmse` en échec | Comparer prédictions et feedbacks dans Postgres | Réentraîner sur des données récentes ; revenir en arrière si une mise en production récente est en cause |
| `Validation failed` dans `training_job` | `artifacts/reports/validation.json` | Corriger ou rejeter les données sources. **Ne jamais assouplir une règle pour la faire passer.** |
| Run Dagster « marked as failed from outside the execution context » | Postgres a-t-il été coupé récemment ? | `docker compose restart dagster-daemon dagster-webserver`, puis relancer le job |

## Sauvegardes

```powershell
New-Item -ItemType Directory -Force backups | Out-Null          # le dossier backups/ est ignoré par Git
$d = Get-Date -Format yyyy-MM-dd
docker compose exec -T postgres pg_dumpall -U postgres --file=/tmp/backup.sql
docker compose cp postgres:/tmp/backup.sql "backups/backup-$d.sql"
docker run --rm -v modern-mlops_mlflow-artifacts:/data -v "$((Get-Location).Path)\backups:/backup" `
  alpine:3.20 tar czf "/backup/mlflow-artifacts-$d.tgz" -C /data .
```

Ces dumps contiennent des données et des empreintes de mots de passe : ne les commitez pas et protégez-les.

**Restauration testée le 2026-11-05** sans toucher à la production : un dump de la base `monitoring` a été
rechargé dans une base temporaire (`restore_test`) ; les comptes étaient identiques (3 899 prédictions,
402 feedbacks, 6 contrôles), puis la base temporaire a été supprimée.
Pour restaurer pour de vrai : recréer les volumes (`docker compose down -v`, `docker compose up -d`), puis
recharger le dump avec `psql -U postgres -f /tmp/backup.sql` depuis le conteneur Postgres. Entraînez-vous à une
restauration complète au moins une fois.

## Changer une clé d'API

1. Ajouter la nouvelle clé à `API_KEYS` dans `.env` (séparées par des virgules), puis `docker compose up -d api`.
2. Faire migrer les clients vers la nouvelle clé.
3. Retirer l'ancienne clé, puis `docker compose up -d api`.

## Politique de réentraînement

- **Planifié** : `training_job` tourne chaque lundi et produit un *candidat*.
- **Sur signal** : le lancer à la main quand le contrôle de dérive ou de précision échoue.
- **Promotion** : toujours par une personne, uniquement pour une version qui a passé la gate, et après lecture
  de la fiche du modèle.
- **Retour en arrière** : quand le taux d'erreur, la latence ou la précision réelle se dégradent après une
  mise en production.
- Le CSV brut n'est téléchargé qu'une fois : relancer l'entraînement réutilise les mêmes données (même
  empreinte SHA-256) tant que `data/raw/housing.csv` n'est pas remplacé.

## Dépannage

**`docker compose config` dit qu'une variable n'est pas définie.** Il manque une valeur dans `.env`, ou le fichier
n'est pas dans le dossier du projet. Comparez avec `.env.example`.

**« ports are not available … 5432 ».** Le port est déjà pris. Mettez `POSTGRES_HOST_PORT=5433` dans `.env` et
utilisez le même port dans `DATABASE_URL`. Les conteneurs ne sont pas concernés.

**Postgres n'exécute pas le script d'initialisation, ou un service ne peut pas se connecter.** Le fichier `.sh`
doit avoir des fins de ligne Unix (LF). Le script ne s'exécute qu'à la **première** création de la base : si vous
avez changé des mots de passe après, il faut réinitialiser avec `docker compose down -v` (efface les données).

**`/ready` répond 503.** Aucun modèle n'est chargé. `docker compose exec api mlops-model status` : sans champion,
entraîner puis promouvoir (voir plus haut). Sinon lire `docker compose logs api` ; l'API réessaie toutes les 30 s,
ou redémarrez-la.

**`image ... already exists` lors d'un `up --build`.** Voir « Opérations courantes » : construire avec
`docker compose build mlflow api`, puis `docker compose up -d`.

**Dagster ne réagit plus après une coupure de Postgres.** Ses connexions sont perdues :
`docker compose restart dagster-daemon dagster-webserver`.

**Git Bash : `/app/workspace.yaml` devient `C:/Program Files/Git/app/...`.** Git Bash réécrit les chemins qui
ressemblent à des chemins Linux. Mettez `MSYS_NO_PATHCONV=1` devant la commande, ou utilisez PowerShell.

**Les tests Spark échouent (« Python worker failed to connect back »).** Lancer les tests avec `uv run pytest`
pour que Spark utilise le Python du projet ; vérifier la mémoire libre (1 à 2 Go).

**Mémoire insuffisante ou tout est très lent.** Fermer d'autres applications, `docker stats --no-stream`, et
vérifier que Docker dispose d'au moins 4 Go. Pour entraîner, seuls `postgres`, `mlflow` et les services Dagster
sont nécessaires.

**Prometheus n'affiche aucune métrique de l'API.** Ouvrir http://localhost:9090/targets : la cible doit être
`api:8000`, pas `localhost:8000`.

**Grafana : la source « Monitoring DB » échoue.** L'utilisateur en lecture seule est créé par le script
d'initialisation à la première création de la base ; si la base existait avant, réinitialiser avec
`docker compose down -v` (efface les données).

**`uv lock` échoue.** Lire tout le message : il nomme les deux paquets en conflit. Ne changer que ces bornes de
version, puis relancer les tests. Ne pas supprimer `uv.lock` pour « repartir de zéro ».

## Dépendances : vulnérabilités connues (au 2026-11-05)

`pip-audit` signale des vulnérabilités connues dans 4 paquets verrouillés : **mlflow 2.22.5** (29),
**feast 0.47.0** (3), **pyarrow 18.0.0** (1) et **pytest 8.4.2** (1, outil de développement). Les corrections
exigent de relever des bornes de version fixées par le projet (`mlflow<3`, `feast<0.50`, `pyarrow<20`,
`pytest<9`), ce qui demande une migration testée. Tant que ce n'est pas fait, l'étape d'audit du CI échouera.
À traiter avant toute exposition hors d'une machine locale.
