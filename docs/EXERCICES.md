# Exercices de sécurité : observations

> « Un mécanisme de sécurité que vous n'avez jamais vu fonctionner est un mécanisme auquel vous ne
> pouvez pas vous fier. » Pour chaque exercice : **ce qui s'est passé**, **où je l'ai vu**, **pourquoi**.

Contexte au départ : pile Docker complète en marche, modèle `house-price-model` **version 1 = champion**
(RMSE de test 31 823 $, R² 0,845), 201 prédictions réelles avec feedback en base.

| # | Exercice | État |
|---|---|---|
| 1 | Alerte de précision (marché +60 %) | ✅ fait |
| 2 | Rejet par la gate (`RIDGE_ALPHA=1000`) | ✅ fait |
| 3 | Promotion et retour en arrière | ✅ fait |
| 4 | Panne de la base de données | ✅ fait |
| 5 | Mauvaises données | ✅ fait |

---

## Exercice 1 : alerte de précision

**Manipulation.** `uv run python scripts/replay_holdout.py --api-key <clé> --rows 200 --price-shift 1.6`
puis lancement du `monitoring_job` dans Dagster.

L'option `--price-shift 1.6` ne modifie ni le modèle ni le code de l'API : le script envoie les 200
ventes les plus récentes à `/predict` (le modèle prédit comme d'habitude), puis renvoie à `/feedback`
le vrai prix **multiplié par 1,6** (`scripts/replay_holdout.py`, ligne 38). On simule ainsi un marché où
les maisons se vendent 60 % plus cher que ce que le modèle attend.

**Ce qui s'est passé.**

| | Avant | Après |
|---|---|---|
| Lignes avec feedback en base | 202 | 402 |
| RMSE réel du champion | 32 507 $ | **95 022 $** |
| Seuil d'échec (1,5 × RMSE de test) | 47 735 $ | 47 735 $ |

Le `monitoring_job` a **échoué** : l'étape `input_drift_check` est restée verte, l'étape
`live_quality_check` est passée au rouge avec le message
`live_rmse failed: 95021.982 > 47734.724`. Une ligne `live_rmse` avec `passed = false` a été écrite
dans la table `monitoring_results` (valeur 95 022, seuil 47 735, 402 lignes analysées).

**Où je l'ai vu.**
- *Dagster* (http://localhost:3000) : run `monitoring_job` en échec, étape `live_quality_check` en rouge.
- *Grafana* (tableau « Latest monitoring checks ») : la dernière ligne `live_rmse` affiche `passed = false`.
- *Postgres* : `SELECT * FROM monitoring_results ORDER BY checked_at DESC` renvoie cette même ligne.

**Pourquoi.**
- Le contrôle compare l'erreur **mesurée en conditions réelles** (prédictions rapprochées des vrais prix
  reçus par `/feedback`) à l'erreur mesurée au moment du test. Au-delà de 1,5 fois cette référence, le
  modèle est jugé dégradé, et le run échoue **volontairement** pour qu'il apparaisse en rouge plutôt que
  de passer inaperçu.
- `input_drift_check` est resté **vert** : les *maisons* envoyées sont les mêmes qu'à l'entraînement, seuls
  les *prix* ont changé. C'est la différence entre la **dérive des données** (les entrées changent : un
  simple avertissement) et la **dégradation de la performance** (les prédictions sont fausses : la
  preuve). Seule la comparaison avec les vrais prix permet de détecter la seconde.
- Aucune alerte Prometheus ne s'est déclenchée, et c'est normal : les alertes de Prometheus surveillent
  la **santé de l'API** (disponibilité, erreurs, latence, modèle chargé, écritures en base), pas la
  qualité du modèle. La qualité est surveillée par ce contrôle quotidien de Dagster.

**Remarque.** Ces 200 feedbacks truqués restent en base et faussent la précision réelle tant que la
fenêtre de 30 jours les contient : le `monitoring_job` continuera d'échouer jusqu'à leur suppression.

---

## Exercice 2 : rejet par la gate

**Manipulation.** `RIDGE_ALPHA=1000` dans `.env` (régularisation 1000 fois plus forte : un modèle bridé),
`docker compose up -d` (recrée seulement `dagster-daemon` et `dagster-webserver`, l'API continue de servir
le champion), puis lancement du `training_job` dans Dagster (deux fois : versions 2 et 3, identiques).

**Ce qui s'est passé.**

| | Version 1 (champion) | Versions 2 et 3 |
|---|---|---|
| `alpha` | 1,0 | 1000,0 |
| RMSE sur le holdout | 31 823 $ | **41 975 $** |
| R² | 0,845 | 0,730 |
| `gate_passed` | true | **false** |
| `gate_reasons` | none | `RMSE 41,975 is worse than the champion's 31,823.` |
| Alias | `champion` | **aucun** (pas de `challenger`) |

- Le `training_job` est **resté vert** (SUCCESS) : un modèle refusé par la gate n'est pas une panne du
  pipeline, c'est une décision enregistrée. Dans Dagster, l'étape `candidate_model` indique
  `gate_passed = False` et la raison, avec un avertissement dans les logs.
- Tentatives de promotion forcée :
  - `mlops-model promote --version 2` → `v2 did not pass the gate: RMSE 41,975 is worse than the
    champion's 31,823.` (code de sortie 1)
  - `mlops-model promote --version 3` → même refus (code 1)
  - `mlops-model promote` sans argument → `No challenger to promote. Run the training job first.` (code 1)
- Le champion est resté la **version 1** et `/ready` répond toujours `model_version: "1"`.
- Retour à la normale : `RIDGE_ALPHA=1.0` dans `.env`, `docker compose up -d`, valeur vérifiée dans les
  conteneurs.

**Où je l'ai vu.**
- *MLflow* (Models → house-price-model) : versions 2 et 3 sans alias, tags `gate_passed=false` et
  `gate_reasons`; la `model_card.md` de ces versions affiche « Promotion gate : FAILED ».
- *Dagster* : métadonnées de l'asset `candidate_model`.
- *Terminal* : le refus de `mlops-model promote`.

**Pourquoi.**
- La gate applique trois règles : R² ≥ 0,70, comportement sensé (le prix augmente avec la surface et la
  qualité), et **pas pire que le champion** mesuré sur les mêmes données de test. Ici, c'est la **troisième
  règle seule** qui a refusé le modèle : son R² (0,730) dépasse le plancher de 0,70. Le seuil absolu
  n'aurait donc pas suffi à l'arrêter ; la comparaison avec le champion est ce qui empêche une régression.
- La commande `promote` ne fait pas confiance à l'appelant : elle relit le tag `gate_passed` de la version et
  refuse si elle n'a pas passé la gate, même quand on précise `--version`. La protection ne dépend donc pas
  de l'interface utilisée.
- L'entraînement est automatisable sans risque, car il ne produit que des *candidats* : un mauvais modèle
  ne peut pas atteindre la production par accident.

## Exercice 3 : promotion et retour en arrière

**Manipulation.** `training_job` relancé avec `RIDGE_ALPHA=1.0` (version 4), puis, dans le conteneur de
l'API : `mlops-model promote`, `docker compose restart api`, contrôle de `/ready`, `mlops-model rollback`,
`docker compose restart api`, contrôle de `/ready`.

**Ce qui s'est passé.**

| Étape | Résultat |
|---|---|
| Entraînement (alpha 1,0) | version 4 : RMSE 31 823 $ (identique au champion), `gate_passed = true`, alias `challenger` |
| `mlops-model promote` | `v4 is now the champion (previous: 1).` L'alias `challenger` est retiré. |
| `/ready` après redémarrage | `{"status":"ready","model_version":"4"}` |
| `mlops-model rollback` | `Rolled back: v1 is the champion again (was v4).` Aucune version à préciser. |
| `/ready` après redémarrage | `{"status":"ready","model_version":"1"}` |

Tags de traçabilité écrits sur les versions :
- version 4 : `previous_champion = 1`, `promoted_at = 2026-11-05T13:58:51Z`, `promoted_by = app`
- version 1 : `rolled_back_from = 4`, `rolled_back_at = 2026-11-05T13:59:04Z` (en plus de sa promotion initiale)

**Où je l'ai vu.** Terminal (sorties des commandes), `/ready` de l'API, et *MLflow* : alias déplacé de la
version 1 à la 4 puis ramené à la 1, avec les tags ci-dessus.

**Pourquoi.**
- Un modèle qui égale le champion passe la gate : la règle est « pas **pire** que le champion », donc un
  RMSE égal est accepté. Ce n'est pas un défaut : un ré-entraînement sur les mêmes données donne le même modèle.
- La promotion enregistre **qui, quand, et quelle version elle remplace** (`previous_champion`). C'est ce qui
  permet au `rollback` de savoir où revenir sans qu'on le précise : une seule commande, sans avoir à réfléchir,
  en cas d'incident.
- L'API ne change de modèle qu'à son **redémarrage** : elle charge le champion une seule fois, au démarrage.
  Changer l'alias dans MLflow ne modifie donc jamais le modèle servi en cours de route, et la bascule
  est un geste volontaire. Le redémarrage coupe l'API quelques secondes ; l'alerte `ApiDown` exige 2 minutes
  d'indisponibilité, elle ne s'est donc pas déclenchée.
- Après le rollback, la version 4 n'a plus d'alias : elle reste dans le registre (rien n'est supprimé) et
  pourrait être promue de nouveau avec `promote --version 4`.

## Exercice 4 : panne de la base de données

**Manipulation.** `docker compose stop postgres`, trois requêtes `POST /predict`, observation de l'alerte
dans Prometheus, puis `docker compose start postgres` et une nouvelle prédiction.

**Ce qui s'est passé.**
- Pendant la panne, les 3 requêtes ont répondu **HTTP 200 avec un prix** (211 827 $, modèle v1). La première
  a pris 0,1 s (connexion déjà ouverte qui échoue vite), les suivantes 2,0 s (attente du délai maximal du
  pool de connexions, fixé à 2 s) : l'API reste utilisable, un peu plus lente.
- Le compteur `prediction_log_failures` est passé de 0 à 3, et le journal de l'API contient 3 lignes
  `Could not store prediction`.
- L'alerte **`PredictionLogFailures`** est passée à l'état `firing` (sévérité `warning`) environ **50 secondes**
  après la première panne, avec le message « Predictions are not being stored; monitoring data is being
  lost. ». Elle n'a pas de délai d'attente (`for:`) et se déclenche dès que `increase(...[10m]) > 0` ; il faut
  seulement deux relevés de Prometheus (15 s chacun).
- Après `docker compose start postgres` : base saine en ~6 s, la prédiction suivante est **enregistrée**
  (402 → 403 lignes). Les 3 prédictions de la panne sont **définitivement perdues** pour le monitoring ; le
  compteur d'échecs reste à 3. L'API s'est rétablie **toute seule**, sans redémarrage.

**Où je l'ai vu.** Terminal (réponses HTTP et durées), *Prometheus* → Alerts (http://localhost:9090/alerts),
endpoint `/metrics` de l'API, journal `docker compose logs api`, et nombre de lignes dans `predictions`.

**Pourquoi.**
- Dans `/predict`, l'écriture en base est entourée d'un `try/except` : si elle échoue, l'erreur est journalisée
  et **comptée**, mais le client reçoit quand même son prix. Un problème de supervision ne doit pas devenir
  une panne pour l'utilisateur.
- Comme les données de monitoring sont perdues, il faut le **savoir** : c'est le rôle du compteur et de l'alerte.
- Le pool de connexions rouvre des connexions à la volée, et la création des tables est paresseuse
  (`_ensure_schema`), d'où le rétablissement automatique.
- **Effet de bord constaté :** Dagster (webserver et daemon) a perdu ses connexions à Postgres (« could not
  translate host name postgres ») et ne s'est pas rétabli seul : le `training_job` lancé ensuite a été marqué
  « failed from outside the execution context » sans exécuter la moindre étape. Un
  `docker compose restart dagster-daemon dagster-webserver` a réglé le problème. Après une panne de la
  base, il faut donc redémarrer ces deux services. (Le comportement de MLflow n'a pas été examiné ici.)

---

## Exercice 5 : mauvaises données

**Manipulation.** Sauvegarde de `data/raw/housing.csv`, puis `OverallQual = 42` sur les 5 premières maisons
(échelle valide : 1 à 10). Empreinte SHA-256 : `b214efc1…` → `85160650…`. Lancement du `training_job`.

**Ce qui s'est passé.**
- Le run Dagster a **échoué** : `raw_housing_data` en vert, **`processed_housing_data` en rouge**, et
  `feast_feature_view` et `candidate_model` **n'ont jamais été lancés**.
- `artifacts/reports/validation.json` a été écrit avec `"passed": false`, 1 460 lignes lues et la cause :
  **`OverallQual: 5 values outside (1, 10).`**
- **Aucun nouveau modèle** : toujours 4 versions dans MLflow, champion = version 1, aucun challenger. Le
  fichier nettoyé `data/processed/housing.parquet` n'a pas été touché.
- Un premier lancement avait échoué pour une autre raison (connexions Dagster perdues, voir l'exercice 4) :
  le rapport de validation n'avait pas été réécrit, ce qui permettait de distinguer les deux causes.
- Restauration : copie de la sauvegarde sur `data/raw/housing.csv` ; empreinte `b214efc1…` identique à
  l'originale. (La consigne propose de supprimer le fichier et de relancer le job pour le retélécharger ;
  restaurer la copie donne exactement les mêmes octets, sans dépendre du réseau.)

**Où je l'ai vu.** *Dagster* (étape rouge, journal avec le message de `ValidationError`),
`artifacts/reports/validation.json`, et *MLflow* (nombre de versions inchangé).

**Pourquoi.**
- `prepare_data()` valide les données **telles qu'elles sont arrivées, avant tout nettoyage**, et enregistre le
  rapport même en cas d'échec : c'est à ce moment qu'on en a besoin. En cas de violation, elle lève
  `ValidationError`, donc rien n'est écrit en aval.
- Les règles (plages de valeurs, taux de valeurs manquantes, doublons d'identifiants, colonnes et volume)
  forment un **contrat de données**. Un modèle entraîné sur des données invalides serait faux d'une façon
  qu'aucune métrique ne révèle ; mieux vaut arrêter le pipeline que livrer un modèle qui a l'air correct.
- Le nettoyage (`dropna`) n'aurait pas détecté ces valeurs : elles ne sont pas manquantes, elles sont
  *absurdes*. Seule la vérification de plage les attrape.

**Remarque.** Le `validation.json` en échec reste dans `artifacts/reports/` jusqu'au prochain
`training_job`, qui le réécrit avant d'entraîner quoi que ce soit.
