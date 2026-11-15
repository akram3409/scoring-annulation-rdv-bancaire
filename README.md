# Prédiction d'annulation de rendez-vous clients

Ce projet regroupe un travail complet de préparation de données (en **R**) et de modélisation prédictive (en **Python**) sur des données réelles d'entreprise dans le secteur du courtage bancaire.

L'objectif est de construire un score permettant d'estimer, dès la prise de contact ($t_0$), la probabilité qu'un client annule son rendez-vous ($Y = 1$ pour `"annuler client"`, $Y = 0$ pour `"fait"`).

---

## 1. Données de départ et périmètre d'étude

* **Base initiale :** **31 286 lignes et 117 variables** brutes. Le jeu de données initial comporte beaucoup de bruit : doublons liés à des reports de rendez-vous, valeurs aberrantes, colonnes quasi vides (> 95 % de NA) et variables stockées sous forme de listes textuelles (ex: `"[65.19, 84.17]"` pour les mensualités ou capitaux restants dus).
* **Définition du périmètre :** L'étude se concentre sur les dossiers de propriétaires sans co-emprunteur et sur les rendez-vous dont l'issue finale est connue (`"fait"` ou `"annuler client"`). La baisse du nombre de lignes provient donc principalement de ce filtrage métier et de la suppression des doublons, et non d'une suppression massive de valeurs manquantes.
* **Jeu de données final :** **6 743 observations et 32 variables** propres, sans valeurs manquantes, avec un fort déséquilibre de classe (**14,4 % d'annulations**).

---

## 2. Préparation des données (`R` / `dplyr`)
📄 **[Voir le rapport détaillé de préparation des données (`Preparation_des_donnees.md`)](./Preparation_des_donnees.md)**

Principales étapes réalisées dans le script R Markdown :
1. **Dédoublonnage :** Suppression des doublons exacts, puis traitement des doublons sur le triplet `(id_dossier, date_rdv, heure_rdv_debut)` en gardant la ligne la plus complète et la plus récente.
2. **Suppression du Data Leakage :** Retrait de toutes les variables renseignées après la prise de rendez-vous ($t_0$) comme le statut final ou les frais de notaire, et suppression des délais négatifs incohérents (`horizon_rdv < 0`).
3. **Création de variables (Feature Engineering) :**
   * Extraction des listes de crédits (conso et immo) pour calculer le nombre de crédits, la somme des mensualités, la somme des capitaux restants dus et un indicateur de retard de paiement.
   * Regroupement des revenus, des charges et des incidents bancaires, et calcul du taux d'endettement.
   * Imputation du type de contrat pour les retraités à partir de l'âge et suppression des variables colinéaires (`horizon_retraite_emp` très corrélé à `age_emp`).

---

## 3. Modélisation (`Python` / `scikit-learn` & `PyTorch`)
📓 **[Voir le Notebook de modélisation et les matrices de confusion (`Modelisation_benchmark.ipynb`)](./Modelisation_benchmark.ipynb)**

* **Découpage des données :** Séparation stratifiée en trois ensembles : `Train` (60 %), `Validation` (20 %) et `Test` (20 %).
* **Prétraitement sans fuite de données :** Standardisation des variables numériques (`StandardScaler`) et encodage des variables catégorielles (`OneHotEncoder`) via un `ColumnTransformer` ajusté uniquement sur le `Train`.
* **Gestion du déséquilibre de classe :**
  * Pondération des classes pendant l'entraînement (`class_weight="balanced"` sur scikit-learn et `pos_weight` sur PyTorch) pour pénaliser davantage les erreurs sur la classe minoritaire.
  * Choix du seuil de décision optimal $\tau^*$ sur l'échantillon de `Validation` de façon à maximiser le **F1-score**, avant évaluation finale sur le `Test`.

---

## 4. Résultats sur l'échantillon de Test

*(AUPRC d'un classifieur aléatoire = **0.1438**)*

| Modèle | AUPRC (PR-AUC) | Précision | Rappel | F1-score | Seuil optimal ($\tau^*$ Val) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Régression Logistique (L1)** | 0.2203 | 0.2159 | **0.5876** | **0.3158** | 0.5224 |
| **Random Forest** | **0.2450** | **0.2336** | 0.4588 | 0.3096 | 0.4082 |
| **Réseau de Neurones (PyTorch MLP)** | 0.2384 | 0.2319 | 0.4948 | **0.3158** | 0.5465 |

### Bilan et interprétation métier
1. **Variables les plus importantes :** L'analyse de la Random Forest montre que le risque d'annulation dépend surtout du niveau d'endettement à la consommation (`somme_conso_crd`, `taux_endettement`, `somme_conso_mensualite`), de l'âge du client et du délai avant le rendez-vous (`horizon_rdv`).
2. **Plafond de performance (~0.245 d'AUPRC) :** Les trois modèles (linéaire, arbres et réseau de neurones) donnent des résultats très proches. Cela s'explique par la nature du problème : les données financières et patrimoniales ne suffisent pas à expliquer totalement une annulation de rendez-vous, qui dépend aussi beaucoup d'imprévus personnels non présents dans la base.
3. **Utilisation en pratique :** Avec une précision autour de 23 % (beaucoup de faux positifs), il ne faut pas utiliser ce modèle pour annuler automatiquement un créneau. En revanche, comme il détecte entre **46 % et 59 % des vraies annulations** (rappel), il est très utile pour cibler les clients à relancer en priorité (SMS de rappel ou appel de confirmation avant le rendez-vous).

---

## 5. Contenu du dépôt
* `Preparation_des_donnees.Rmd` & `Preparation_des_donnees.md` : Code R Markdown et rapport compilé du nettoyage des données.
* `Modelisation_benchmark.ipynb` : Notebook Jupyter avec le code d'entraînement, les résultats et les matrices de confusion.
* `Comparaison_Modeles.py` : Script Python complet de la partie modélisation.
