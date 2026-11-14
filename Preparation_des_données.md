Préparation des données - Prédiction d’annulation de RDV
================
Akram El Mahzoum
2026-11-14

## 1. Chargement des données et définition du périmètre

``` r
library(dplyr)
library(tidyr)
library(lubridate)

data = read.csv("Data.csv", na.strings = c("", "NA", " ", "NaN"))
cat("Dimension de départ :", dim(data), "\n")
```

    ## Dimension de départ : 31286 117

La variable cible est `etat`. L’objectif est de prédire dès la prise de
rendez-vous ($t_0$) la probabilité que le client annule son rendez-vous
: \* `"annuler client"` correspond à la cible positive ($Y = 1$). \*
`"fait"` correspond à un rendez-vous honoré ($Y = 0$). \* Les autres
modalités (`"refus entreprise"`, etc.) ainsi que les dossiers locataires
sont exclus du périmètre d’étude.

``` r
data |> count(etat)
```

    ##                     etat     n
    ## 1         annuler client  3607
    ## 2                   fait 21285
    ## 3               pas fait   927
    ## 4 refus avant rendezvous  3826
    ## 5           refus client   483
    ## 6       refus entreprise  1158

``` r
data |> count(type_dossier)
```

    ##   type_dossier     n
    ## 1    locataire   565
    ## 2 proprietaire 30721

``` r
# Filtrage sur la cible, les propriétaires et suppression de l'index technique X
data = data |>
  filter(etat %in% c("annuler client", "fait")) |>
  filter(type_dossier == "proprietaire") |>
  select(-"X")

cat("Dimension après filtrage du périmètre :", dim(data), "\n")
```

    ## Dimension après filtrage du périmètre : 24377 116

------------------------------------------------------------------------

## 2. Premier nettoyage et dédoublonnage métier

### 2.1 Doublons exacts et cohérence temporelle ($t_0$)

On calcule `horizon_rdv` (délai en jours entre la prise de rendez-vous
et la date du rendez-vous) et on retire les observations incohérentes où
le rendez-vous précède la prise de contact.

``` r
# 1. Suppression des doublons exacts
nb_doublons = sum(duplicated(data))
cat("Nombre de lignes en double exactes :", nb_doublons, "\n")
```

    ## Nombre de lignes en double exactes : 1789

``` r
data = distinct(data)

# 2. Calcul de l'horizon du RDV et filtrage des horizons négatifs
data$date_aboutisant_azur = as.Date(data$date_aboutisant_azur)
data$date_rdv = as.Date(data$date_rdv)

data = data |>
  mutate(horizon_rdv = as.numeric(date_rdv - date_aboutisant_azur))

nb_negatifs = sum(data$horizon_rdv < 0, na.rm = TRUE)
cat("Nombre de lignes avec un horizon négatif :", nb_negatifs, "\n")
```

    ## Nombre de lignes avec un horizon négatif : 16

``` r
data = data |>
  filter(horizon_rdv >= 0 | is.na(horizon_rdv))

cat("Dimension actuelle :", dim(data), "\n")
```

    ## Dimension actuelle : 22572 117

### 2.2 Suppression des variables quasi vides (\> 95 % de NA)

``` r
seuil_na = 0.95
data = data |>
  select(where(~ mean(is.na(.x)) <= seuil_na)) |>
  select(-"segmentation")

cat("Dimension après retrait des colonnes > 95% NA :", dim(data), "\n")
```

    ## Dimension après retrait des colonnes > 95% NA : 22572 111

### 2.3 Dédoublonnage sur la clé métier `(id_dossier, date_rdv, heure_rdv_debut)`

Un même dossier ayant la même date et heure de rendez-vous correspond au
même événement. On vérifie d’abord l’absence de contradiction sur la
variable cible `etat`.

``` r
data_doublons = data |>
  group_by(id_dossier, date_rdv, heure_rdv_debut) |>
  filter(n() > 1) |>
  ungroup() |>
  arrange(id_dossier, date_rdv, heure_rdv_debut)

cat("Nombre de lignes impliquées dans des doublons de RDV :", nrow(data_doublons), "\n")
```

    ## Nombre de lignes impliquées dans des doublons de RDV : 9325

``` r
nb_conflits_cible = data |>
  group_by(id_dossier, date_rdv, heure_rdv_debut) |>
  filter(n() > 1) |>
  summarise(nb_etats_differents = n_distinct(etat), .groups = "drop") |>
  filter(nb_etats_differents > 1) |>
  nrow()

cat("Nombre de groupes de doublons avec des cibles contradictoires :", nb_conflits_cible, "\n")
```

    ## Nombre de groupes de doublons avec des cibles contradictoires : 0

Inspection des différences entre deux lignes d’un même rendez-vous :

``` r
colonnes_diff = names(data_doublons)[sapply(names(data_doublons), function(col) !identical(data_doublons[11, col], data_doublons[12, col]))]
print(colonnes_diff)
```

    ## [1] "date_aboutisant_azur" "horizon_rdv"

``` r
print(data_doublons[11:12, colonnes_diff])
```

    ## # A tibble: 2 × 2
    ##   date_aboutisant_azur horizon_rdv
    ##   <date>                     <dbl>
    ## 1 2023-09-07                    11
    ## 2 2023-09-06                    12

Il s’agit de fiches mises à jour ou reportées. On conserve la ligne la
mieux renseignée (moins de valeurs manquantes), et en cas d’égalité
celle ayant l’horizon le plus court (version la plus récente) :

``` r
nb_lignes_avant = nrow(data)

# 1. Doublons stricts hors date de contact
data = data |>
  group_by(across(-c(date_aboutisant_azur, horizon_rdv))) |>
  arrange(horizon_rdv) |>
  slice(1) |>
  ungroup()

# 2. Doublons partiels sur le triplet (id_dossier, date_rdv, heure_rdv_debut)
data$nb_na = rowSums(is.na(data))

data = data |>
  group_by(id_dossier, date_rdv, heure_rdv_debut) |>
  arrange(nb_na, horizon_rdv) |>
  slice(1) |>
  ungroup() |>
  select(-"nb_na")

cat("Nombre total de doublons métier supprimés :", nb_lignes_avant - nrow(data), "\n")
```

    ## Nombre total de doublons métier supprimés : 4881

``` r
cat("Dimension actuelle :", dim(data), "\n")
```

    ## Dimension actuelle : 17691 111

------------------------------------------------------------------------

## 3. Périmètre emprunteur, variables temporelles et valeurs aberrantes

### 3.1 Filtrage des dossiers sans co-emprunteur

On identifie la présence d’un co-emprunteur pour restreindre l’étude aux
dossiers sans co-emprunteur et retirer les colonnes associées.

``` r
data = data |>
  mutate(presence_coemp = ifelse(is.na(intitule_coemp) & is.na(date_naissance_coemp), 0, 1)) |>
  filter(presence_coemp == 0) |>
  select(-ends_with("coemp"), -coemp_id)

cat("Dimension après filtrage des co-emprunteurs :", dim(data), "\n")
```

    ## Dimension après filtrage des co-emprunteurs : 7577 91

### 3.2 Calcul de l’âge et de l’horizon de retraite à $t_0$

``` r
data$date_naissance_emp = as.Date(data$date_naissance_emp)
data$date_retraite_emp  = as.Date(data$date_retraite_emp)

data = data |>
  mutate(
    age_emp = round(as.numeric(date_aboutisant_azur - date_naissance_emp) / 365.25, 1),
    horizon_retraite_emp = round(as.numeric(date_retraite_emp - date_aboutisant_azur) / 365.25, 1)
  ) |>
  select(-date_naissance_emp, -date_retraite_emp)
```

### 3.3 Filtrage des valeurs quantitatives aberrantes

``` r
summary(select(data, where(is.numeric)))
```

    ##  penalite_remboursement frais_notaire    id_dossier          duree      
    ##  Min.   :    0.0        Min.   :   0   Min.   :2221046   Min.   : 12.0  
    ##  1st Qu.:    0.0        1st Qu.:1260   1st Qu.:2504184   1st Qu.:180.0  
    ##  Median :    0.0        Median :1540   Median :2684106   Median :360.0  
    ##  Mean   :  926.6        Mean   :1823   Mean   :2631692   Mean   :310.8  
    ##  3rd Qu.: 1470.0        3rd Qu.:2255   3rd Qu.:2777632   3rd Qu.:420.0  
    ##  Max.   :35248.2        Max.   :8520   Max.   :2875576   Max.   :420.0  
    ##                                                                         
    ##      emp_id         salaire_emp    rev_foncier_emp   conge_parental_emp
    ##  Min.   : 547164   Min.   :    0   Min.   :    0.0   Min.   :0         
    ##  1st Qu.:3328109   1st Qu.: 1833   1st Qu.:    0.0   1st Qu.:0         
    ##  Median :3580277   Median : 2250   Median :    0.0   Median :0         
    ##  Mean   :3488330   Mean   : 2537   Mean   :  139.1   Mean   :0         
    ##  3rd Qu.:3713381   3rd Qu.: 2896   3rd Qu.:    0.0   3rd Qu.:0         
    ##  Max.   :3847545   Max.   :37333   Max.   :17266.0   Max.   :0         
    ##                                                                        
    ##  allocation_familiale_emp    apl_emp        pension_alimentaire_emp
    ##  Min.   :   0.00          Min.   :  0.000   Min.   :   0.00        
    ##  1st Qu.:   0.00          1st Qu.:  0.000   1st Qu.:   0.00        
    ##  Median :   0.00          Median :  0.000   Median :   0.00        
    ##  Mean   :  11.39          Mean   :  0.446   Mean   :  12.09        
    ##  3rd Qu.:   0.00          3rd Qu.:  0.000   3rd Qu.:   0.00        
    ##  Max.   :1500.00          Max.   :397.000   Max.   :2400.00        
    ##                                                                    
    ##  pension_invalidite_emp anciennete_emp   nb_enfants_emp   retard_loyer_emp  
    ##  Min.   :   0.00        Min.   : 0.000   Min.   :0.0000   Min.   :  0.0000  
    ##  1st Qu.:   0.00        1st Qu.: 2.000   1st Qu.:0.0000   1st Qu.:  0.0000  
    ##  Median :   0.00        Median : 7.000   Median :0.0000   Median :  0.0000  
    ##  Mean   :  15.07        Mean   : 9.446   Mean   :0.1831   Mean   :  0.2731  
    ##  3rd Qu.:   0.00        3rd Qu.:15.000   3rd Qu.:0.0000   3rd Qu.:  0.0000  
    ##  Max.   :2840.00        Max.   :49.000   Max.   :5.0000   Max.   :826.0000  
    ##                                                           NA's   :11        
    ##  dette_famille_ami_emp retard_impot_emp  decouvert_emp     autre_dette_emp   
    ##  Min.   :    0.0       Min.   :    0.0   Min.   :    0.0   Min.   :    0.00  
    ##  1st Qu.:    0.0       1st Qu.:    0.0   1st Qu.:    0.0   1st Qu.:    0.00  
    ##  Median :    0.0       Median :    0.0   Median :  300.0   Median :    0.00  
    ##  Mean   :  185.4       Mean   :  115.8   Mean   :  630.4   Mean   :   69.22  
    ##  3rd Qu.:    0.0       3rd Qu.:    0.0   3rd Qu.: 1000.0   3rd Qu.:    0.00  
    ##  Max.   :82800.0       Max.   :60000.0   Max.   :35000.0   Max.   :52000.00  
    ##  NA's   :11            NA's   :11        NA's   :11        NA's   :11        
    ##  saisie_sur_salaire_emp avis_a_tiers_detenteurs_emp   loyer_emp      
    ##  Min.   :0.00000        Min.   :0.0000000           Min.   :   0.00  
    ##  1st Qu.:0.00000        1st Qu.:0.0000000           1st Qu.:   0.00  
    ##  Median :0.00000        Median :0.0000000           Median :   0.00  
    ##  Mean   :0.00119        Mean   :0.0006609           Mean   :  40.12  
    ##  3rd Qu.:0.00000        3rd Qu.:0.0000000           3rd Qu.:   0.00  
    ##  Max.   :1.00000        Max.   :1.0000000           Max.   :2700.00  
    ##  NA's   :11             NA's   :11                  NA's   :3078     
    ##  charges_loyer_emp pension_versee_emp charge_recurrente_emp charge_courante_emp
    ##  Min.   :  0.000   Min.   :   0.00    Min.   :   0.000      Min.   :   0.00    
    ##  1st Qu.:  0.000   1st Qu.:   0.00    1st Qu.:   0.000      1st Qu.:   0.00    
    ##  Median :  0.000   Median :   0.00    Median :   0.000      Median :   0.00    
    ##  Mean   :  2.402   Mean   :  24.17    Mean   :   1.172      Mean   :  10.64    
    ##  3rd Qu.:  0.000   3rd Qu.:   0.00    3rd Qu.:   0.000      3rd Qu.:   0.00    
    ##  Max.   :600.000   Max.   :2100.00    Max.   :1265.000      Max.   :2300.00    
    ##  NA's   :3078      NA's   :3078       NA's   :3078          NA's   :3078       
    ##  charge_future_eventuelle_emp   tresorerie     tresorerie_sur_facture
    ##  Min.   :0                    Min.   :     0   Min.   :     0        
    ##  1st Qu.:0                    1st Qu.:  2500   1st Qu.:     0        
    ##  Median :0                    Median :  5000   Median :     0        
    ##  Mean   :0                    Mean   : 10230   Mean   :   395        
    ##  3rd Qu.:0                    3rd Qu.: 10000   3rd Qu.:     0        
    ##  Max.   :0                    Max.   :300000   Max.   :200000        
    ##  NA's   :3078                 NA's   :1447     NA's   :1447          
    ##   valeur_gage      valeur_residence_principale valeur_bien_immobilier
    ##  Min.   :      0   Min.   :      0             Min.   :     0.0      
    ##  1st Qu.:      0   1st Qu.:      0             1st Qu.:     0.0      
    ##  Median :      0   Median :      0             Median :     0.0      
    ##  Mean   :  31283   Mean   :  30209             Mean   :   326.6      
    ##  3rd Qu.:      0   3rd Qu.:      0             3rd Qu.:     0.0      
    ##  Max.   :3000000   Max.   :3000000             Max.   :800000.0      
    ##                                                                      
    ##  valeur_produit_epargne_placement valeur_bien_mobilier valeur_totale    
    ##  Min.   :   0.0000                Min.   :    0.0      Min.   :      0  
    ##  1st Qu.:   0.0000                1st Qu.:    0.0      1st Qu.:      0  
    ##  Median :   0.0000                Median :    0.0      Median :      0  
    ##  Mean   :   0.9238                Mean   :  677.6      Mean   :  29292  
    ##  3rd Qu.:   0.0000                3rd Qu.:    0.0      3rd Qu.:      0  
    ##  Max.   :5000.0000                Max.   :75000.0      Max.   :3000000  
    ##                                                                         
    ##  valeur_acquisition nb_etablissement_ficp nombre_rejets    
    ##  Min.   :     0     Min.   :0             Min.   : 0.0000  
    ##  1st Qu.:     0     1st Qu.:0             1st Qu.: 0.0000  
    ##  Median :     0     Median :0             Median : 0.0000  
    ##  Mean   :  8444     Mean   :0             Mean   : 0.2815  
    ##  3rd Qu.:     0     3rd Qu.:0             3rd Qu.: 0.0000  
    ##  Max.   :640000     Max.   :0             Max.   :45.0000  
    ##  NA's   :2342                                              
    ##  nombre_commissions_intervention     rc_id         csp_emp      horizon_rdv    
    ##  Min.   : 0.00000                Min.   : 146   Min.   :2140   Min.   : 0.000  
    ##  1st Qu.: 0.00000                1st Qu.:1796   1st Qu.:5217   1st Qu.: 3.000  
    ##  Median : 0.00000                Median :2424   Median :6312   Median : 5.000  
    ##  Mean   : 0.05847                Mean   :2188   Mean   :6239   Mean   : 4.963  
    ##  3rd Qu.: 0.00000                3rd Qu.:2656   3rd Qu.:7700   3rd Qu.: 6.000  
    ##  Max.   :50.00000                Max.   :3070   Max.   :8600   Max.   :99.000  
    ##                                  NA's   :4179   NA's   :7135                   
    ##     age_emp       horizon_retraite_emp
    ##  Min.   :  0.00   Min.   :-2023.00    
    ##  1st Qu.: 44.60   1st Qu.:   -2.90    
    ##  Median : 54.50   Median :    9.20    
    ##  Mean   : 54.89   Mean   :  -12.89    
    ##  3rd Qu.: 66.20   3rd Qu.:   19.10    
    ##  Max.   :126.70   Max.   :   64.90    
    ##  NA's   :47       NA's   :483

``` r
nb_lignes_avant = nrow(data)

data = data |>
  filter(
    (age_emp >= 18 & age_emp <= 95) | is.na(age_emp),
    (anciennete_emp <= 60) | is.na(anciennete_emp),
    (valeur_residence_principale <= 10000000 | is.na(valeur_residence_principale)),
    (valeur_gage <= 10000000 | is.na(valeur_gage)),
    (valeur_totale <= 10000000 | is.na(valeur_totale)),
    (horizon_retraite_emp >= -10 & horizon_retraite_emp <= 50) | is.na(horizon_retraite_emp)
  )

cat("Observations aberrantes supprimées :", nb_lignes_avant - nrow(data), "\n")
```

    ## Observations aberrantes supprimées : 724

``` r
cat("Dimension actuelle :", dim(data), "\n")
```

    ## Dimension actuelle : 6853 91

------------------------------------------------------------------------

## 4. Feature Engineering et aplatissement des listes

### 4.1 Traitement des colonnes de listes (crédits conso et immo)

On supprime les listes textuelles non exploitables, on extrait une
indicatrice de retard (`presence_retard`), puis on résume chaque liste
financière par son nombre de crédits et la somme des montants.

``` r
colonnes_listes_texte = c(
  "conso_type", "conso_libelle", "conso_conserve", 
  "conso_date_debut", "conso_date_fin", "conso_duree", "conso_taux",
  "immo_garantie", "immo_libelle", "immo_conserve", 
  "immo_date_debut", "immo_date_fin", "immo_duree", "immo_taux"
)

extraire_nombres = function(x) {
  if (is.na(x) || x == "[]" || x == "") return(numeric(0))
  as.numeric(unlist(regmatches(x, gregexpr("[0-9]+\\.?[0-9]*", x))))
}

data = data |>
  select(-any_of(colonnes_listes_texte)) |>
  mutate(
    retard_conso = sapply(conso_echeance_de_retard, function(x) any(extraire_nombres(x) > 0, na.rm = TRUE)),
    retard_immo  = sapply(immo_echeance_de_retard, function(x) any(extraire_nombres(x) > 0, na.rm = TRUE)),
    presence_retard = retard_conso | retard_immo,
    
    nb_conso_mensualite    = sapply(conso_mensualite, function(x) length(extraire_nombres(x))),
    somme_conso_mensualite = sapply(conso_mensualite, function(x) sum(extraire_nombres(x))),
    nb_conso_crd           = sapply(conso_crd, function(x) length(extraire_nombres(x))),
    somme_conso_crd        = sapply(conso_crd, function(x) sum(extraire_nombres(x))),
    
    nb_immo_mensualite     = sapply(immo_mensualite, function(x) length(extraire_nombres(x))),
    somme_immo_mensualite  = sapply(immo_mensualite, function(x) sum(extraire_nombres(x))),
    nb_immo_crd            = sapply(immo_crd, function(x) length(extraire_nombres(x))),
    somme_immo_crd         = sapply(immo_crd, function(x) sum(extraire_nombres(x)))
  ) |>
  select(
    -retard_conso, -retard_immo, -conso_echeance_de_retard, -immo_echeance_de_retard,
    -conso_mensualite, -conso_crd, -immo_mensualite, -immo_crd
  )
```

### 4.2 Agrégations financières et temporelles

``` r
data = data |>
  mutate(
    # 1. Indicateur global d'incident bancaire / dette
    a_incident = case_when(
      (retard_loyer_emp > 0 | retard_impot_emp > 0 | dette_famille_ami_emp > 0 | 
       autre_dette_emp > 0 | saisie_sur_salaire_emp > 0 | avis_a_tiers_detenteurs_emp > 0 | 
       nombre_rejets > 0 | nombre_commissions_intervention > 0) ~ 1,
      is.na(retard_loyer_emp) | is.na(retard_impot_emp) | is.na(dette_famille_ami_emp) | 
      is.na(autre_dette_emp) | is.na(saisie_sur_salaire_emp) | is.na(avis_a_tiers_detenteurs_emp) | 
      is.na(nombre_rejets) | is.na(nombre_commissions_intervention) ~ NA_real_,
      TRUE ~ 0
    ),
    
    # 2. Agrégation des revenus, charges, dettes et taux d'endettement
    charges_tot = loyer_emp + charges_loyer_emp + pension_versee_emp + charge_recurrente_emp + charge_courante_emp,
    revenu_tot = salaire_emp + rev_foncier_emp + allocation_familiale_emp + apl_emp + pension_alimentaire_emp + pension_invalidite_emp,
    dettes_annexes = (retard_impot_emp + retard_loyer_emp + dette_famille_ami_emp + autre_dette_emp) > 0,
    decouvert_emp = decouvert_emp > 0,
    taux_endettement = (somme_conso_mensualite + somme_immo_mensualite) / revenu_tot,
    taux_endettement = ifelse(is.infinite(taux_endettement), NA_real_, taux_endettement),
    
    # 3. Variables contextuelles et cible binaire Y
    contrat_mariage_emp = ifelse(is.na(contrat_mariage_emp), 0, 1),
    heure_rdv_debut = as.numeric(substr(as.character(heure_rdv_debut), 1, 2)),
    jour_semaine_prise_rdv = wday(ymd(date_aboutisant_azur), label = TRUE, abbr = FALSE, week_start = 1),
    Y = ifelse(etat == "annuler client", 1, 0)
  ) |>
  select(
    -etat,
    -retard_loyer_emp, -retard_impot_emp, -dette_famille_ami_emp, -autre_dette_emp,
    -saisie_sur_salaire_emp, -avis_a_tiers_detenteurs_emp, -nombre_rejets, -nombre_commissions_intervention,
    -loyer_emp, -charges_loyer_emp, -pension_versee_emp, -charge_recurrente_emp, -charge_courante_emp,
    -salaire_emp, -rev_foncier_emp, -allocation_familiale_emp, -apl_emp, -pension_alimentaire_emp, -pension_invalidite_emp
  )
```

------------------------------------------------------------------------

## 5. Suppression des variables non pertinentes ou fuitées (Data Leakage)

On retire en une seule étape : \* Les variables renseignées après $t_0$
(`statut_final`, `type_traitement_dossier`, `affectation`,
`frais_notaire`, etc.). \* Les identifiants techniques (`id_dossier`,
`emp_id`, `rc_id`). \* Les variables constantes ou trop peu renseignées
(`type_dossier`, fichages, `csp_emp`, etc.).

``` r
data = data |>
  select(
    # Post-t0 / risque de Data Leakage
    -statut_final, -type_traitement_dossier, -affectation, -commentaire,
    -nature_de_projet, -duree, -penalite_remboursement, -frais_notaire,
    -consentement_emp, -accept_emailing, -prospect_sms,
    
    # Identifiants, dates brutes et variables écartées
    -id_dossier, -emp_id, -rc_id, -date_acquisition, -valeur_acquisition,
    -date_creation, -date_aboutisant_azur, -date_rdv, -cp_emp, -csp_emp,
    -type_support, -profession_emp, -valeur_gage, -valeur_bien_immobilier,
    -local_agence, -tresorerie_sur_facture,
    
    # Variables constantes
    -type_dossier, -conge_parental_emp, -nb_etablissement_ficp,
    -charge_future_eventuelle_emp, -fichage_FICP, -fichage_FCC_cheque, -fichage_FCC_carte
  )

cat("Dimension après sélection des variables :", dim(data), "\n")
```

    ## Dimension après sélection des variables : 6853 33

------------------------------------------------------------------------

## 6. Traitement final des valeurs manquantes et de la colinéarité

### 6.1 Discrétisation de `tresorerie` (vérification d’absence de Target Leakage)

``` r
data = data |>
  mutate(
    statut_tresorerie = case_when(
      is.na(tresorerie) ~ "Non_renseigne",
      tresorerie == 0   ~ "Nulle",
      tresorerie > 0    ~ "Positive",
      tresorerie < 0    ~ "Negative"
    ) |> as.factor()
  )

data |>
  group_by(statut_tresorerie) |>
  summarise(effectif = n(), taux_cible = mean(Y == 1, na.rm = TRUE))
```

    ## # A tibble: 3 × 3
    ##   statut_tresorerie effectif taux_cible
    ##   <fct>                <int>      <dbl>
    ## 1 Non_renseigne         1308      0.162
    ## 2 Nulle                  861      0.151
    ## 3 Positive              4684      0.136

``` r
data = data |> select(-tresorerie)
```

### 6.2 Imputation croisée de `contrat_emp` et colinéarité `age_emp` / `horizon_retraite_emp`

On utilise `horizon_retraite_emp` et `age_emp` pour imputer la modalité
`"RETRAITE"` sur `contrat_emp`, puis on vérifie la colinéarité entre
l’âge et l’horizon de retraite avant de retirer `horizon_retraite_emp`.

``` r
data = data |>
  mutate(
    contrat_emp = as.character(contrat_emp),
    contrat_emp = case_when(
      (is.na(contrat_emp) | contrat_emp == "Non_renseigne") & 
        (horizon_retraite_emp <= 0 | age_emp >= 65) ~ "RETRAITE",
      is.na(contrat_emp) ~ "Non_renseigne",
      TRUE ~ contrat_emp
    ) |> as.factor(),
    horizon_retraite_emp = case_when(
      is.na(horizon_retraite_emp) & !is.na(age_emp) ~ 64 - age_emp,
      TRUE ~ horizon_retraite_emp
    )
  )

cat("Corrélation entre age_emp et horizon_retraite_emp :", 
    cor(data$age_emp, data$horizon_retraite_emp, use = "complete.obs"), "\n")
```

    ## Corrélation entre age_emp et horizon_retraite_emp : -0.9934438

``` r
data = data |> select(-horizon_retraite_emp)
```

### 6.3 Nettoyage des dernières valeurs manquantes et export

``` r
# Proportion de NA par variable restante
data |>
  summarise(across(everything(), ~ mean(is.na(.x)))) |>
  pivot_longer(everything(), names_to = "variable", values_to = "prop_na") |>
  arrange(desc(prop_na)) |>
  head(10)
```

    ## # A tibble: 10 × 2
    ##    variable                  prop_na
    ##    <chr>                       <dbl>
    ##  1 charges_tot               0.403  
    ##  2 taux_endettement          0.0136 
    ##  3 situation_fam_emp         0.00744
    ##  4 age_emp                   0.00686
    ##  5 decouvert_emp             0.00161
    ##  6 a_incident                0.00161
    ##  7 dettes_annexes            0.00161
    ##  8 intitule_emp              0      
    ##  9 assistante_maternelle_emp 0      
    ## 10 anciennete_emp            0

``` r
# Suppression des quelques lignes ayant des NA résiduels (hors charges_tot qui a ~40% de NA)
data = data |>
  drop_na(-charges_tot) |>
  select(-charges_tot)

cat("Total de NA restants :", sum(is.na(data)), "\n")
```

    ## Total de NA restants : 0

``` r
cat("Dimension finale :", dim(data), "\n")
```

    ## Dimension finale : 6743 31

``` r
write.csv(data, "Data_nettoye2.csv", row.names = FALSE)
```
