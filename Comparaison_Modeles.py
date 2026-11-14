#%%
import numpy as np
import pandas as pd
from sklearn import preprocessing # Pour le prétraitement et la normalisation des données
from sklearn.model_selection import train_test_split # Pour diviser le jeu de données en train et test
from sklearn.linear_model import LogisticRegression # Pour l'implémentation du modèle de régression logistique
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline

# %%
data = pd.read_csv("Data_nettoye2.csv")
data.head()
#%%
# =========================================================================
# Partie 1 : preparations des datasets pour les modeles
# =========================================================================
#%%
X = data.loc[:, data.columns != 'Y']
y = data['Y']
pd.Series(y).value_counts()

#%% 
# 1. Premier split : isoler le Test final (20% du total)
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Second split sur les 80% restants : 75% Train (60% total) / 25% Val (20% total)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
)

print(f"Tailles -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
# %%
# 3. Prétraitement appris UNIQUEMENT sur X_train (60%)
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop=None, handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep   = preprocessor.transform(X_val)
X_test_prep  = preprocessor.transform(X_test)
# %%
# 4. Conversion en tenseurs PyTorch pour les 3 jeux
X_train_t = torch.tensor(X_train_prep, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_prep, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_prep, dtype=torch.float32)

y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_val_t   = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
y_test_t  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Poids de la classe positive calculé uniquement sur y_train
pos_weight_value = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
# %%
# =========================================================================
# Partie 2 : création des differents modeles
# =========================================================================
#%%
# =========================================================================
# Regressions logistiques
# =========================================================================
#%%
import time

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    classification_report
)

#%%

def evaluer_modele(nom, y_true, y_proba, seuil=0.5, duree=0.0):
    y_pred = (y_proba >= seuil).astype(int)
    pr_auc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    
    print(f"=== {nom} (Temps: {duree:.4f}s | Seuil: {seuil:.3f}) ===")
    print(f"PR-AUC (AUPRC) : {pr_auc:.4f} | F1-score : {f1:.4f}")
    print("Matrice de confusion (Test) :")
    print(pd.crosstab(y_true, y_pred, rownames=['Réel'], colnames=['Prédit']))
    print("-" * 55)
#%%
# ==============================================================================
# MODÈLE 1 : Régression Logistique standard (sans pondération, L2 par défaut)
# ==============================================================================
t0 = time.perf_counter()
logit_base = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
logit_base.fit(X_train_prep, y_train)
t_base = time.perf_counter() - t0

proba_test_1 = logit_base.predict_proba(X_test_prep)[:, 1]
evaluer_modele("Modèle 1 : Base (non pondéré)", y_test, proba_test_1, seuil=0.5, duree=t_base)
#%%
# ==============================================================================
# MODÈLE 2 : Régression Logistique avec class_weight="balanced"
# ==============================================================================
t0 = time.perf_counter()
logit_bal = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
logit_bal.fit(X_train_prep, y_train)
t_bal = time.perf_counter() - t0

proba_test_2 = logit_bal.predict_proba(X_test_prep)[:, 1]
evaluer_modele("Modèle 2 : Balanced (seuil 0.5)", y_test, proba_test_2, seuil=0.5, duree=t_bal)
#%%
# ==============================================================================
# MODÈLE 3 : Optimisation Hyperparamètres (C, L1/L2) + Seuil sur Validation
# ==============================================================================
t0 = time.perf_counter()

grille_C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
grille_penalty = ['l1', 'l2']

meilleur_score_val = -1.0
meilleur_modele = None
meilleurs_params = None

# 1. Recherche des meilleurs hyperparamètres sur X_val_prep (critère : PR-AUC)
for pen in grille_penalty:
    for c_val in grille_C:
        # solver='liblinear' gère à la fois L1 et L2 avec class_weight='balanced'
        clf = LogisticRegression(
            penalty=pen, C=c_val, solver='liblinear',
            class_weight="balanced", max_iter=1000, random_state=42
        )
        clf.fit(X_train_prep, y_train)
        
        proba_val = clf.predict_proba(X_val_prep)[:, 1]
        score_val = average_precision_score(y_val, proba_val)
        
        if score_val > meilleur_score_val:
            meilleur_score_val = score_val
            meilleur_modele = clf
            meilleurs_params = (pen, c_val)

# 2. Optimisation du seuil de décision sur X_val_prep (critère : F1-score max)
proba_val_best = meilleur_modele.predict_proba(X_val_prep)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, proba_val_best)

# Calcul du F1 pour chaque seuil (en évitant la division par zéro)
f1_scores = np.divide(
    2 * precisions[:-1] * recalls[:-1],
    precisions[:-1] + recalls[:-1],
    out=np.zeros_like(thresholds),
    where=(precisions[:-1] + recalls[:-1]) != 0
)
meilleur_seuil = thresholds[np.argmax(f1_scores)]

t_opt = time.perf_counter() - t0

# 3. Évaluation finale unique sur X_test_prep
print(f"Meilleurs hyperparamètres trouvés sur Val : penalty={meilleurs_params[0]}, C={meilleurs_params[1]} (PR-AUC Val = {meilleur_score_val:.4f})")
proba_test_3 = meilleur_modele.predict_proba(X_test_prep)[:, 1]
evaluer_modele("Modèle 3 : Balanced + Optimisé", y_test, proba_test_3, seuil=meilleur_seuil, duree=t_opt)

# %%
# ==============================================================================
# Forets aleatoires
# ==============================================================================
# %%
# ==============================================================================
# 1. Modele 1 : RF Base (paramètres par défaut, sans pondération)
# ==============================================================================
t0 = time.perf_counter()
rf_base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_base.fit(X_train_prep, y_train)
evaluer_modele("RF 1 : Base", y_test, rf_base.predict_proba(X_test_prep)[:, 1], 0.5, time.perf_counter() - t0)
#%%
# ==============================================================================
# Modele 2 : RF Balanced (pondération des classes, seuil 0.5)
# ==============================================================================
t0 = time.perf_counter()
rf_bal = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
rf_bal.fit(X_train_prep, y_train)
evaluer_modele("RF 2 : Balanced", y_test, rf_bal.predict_proba(X_test_prep)[:, 1], 0.5, time.perf_counter() - t0)

#%%
# ==============================================================================
# Modele 3 : RF Optimisée (régularisation des arbres + seuil calibré sur Val)
# ==============================================================================
t0 = time.perf_counter()
best_score, best_rf, best_params = -1, None, None

for depth in [8, 14, None]:
    for min_leaf in [2, 10, 20]:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=depth, min_samples_leaf=min_leaf,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_train_prep, y_train)
        score = average_precision_score(y_val, rf.predict_proba(X_val_prep)[:, 1])
        if score > best_score:
            best_score, best_rf, best_params = score, rf, (depth, min_leaf)

# Optimisation du seuil F1 sur Val
prec, rec, thresh = precision_recall_curve(y_val, best_rf.predict_proba(X_val_prep)[:, 1])
f1 = np.divide(2 * prec[:-1] * rec[:-1], prec[:-1] + rec[:-1], out=np.zeros_like(thresh), where=(prec[:-1] + rec[:-1]) != 0)
best_seuil = thresh[np.argmax(f1)]

print(f"Meilleurs params Val : max_depth={best_params[0]}, min_samples_leaf={best_params[1]}")
evaluer_modele("RF 3 : Optimisée", y_test, best_rf.predict_proba(X_test_prep)[:, 1], best_seuil, time.perf_counter() - t0)

# %%
# ==============================================================================
# RDN
# ==============================================================================
#%%
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
#%%
# 1. Reproductibilité et choix du matériel (MPS sur Mac Apple Silicon, sinon CPU)
torch.manual_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#%%
# 2. Architecture compacte adaptée au tabulaire
class TabularMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

rdn = TabularMLP(X_train_t.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value.to(device))
optimizer = torch.optim.Adam(rdn.parameters(), lr=1e-3, weight_decay=1e-4)

#%%
# 3. Entraînement avec Early Stopping sur X_val_t
t0 = time.perf_counter()
best_val_loss = float("inf")
best_weights = copy.deepcopy(rdn.state_dict())
patience, compteur = 10, 0

X_val_dev = X_val_t.to(device)
y_val_dev = y_val_t.to(device)

for epoch in range(1, 101):
    rdn.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(rdn(xb), yb)
        loss.backward()
        optimizer.step()

    # Évaluation sur Validation uniquement
    rdn.eval()
    with torch.no_grad():
        val_loss = criterion(rdn(X_val_dev), y_val_dev).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(rdn.state_dict())
        compteur = 0
    else:
        compteur += 1
        if compteur >= patience:
            print(f"Early stopping à l'époque {epoch} (meilleure val_loss = {best_val_loss:.4f})")
            break

# Restauration des meilleurs poids vus sur Val
rdn.load_state_dict(best_weights)

#%%
# 4. Calibrage du seuil de décision sur Val + Prédiction finale sur Test
rdn.eval()
with torch.no_grad():
    proba_val_rdn = torch.sigmoid(rdn(X_val_dev)).cpu().numpy().ravel()
    proba_test_rdn = torch.sigmoid(rdn(X_test_t.to(device))).cpu().numpy().ravel()

prec, rec, thresh = precision_recall_curve(y_val, proba_val_rdn)
f1 = np.divide(2 * prec[:-1] * rec[:-1], prec[:-1] + rec[:-1], out=np.zeros_like(thresh), where=(prec[:-1] + rec[:-1]) != 0)
seuil_rdn = thresh[np.argmax(f1)]

t_rdn = time.perf_counter() - t0
evaluer_modele("RDN : PyTorch MLP", y_test, proba_test_rdn, seuil_rdn, t_rdn)
# %%
# =========================================================================
# Comparaison finale des 3 meilleurs modèles sur le jeu de Test
# =========================================================================
# %% 
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

proba_test_rf3 = best_rf.predict_proba(X_test_prep)[:, 1]

modeles_finaux = [
    ("Modèle 3 : Balanced + Optimisé", proba_test_3, meilleur_seuil),
    ("RF 3 : Optimisée", proba_test_rf3, best_seuil),
    ("RDN : PyTorch MLP", proba_test_rdn, seuil_rdn)
]

resultats = []
for nom, proba, seuil in modeles_finaux:
    y_pred = (proba >= seuil).astype(int)
    resultats.append({
        "Modèle": nom,
        "AUPRC": average_precision_score(y_test, proba),
        "Précision": precision_score(y_test, y_pred, zero_division=0),
        "Rappel": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "Seuil (Val)": seuil
    })

df_comparaison = pd.DataFrame(resultats).round(4)
print(df_comparaison.to_string(index=False))
# %%
