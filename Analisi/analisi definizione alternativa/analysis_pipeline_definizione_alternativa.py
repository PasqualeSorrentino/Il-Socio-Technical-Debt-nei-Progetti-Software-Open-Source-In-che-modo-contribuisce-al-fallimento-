"""
Socio-Technical Debt & OSS Project Failure
=============================================================
Variabile dipendente: ACTIVE vs FAILED

Definizione:
    Un progetto è inattivo se ha avuto in media < 1
    commit/mese nei 12 mesi precedenti al suo commit più recente.
"""

import pandas as pd
import numpy as np
import json
import warnings
import matplotlib; matplotlib.use('Agg') # salva PNG evitando che si aprano le finestre durante l'analisi
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore') # silenzia i warning

from scipy import stats as _ss
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text
from lifelines import CoxTimeVaryingFitter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent

DATA_PATH = ROOT_DIR / "Analisi" / "dataset" / "mining.csv"
OUT_DIR = BASE_DIR / "outputs" / "file"
OUT_DIR.mkdir(exist_ok=True, parents=True)

INACTIVITY_THRESHOLD = 4    # >= 4 finestre (~12 mesi) = FAILED
EARLY_STAGE_WINDOWS  = 3    # prime 3 finestre = early stage
VIF_THRESHOLD        = 5.0  # Variance Inflation Factor soglia di rimozione feature > 5
OUTLIER_QUANTILE     = 0.99 # cap al 99° percentile

def cliffs_delta(a, b):
    """Effect size non parametrico (-1..+1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a) * len(b)
    if n == 0: return np.nan
    return (sum(np.sum(x > b) for x in a) - sum(np.sum(x < b) for x in a)) / n

def eff_label(d):
    ad = abs(d)
    return 'trascurabile' if ad < .147 else 'piccolo' if ad < .33 else 'medio' if ad < .474 else 'grande'

def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if n == 0 or x.sum() == 0: return np.nan
    return (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n

def youden(y, x):
    fpr, tpr, thr = roc_curve(y, x); j = tpr - fpr; i = j.argmax()
    return thr[i], auc(fpr, tpr), tpr[i], 1 - fpr[i]

EMO = ['Anger', 'Fear', 'Sadness', 'Love', 'Joy', 'Surprise']


# ─────────────────────────────────────────────
# STEP 1 — CARICAMENTO
# ─────────────────────────────────────────────

print("STEP 1 — Caricamento dati")


df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Shape: {df.shape[0]:,} righe × {df.shape[1]} colonne")
print(f"Progetti: {df['project_url'].nunique():,}")
print(f"Developer: {df['developer_id'].nunique():,}")
print(f"Finestre: {df['time_window_id'].nunique()} "
      f"({df['time_window_id'].min()} -> {df['time_window_id'].max()})")


# Risultato analisi:
# Shape: 24.916 righe × 76 colonne
# Progetti: 332
# Developer: 10.416
# Finestre: 154 (20070601_20070831 -> 20250701_20250930)
# 
# Possibilità teoriche:
# 10.416 developer su 154 finestre darebbero generare oltre 1,6 milioni di righe se tutti i developer fossero attivi in tutte le finestre.
# Avere solo 24.916 righe significa che ogni developer compare in pochissime finestre temporali (in media: ~2.4) -> la maggioranza sono contributori "mordi e fuggi"


# ─────────────────────────────────────────────
# STEP 2 — ESTRAZIONE STQF DA JSON
# ─────────────────────────────────────────────
print("\n")
print("STEP 2 — Estrazione Socio-Technical Quality Factors")

"Evita errori di parsing JSON e valori NaN"
def safe_json(s):
    try:
        if pd.isna(s) or str(s).strip() in ('', '{}'): return {}
        return json.loads(s)
    except: return {}


# La colonna grezza Socio-Technical Quality Factors contiene, in ogni cella, un oggetto JSON con 34 chiavi.
# Ne sono state selezionate 20 per l'analisi, che vengono estratte in colonne separate.
# Le 14 chiavi rimanenti non sono state considerate per il focus dell'analisi o per la ridondanza con altre feature già presenti nel dataset.
# E sono: ml.code.devs, perc.ml.only.devs, perc.code.only.devs, perc.ml.code.devs, sponsored.devs, core.mail.devs,
# sponsored.core.devs, ratio.sponsored.core, mail.truck, code.truck, code.turnover, core.global.turnover, core.mail.turnover, mail.mod

STQF_KEYS = [
    'devs', 'ml.only.devs', 'code.only.devs',
    'ratio.smelly.devs', 'global.turnover', 'core.code.turnover',
    'global.truck', 'density', 'st.congruence', 'communicability',
    'num.tz', 'ratio.smelly.quitters', 'ratio.sponsored',
    'closeness.centr', 'betweenness.centr', 'degree.centr',
    'global.mod', 'code.mod', 'core.global.devs', 'core.code.devs'
]

stqf_parsed = df['Socio-Technical Quality Factors'].apply(safe_json)
stqf_df = pd.DataFrame(
    [{k: d.get(k, np.nan) for k in STQF_KEYS} for d in stqf_parsed]
)
stqf_df.columns = ['stqf_' + c.replace('.', '_') for c in stqf_df.columns]    #viene sostituito il punto con underscore per evitare problemi di analisi
df = pd.concat([df.reset_index(drop=True), stqf_df.reset_index(drop=True)], axis=1)
print(f"Feature STQF estratte: {len(stqf_df.columns)}")

# ─────────────────────────────────────────────
# STEP 3 — VARIABILE DIPENDENTE BINARIA
# ─────────────────────────────────────────────
print("\n")
print("STEP 3 — Variabile dipendente: ACTIVE vs FAILED")


# Ranking temporale delle finestre
all_windows     = sorted(df['time_window_id'].astype(str).unique())
window_rank_map = {w: i for i, w in enumerate(all_windows)}         # ordina le finestre e assegna un rango 0..153
last_rank       = window_rank_map[all_windows[-1]]                  # rango dell'ultima finestra del dataset

df['time_window_rank']      = df['time_window_id'].astype(str).map(window_rank_map)
df['last_interaction_rank'] = df['last_interaction_window_id'].astype(str).map(window_rank_map)

# Inactivity lag: quante finestre fa ha smesso di contribuire
df['inactivity_lag'] = last_rank - df['last_interaction_rank']


# has_failed = 1 se inattivo da 4 trimestri consecutivi (~12 mesi)
# has_failed = 0 se ancora attivo (interazione presente negli ultimi 12 mesi)
df['has_failed'] = (df['inactivity_lag'] >= INACTIVITY_THRESHOLD).astype(int)

print(f"Soglia adottata: >= {INACTIVITY_THRESHOLD} finestre = ~{INACTIVITY_THRESHOLD*3} mesi di inattività")
print(f"\nDistribuzione has_failed:")
vc = df['has_failed'].value_counts()
print(f"  ACTIVE (0): {vc.get(0,0):6,} osservazioni  ({vc.get(0,0)/len(df)*100:.1f}%)")
print(f"  FAILED (1): {vc.get(1,0):6,} osservazioni  ({vc.get(1,0)/len(df)*100:.1f}%)")

# A livello developer unico
dev_status = df.groupby('developer_id')['has_failed'].max()
print(f"\nDeveloper unici:")
print(f"  ACTIVE: {(dev_status == 0).sum():,}  ({(dev_status==0).mean()*100:.1f}%)")
print(f"  FAILED: {(dev_status == 1).sum():,}  ({(dev_status==1).mean()*100:.1f}%)")

# A livello progetto: FAILED se tutti i developer sono failed
proj_status = df.groupby('project_url')['has_failed'].min()
# min=1 solo se TUTTI i developer sono failed -> progetto completamente fallito
# min=0 se almeno uno è ancora attivo -> progetto attivo
print(f"\nProgetti unici:")
print(f"  ACTIVE (almeno 1 dev attivo): {(proj_status == 0).sum():,}")
print(f"  FAILED (tutti i dev failed):  {(proj_status == 1).sum():,}")

# Risultato analisi (con censura a destra + definizione progetto v2):
# Distribuzione has_failed (developer, censura a destra):
#   CENSURATO (0):  7.424  (29,8%)
#   EVENTO    (1): 17.492  (70,2%)
# 
# Developer unici:   EVENTO 8.984 (86,3%)  |  CENSURATO 1.432 (13,7%)
# Progetti (v2, <1 commit/mese ultimo anno):  INATTIVI 166  |  ATTIVI 166


# ═══════════════════════════════════════════════════════════════════════════
# CENSURA A DESTRA 
# has_failed = 1 -> EVENTO (abbandono): inattivo >= soglia PRIMA della fine
#                   del PROPRIO progetto (non rispetto al cutoff globale).
# has_failed = 0 -> CENSURATO: ancora attivo entro la soglia dall'orizzonte.
# obs_duration   -> durata a rischio (finestre) per i modelli di sopravvivenza.
# proj_failed    -> a livello progetto: FALLITO se morto prima del cutoff.
# ═══════════════════════════════════════════════════════════════════════════
GLOBAL_LAST = last_rank
_proj_last = df.groupby('project_url')['time_window_rank'].transform('max')
_dev_first = df.groupby(['project_url', 'developer_id'])['time_window_rank'].transform('min')
_dev_last  = df.groupby(['project_url', 'developer_id'])['last_interaction_rank'].transform('max')

df['inactivity_lag'] = _proj_last - _dev_last                                   # rispetto all'orizzonte del progetto
df['has_failed']  = (df['inactivity_lag'] >= INACTIVITY_THRESHOLD).astype(int)  # 1=EVENTO, 0=CENSURATO
df['obs_duration'] = np.where(df['has_failed'] == 1,
                              _dev_last - _dev_first,
                              _proj_last - _dev_first) + 1
_alive = (GLOBAL_LAST - _proj_last) < INACTIVITY_THRESHOLD                       # progetto ancora vivo al cutoff?
df['censor_type'] = np.where(df['has_failed'] == 1, 'evento',
                    np.where(_alive, 'admin', 'fine_progetto'))
# proj_failed (DEFINIZIONE v2 = PRINCIPALE): progetto INATTIVO se ha avuto in media < 1 commit/mese
# nei 12 mesi (4 finestre) fino al suo commit piu' recente. Piu' robusta all'attivita' residua della
# soglia temporale (cattura "non piu' mantenuto/spento" e non "fermo da tempo": robusta ai commit di
# deprecation, ai progetti "completati" e ai cambi di proprieta').
_cc3 = pd.to_numeric(df['commits_count'], errors='coerce').fillna(0)
_cpw3 = df.assign(_cc=_cc3).groupby(['project_url', 'time_window_rank'])['_cc'].sum().reset_index()
_pf3 = {}
for _pj, _g in _cpw3.groupby('project_url'):
    _last = _g['time_window_rank'].max()
    _trail = _g[(_g['time_window_rank'] >= _last - 3) & (_g['time_window_rank'] <= _last)]['_cc'].sum()
    _months = min(4, _g['time_window_rank'].nunique()) * 3
    _pf3[_pj] = int((_trail / _months) < 1)
df['proj_failed'] = df['project_url'].map(_pf3).astype(int)

# Estrazione delle 6 emozioni dal JSON grezzo (necessarie allo STEP 14)
_emo = df['developer_raw_json'].apply(safe_json).apply(lambda d: (d.get('sentiment_emotions') or {}))
for e in EMO:
    df['emo_' + e] = pd.to_numeric(_emo.apply(lambda d: d.get(e, np.nan)), errors='coerce')

_vc = df['has_failed'].value_counts()
print("\n[CENSURA A DESTRA] esito ricalcolato e usato da tutti gli step:")
print(f"  CENSURATO (0, attivo): {_vc.get(0,0):6,} ({_vc.get(0,0)/len(df)*100:.1f}%)  |  "
      f"EVENTO (1, abbandono): {_vc.get(1,0):6,} ({_vc.get(1,0)/len(df)*100:.1f}%)")
_ps = df.groupby('project_url')['proj_failed'].max()
print(f"  Progetti (v2, <1 commit/mese ultimo anno): INATTIVI {(_ps==1).sum()} | ATTIVI {(_ps==0).sum()}")
print(f"  Durata a rischio (mediana): {df['obs_duration'].median():.0f} finestre")


# ─────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING E PREPROCESSING
# ─────────────────────────────────────────────
print("\n")
print("STEP 4 — Feature engineering e preprocessing")

# Estrazione nuove variabili: project age e developer tenure
df = df.sort_values(['project_url', 'developer_id', 'time_window_rank']).reset_index(drop=True)

# ordinamento del panel e calcola due variabili temporali ossia l'età del progetto e la tenure del developer — come distanza (in finestre) dalla prima apparizione.
proj_first = (df.groupby('project_url')['time_window_rank']
                .min().rename('proj_first_rank'))
dev_first  = (df.groupby(['project_url','developer_id'])['time_window_rank']
                .min().rename('dev_first_rank'))
df = df.merge(proj_first, on='project_url', how='left')
df = df.merge(dev_first,  on=['project_url','developer_id'], how='left')

df['project_age']     = df['time_window_rank'] - df['proj_first_rank']
df['developer_tenure']= df['time_window_rank'] - df['dev_first_rank']

# Lista completa delle feature
FEATURES_BASE = [
    # Social Debt 16 feature
    'community_smell_count',
    'stqf_ratio_smelly_devs',
    'stqf_global_turnover',
    'stqf_core_code_turnover',
    'stqf_global_truck',
    'stqf_density',
    'stqf_st_congruence',
    'stqf_communicability',
    'stqf_num_tz',
    'stqf_ratio_smelly_quitters',
    'stqf_ratio_sponsored',
    'stqf_devs',
    'stqf_closeness_centr',
    'stqf_degree_centr',
    'stqf_core_global_devs',
    'stqf_core_code_devs',
    # Technical Debt 5 feature
    'traditional_smell_count',
    'ml_smell_count',
    'vulnerability_count',
    'vulnerability_high',
    'bug_introduced_count_rszz',
    # Attività (controllo) 4 feature
    'commits_count',
    'code_churn',
    'bug_fix_commits_count',
    'files_touched_count',
    # Developer (controllo) 2 feature
    'se_score',
    'sentiment_score',
    # Temporali (controllo) 2 feature
    'project_age',
    'developer_tenure',
]
FEATURES_BASE = [f for f in FEATURES_BASE if f in df.columns] # tiene solo quelle esistenti

# Motivazione selezione feature:
# queste 29 sono le uniche colonne che sono contemporaneamente numeriche e utilizzabili, collegate alle ipotesi dello studio:
# che il debito sociale e il debito tecnico influenzino il fallimento;
# e prive di leakage rispetto all'esito.
# Le 4 variabili di attività, le 2 developer e le 2mtemporali non sono "predittori d'interesse" ma controlli:
# servono a isolare l'effetto del debito da quello di quanto uno lavora e da quanto tempo è nel progetto.


# Le feature escluse sono:

# Variabili di leakage, ossia che costruiscono l'esito (11):
# is_abandoned, abandonment_status, last_interaction_window_id, last_interaction_window_label, abandoned_since_window_id,
# abandoned_since_window_label, abandoned_since_date, project_abandoned_developers_count, project_abandoned_developers_ids,
# più le derivate inactivity_lag e last_interaction_rank create precedentemente.

# Identificatori e metadati temporali (6):
# project_url, developer_id, time_window_id, time_window_label, time_window_start, time_window_end.

# Campi testuali (13):
# aliases, emails, classification, gender_source, pronouns_detected, sentiment_label, last_commit_hash, last_commit_date, last_commit_message,
# last_message_before_abandonment_hash, last_message_before_abandonment_date, last_message_before_abandonment.

#JSON grezzi (10):
# Socio-Technical Quality Factors, developer_raw_json, project_metrics_json, project_community_smells_count_json, project_ml_smells_count_json,
# project_traditional_smells_count_json, project_vulnerabilities_count_json, project_vulnerabilities_severity_count_json,
# project_community_smell_instances_json,project_collaboration_edges_json.

#Ridondanti:
# project_loc, project_nom, i vari project_*_total e project_vulnerabilities_high/medium/low, project_collaboration_edges_count,
# lines_added, lines_deleted, avg_files_per_commit, community_smells/ml_smells/traditional_smells e le loro *_instances, vulnerability_medium, vulnerability_low,
# gender, gender_confidence, ai_score, ml_score, sentiment_messages_count.


# feature engineering: conversione a numerico (NaN -> 0) e rimozione outlier top 1%
for f in FEATURES_BASE:
    df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

# Rimozione outlier top 1%
for col in FEATURES_BASE:
    cap = df[col].quantile(OUTLIER_QUANTILE) #0.99
    if cap > 0:
        df[col] = df[col].clip(upper=cap) #schiacciamo gli outlier sopra il 99° percentile senza rimuovere le righe (per evitare di perdere i developer con più contributi)

# Log-trasformazione variabili skewed
LOG_COLS = [
    'commits_count', 'code_churn', 'traditional_smell_count',
    'community_smell_count', 'bug_introduced_count_rszz', 'files_touched_count'
]
LOG_COLS = [c for c in LOG_COLS if c in df.columns]
for c in LOG_COLS:
    df[f'log_{c}'] = np.log1p(df[c])    # log(1+x): per comprimere le code lunghe, riducendo l'effetto degli outlier e rendendo la distribuzione più simmetrica
# log1p gestisce anche i valori zero senza generare errori.


# Feature finali (sostituisce le log-trasformate)
FEAT_FINAL = (
    [f'log_{c}' for c in LOG_COLS] +
    [f for f in FEATURES_BASE if f not in LOG_COLS]
)
FEAT_FINAL = [f for f in FEAT_FINAL if f in df.columns]
print(f"Feature totali: {len(FEAT_FINAL)}")

# ─────────────────────────────────────────────
# STEP 5 — EDA
# ─────────────────────────────────────────────
print("\n")
print("STEP 5 — EDA e statistiche descrittive")

key_cols = [
    'community_smell_count', 'traditional_smell_count', 'ml_smell_count',
    'vulnerability_count', 'stqf_global_turnover', 'stqf_ratio_smelly_devs',
    'commits_count', 'code_churn', 'stqf_devs', 'stqf_st_congruence'
]
key_cols = [c for c in key_cols if c in df.columns]

desc = (df.groupby('has_failed')[key_cols]
          .agg(['mean', 'median', 'std'])
          .round(3))
print(desc.to_string())
desc.to_csv(OUT_DIR / 'descriptive_stats.csv')

# Risultato analisi:
# metrica                    ACTIVE (0)        FAILED (1)
# community_smell_count      0,733             0,711
# traditional_smell_count    1,381             1,109
# vulnerability_count        0,665             0,508
# commits_count              19,649            9,91
# code_churn                 11.326            6.579

# I developer ACTIVE fanno circa il doppio dei commit e churn dei FAILED
# Le metriche di debito sociale e tecnico sono più alte per gli ACTIVE, ma la differenza non è così marcata come per le attività.
# Parrebbe che il debito sociale e tecnico non sia un fattore discriminante tra ACTIVE e FAILED, ma le attività sì.


# Kaplan-Meier per developer
print("\nCalcolo Kaplan-Meier...")
from lifelines import KaplanMeierFitter
# durata = tenure massima +1, evento = has_failed; stratificazione per dimensione team (mediana)

rows_km = []
for (proj, dev), grp in df.groupby(['project_url', 'developer_id']):
    rows_km.append({
        'duration' : grp['obs_duration'].max(),
        'event'    : int(grp['has_failed'].max()),
        'proj_size': grp['stqf_devs'].mean()
    })
km_df = pd.DataFrame(rows_km)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Curva globale
kmf = KaplanMeierFitter()
kmf.fit(km_df['duration'], event_observed=km_df['event'], label='Tutti i developer')
kmf.plot_survival_function(ax=axes[0], color='steelblue')
axes[0].set_title("Kaplan-Meier: Sopravvivenza globale\n(ACTIVE vs FAILED)")
axes[0].set_xlabel("Finestre trimestrali dalla prima contribuzione")
axes[0].set_ylabel("P(non fallimento)")
axes[0].grid(alpha=0.3)

# costruisce una curva di sopravvivenza per developer, dove l'"evento" è il fallimento e la "durata" è la tenure

# Stratificato per dimensione progetto (piccolo vs grande)
median_size = km_df['proj_size'].median() # mediana della dimensione del team (numero medio di developer per progetto)
for label, mask, color in [
    (f'Team piccolo (≤{median_size:.0f} dev)', km_df['proj_size'] <= median_size, 'tomato'),
    (f'Team grande (>{median_size:.0f} dev)',  km_df['proj_size'] >  median_size, 'seagreen'),
]:
    sub = km_df[mask]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(sub['duration'], event_observed=sub['event'], label=f"{label} (n={len(sub):,})")
    kmf2.plot_survival_function(ax=axes[1], color=color)
axes[1].set_title("Kaplan-Meier stratificato per dimensione team")
axes[1].set_xlabel("Finestre trimestrali")
axes[1].set_ylabel("P(non fallimento)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'kaplan_meier.png', dpi=150)
plt.close()
print("Kaplan-Meier salvato.")

# Analisi risultati Kaplan-Meier:
# La sopravvivenza crolla a ~0,41 dopo la prima finestra. Cioè circa sei developer su dieci spariscono dopo un solo trimestre.
# A 10 finestre siamo a ~0,20,e la curva decade quasi a zero. La stratificazione per dimensione team mostra curve quasi sovrapposte,
# come se la dimensione del team incidesse poco sulla sopravvivenza individuale.

# Distribuzione delle feature per ACTIVE vs FAILED
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plot_pairs = [
    ('community_smell_count',    'Community Smells'),
    ('traditional_smell_count',  'Traditional Code Smells'),
    ('ml_smell_count',           'ML Smells'),
    ('stqf_global_turnover',     'Global Turnover'),
    ('stqf_ratio_smelly_devs',   'Ratio Smelly Devs'),
    ('vulnerability_count',      'Vulnerabilità'),
]
for ax, (col, title) in zip(axes.flat, plot_pairs):
    if col not in df.columns: continue
    for val, color, label in [(0,'steelblue','ACTIVE'), (1,'tomato','FAILED')]:
        sub = df[df['has_failed'] == val][col].dropna()
        sub = sub[sub <= sub.quantile(0.95)]
        ax.hist(sub, bins=30, alpha=0.55, density=True, color=color, label=label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.suptitle("Distribuzione metriche SD/TD per outcome (ACTIVE vs FAILED)", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'eda_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Distribuzioni EDA salvate.")

# ─────────────────────────────────────────────
# STEP 6 — VIF
# ─────────────────────────────────────────────
print("\n")
print("STEP 6 — Test VIF (multicollinearità)")


#VIF per misurare la multicollinearità tra le feature. Rimuove quelle con VIF > 5

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    X_vif = df[FEAT_FINAL].fillna(0)
    X_vif = X_vif.loc[:, (X_vif.std() > 0) & ~X_vif.isin([np.inf, -np.inf]).any()]
    X_c   = add_constant(X_vif)
    vif_data = pd.DataFrame({
        'feature': X_vif.columns,
        'VIF'    : [variance_inflation_factor(X_c.values, i+1)
                    for i in range(X_vif.shape[1])]
    }).sort_values('VIF', ascending=False)

    print(vif_data.to_string(index=False))
    vif_data.to_csv(OUT_DIR / 'vif_results.csv', index=False)

    high_vif   = vif_data[vif_data['VIF'] > VIF_THRESHOLD]['feature'].tolist()
    FEAT_MODEL = [f for f in FEAT_FINAL if f not in high_vif]
    print(f"\nRimosse per VIF > {VIF_THRESHOLD}: {high_vif}")
    print(f"Feature nel modello finale: {len(FEAT_MODEL)}")
except Exception as e:
    print(f"VIF error: {e}")
    FEAT_MODEL = FEAT_FINAL

#stqf_density            inf   ← collinearità PERFETTA
#stqf_core_code_devs     inf
#stqf_degree_centr       inf
#stqf_core_global_devs   inf
#...
#Rimosse per VIF > 5: [density, core_code_devs, degree_centr, core_global_devs,
#                      closeness_centr, devs, log_files_touched_count, log_code_churn]
#Feature nel modello finale: 21

# ─────────────────────────────────────────────
# STEP 7 — EARLY-STAGE: Logistic Regression
# ─────────────────────────────────────────────
print("\n")
print("STEP 7 — Early-Stage Model (Logistic Regression)")

print(f"Definizione early-stage: developer_tenure <= {EARLY_STAGE_WINDOWS} finestre")


# sui soli developer "giovani" (tenure ≤ 3) stima una regressione logistica: probabilità di fallire in funzione delle metriche SD/TD.
# L'idea è: il debito socio-tecnico precoce predice l'abbandono?
# Il McFadden R² misura la bontà di adattamento

df_early = df[df['developer_tenure'] <= EARLY_STAGE_WINDOWS].copy()
print(f"Osservazioni: {len(df_early):,}  |  Event rate: {df_early['has_failed'].mean():.2%}")

try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Logit

    X_e = df_early[FEAT_MODEL].fillna(0)
    X_e = X_e.loc[:, X_e.std() > 0]
    y_e = df_early['has_failed']

    logit_model = Logit(y_e, sm.add_constant(X_e)).fit(disp=False, maxiter=300)

    mcfadden = 1 - logit_model.llf / logit_model.llnull
    print(f"\nMcFadden R²: {mcfadden:.4f}")
    print(logit_model.summary2())

    res_logit = pd.DataFrame({
        'feature'    : logit_model.params.index,
        'coef'       : logit_model.params.values,
        'odds_ratio' : np.exp(np.clip(logit_model.params.values, -15, 15)),
        'p_value'    : logit_model.pvalues.values,
        'sig'        : pd.cut(logit_model.pvalues.values,
                              bins=[-1, 0.001, 0.01, 0.05, 1],
                              labels=['***','**','*','ns'])
    }).sort_values('p_value')
    res_logit.to_csv(OUT_DIR / 'early_stage_logit_results.csv', index=False)

    # Plot odds ratio variabili significative
    sig_logit = res_logit[
        (res_logit['sig'].isin(['***','**','*'])) &
        (res_logit['feature'] != 'const')
    ].sort_values('odds_ratio')

    if not sig_logit.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, len(sig_logit) * 0.6)))
        colors = ['tomato' if o > 1 else 'steelblue' for o in sig_logit['odds_ratio']]
        bars = ax.barh(sig_logit['feature'], sig_logit['odds_ratio'],
                       color=colors, edgecolor='white', alpha=0.85)
        ax.axvline(1, color='black', linewidth=1, linestyle='--')
        for bar, (_, row) in zip(bars, sig_logit.iterrows()):
            ax.text(bar.get_width() + 0.05,
                    bar.get_y() + bar.get_height()/2,
                    f"OR={row['odds_ratio']:.2f} {row['sig']}",
                    va='center', fontsize=8.5)
        ax.set_xlabel("Odds Ratio\nRosso > 1 = aumenta rischio fallimento  |  Blu < 1 = protettivo")
        ax.set_title(f"Logistic Regression — Early Stage (p < 0.05)\n"
                     f"McFadden R² = {mcfadden:.3f}  |  n = {len(df_early):,} osservazioni")
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'logit_odds_ratios.png', dpi=150)
        plt.close()
    print("Logit salvato.")
except Exception as e:
    print(f"Logit error: {e}")
    import traceback; traceback.print_exc()


# Risultato:
# 
# n = 13.951 osservazioni | event rate 82,6% | McFadden R² = 0,108
#
# feature                       Odds Ratio             p           direzione
# stqf_global_truck                2,38              <0,001    aumenta il rischio
# stqf_ratio_smelly_quitters       2,26              <0,001    aumenta il rischio
# stqf_ratio_smelly_devs           2,01              <0,001    aumenta il rischio
# stqf_communicability             1,77              <0,001    aumenta il rischio
# stqf_core_code_turnover          1,24               0,047    aumenta il rischio
# developer_tenure                 0,84              <0,001    protettivo
# log_commits_count                0,59              <0,001    protettivo
#
#
# NB (censura a destra): l'OR contro-intuitivo di st_congruence (era 4,82) SPARISCE, non è più significativo.
# Ora i fattori di rischio sono coerenti col debito sociale: truck factor, smelly quitters/devs, communicability.
# Il debito tecnico (smell tradizionali, ML, vulnerabilità) resta non significativo.
# McFadden 0,108; gran parte del potere predittivo viene comunque da commits e tenure, non dal debito in sé


# ─────────────────────────────────────────────
# STEP 8 — LATER-STAGE: Cox PH
# ─────────────────────────────────────────────
print("\n")
print("STEP 8 — Later-Stage Model (Cox Proportional-Hazards)")


# Sui developer "maturi" (tenure > 3) usa un modello di Cox (survival analysis): stima l'hazard, cioè il rischio istantaneo di fallire, in funzione delle covariate.
# Il penalizer=0.1 (regolarizzazione ridge) stabilizza le stime in presenza di collinearità residua.
# L'exp(coef) è l'Hazard Ratio: >1 aumenta il rischio, <1 lo riduce.
# La concordance (tipo AUC per survival) misura la capacità discriminante.

df_late = df[df['developer_tenure'] > EARLY_STAGE_WINDOWS].copy()
# collassa il panel a 1 riga per developer: durata = tenure max, evento = has_failed max, feature = media
print(f"Osservazioni later-stage: {len(df_late):,}")

# Un record per developer (durata = finestre totali, event = has_failed)
print("Costruzione panel (1 riga per developer)...")
rows_cox = []
for (proj, dev), grp in df_late.groupby(['project_url', 'developer_id']):
    row = {
        'duration': grp['obs_duration'].max(),
        'event'   : int(grp['has_failed'].max()),
    }
    for f in FEAT_MODEL:
        row[f] = grp[f].mean() if f in grp.columns else 0.0
    rows_cox.append(row)

cox_panel = pd.DataFrame(rows_cox)
cox_panel = (cox_panel[cox_panel['duration'] > 0]
               .replace([np.inf, -np.inf], np.nan)
               .fillna(0))

feat_valid = [f for f in FEAT_MODEL if cox_panel[f].std() > 0]
print(f"Panel: {cox_panel.shape[0]:,} developer  |  Event rate: {cox_panel['event'].mean():.2%}")
print(f"Feature valide: {len(feat_valid)}")

try:
    from lifelines import CoxPHFitter

    cph = CoxPHFitter(penalizer=0.1) # penalizzazione L2 per stabilità
    cph.fit(cox_panel[feat_valid + ['duration', 'event']],
            duration_col='duration', event_col='event')
    cph.print_summary()

    cox_results = cph.summary.reset_index()
    cox_results.to_csv(OUT_DIR / 'later_stage_cox_results.csv', index=False)

    # Plot coefficienti significativi
    sig_cox = cox_results[cox_results['p'] < 0.05].sort_values('coef')
    if not sig_cox.empty:
        fig, ax = plt.subplots(figsize=(10, max(5, len(sig_cox) * 0.6)))
        colors = ['tomato' if c > 0 else 'steelblue' for c in sig_cox['coef']]
        bars = ax.barh(sig_cox['covariate'], sig_cox['coef'],
                       color=colors, edgecolor='white', alpha=0.85,
                       xerr=sig_cox['se(coef)'], capsize=3)
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        for _, row in sig_cox.iterrows():
            pstar = ('***' if row['p'] < 0.001 else
                     '**'  if row['p'] < 0.01  else '*')
            offset = row['coef'] + (0.02 if row['coef'] >= 0 else -0.02)
            ax.text(offset + row['se(coef)'],
                    list(sig_cox['covariate']).index(row['covariate']),
                    f"HR={row['exp(coef)']:.2f} {pstar}",
                    va='center', fontsize=8.5,
                    ha='left' if row['coef'] >= 0 else 'right')
        ax.set_xlabel("Log Hazard Ratio (± SE)\nRosso > 0 = aumenta rischio  |  Blu < 0 = protettivo")
        ax.set_title(f"Cox PH — Later Stage (p < 0.05)\n"
                     f"Concordance = {cph.concordance_index_:.3f}  |  "
                     f"n = {len(cox_panel):,} developer")
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'cox_hazard_ratios.png', dpi=150)
        plt.close()
    print(f"Cox salvato. Concordance index: {cph.concordance_index_:.4f}")
except Exception as e:
    print(f"Cox error: {e}")
    import traceback; traceback.print_exc()



# Risultato: Panel: 2.791 developer | event rate 71,69% | Concordance = 0,89
# 
# covariate              HR exp (coef)          p
# stqf_global_truck          2,13            <0,001
# stqf_global_turnover       1,71             0,001   (ora SIGNIFICATIVO)
# stqf_communicability       1,45             0,008
# stqf_ratio_smelly_quitters 1,35             0,024
# log_commits_count          0,65            <0,001   protettivo
# developer_tenure           0,89            <0,001 (z ≈ −9,9)
# 
# Il debito tecnico (smell tradizionali, community smell, vulnerabilità) resta non significativo.
# 
# Interpretazione (censura a destra):
# Con la censura il TURNOVER emerge come fattore di rischio significativo (prima non lo era),
# e developer_tenure non domina più (z da −44,7 a −9,9). Concordance 0,89 (non più gonfiata a 0,93).
# global_truck (bus factor) e communicability aumentano leggermente il rischio (HR ~1,4). Un bus factor più alto associato a più rischio è plausibile
# La maggior parte delle metriche di debito socio-tecnico non ha effetto una volta controllati tenure, età e attività.

# ─────────────────────────────────────────────
# STEP 9 — INTERAZIONI SD × TD
# ─────────────────────────────────────────────
print("\n")
print("STEP 9 — Modello esteso con interazioni SD × TD")

# Si testa l'ipotesi di sinergia tra debito sociale (SD) e tecnico (TD)
# I termini di interazione sono semplici prodotti tra una metrica SD e una TD, poi si rifitta il Cox

interaction_pairs = [
    ('community_smell_count',    'traditional_smell_count', 'SD_comm_x_TD_trad'),
    ('stqf_ratio_smelly_devs',   'vulnerability_count',     'SD_smelly_x_TD_vuln'),
    ('stqf_global_turnover',     'code_churn',              'SD_turnover_x_TD_churn'),
    ('stqf_global_turnover',     'traditional_smell_count', 'SD_turnover_x_TD_trad'),
    ('stqf_ratio_smelly_quitters','bug_introduced_count_rszz','SD_quitters_x_TD_bugs'),
]

for sd, td, name in interaction_pairs:
    if sd in df_late.columns and td in df_late.columns:
        df_late[name] = df_late[sd] * df_late[td]  # termine di interazione dato dal prodotto
        cap = df_late[name].quantile(0.99)
        if cap > 0:
            df_late[name] = df_late[name].clip(upper=cap)
        print(f"  Creato: {name}")

interact_cols = [n for _, _, n in interaction_pairs if n in df_late.columns]

try:
    from lifelines import CoxPHFitter

    rows_int = []
    for (proj, dev), grp in df_late.groupby(['project_url', 'developer_id']):
        row = {
            'duration': grp['obs_duration'].max(),
            'event'   : int(grp['has_failed'].max()),
        }
        for f in feat_valid + interact_cols:
            row[f] = grp[f].mean() if f in grp.columns else 0.0
        rows_int.append(row)

    cox_int = (pd.DataFrame(rows_int)
                 .pipe(lambda x: x[x['duration'] > 0])
                 .replace([np.inf, -np.inf], np.nan)
                 .fillna(0))

    feat_int_ok = [f for f in feat_valid + interact_cols if cox_int[f].std() > 0]
    cph_int = CoxPHFitter(penalizer=0.1)
    cph_int.fit(cox_int[feat_int_ok + ['duration', 'event']],
                duration_col='duration', event_col='event')

    print("\nModello Cox con interazioni SD × TD:")
    cph_int.print_summary()

    int_results = cph_int.summary.reset_index()
    int_results.to_csv(OUT_DIR / 'cox_interactions_results.csv', index=False)

    # Plot solo i termini di interazione
    int_only = int_results[int_results['covariate'].str.startswith('SD_')]
    if not int_only.empty:
        fig, ax = plt.subplots(figsize=(10, max(3, len(int_only) * 0.9)))
        colors = ['tomato' if c > 0 else 'steelblue' for c in int_only['coef']]
        ax.barh(int_only['covariate'], int_only['coef'],
                color=colors, edgecolor='white', alpha=0.85,
                xerr=int_only['se(coef)'], capsize=4)
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        for _, row in int_only.iterrows():
            pstar = ('***' if row['p'] < 0.001 else
                     '**'  if row['p'] < 0.01  else
                     '*'   if row['p'] < 0.05  else 'ns')
            ax.text(row['coef'] + row['se(coef)'] + 0.01,
                    list(int_only['covariate']).index(row['covariate']),
                    f"p={row['p']:.3f} {pstar}", va='center', fontsize=9)
        ax.set_xlabel("Log Hazard Ratio (± SE)")
        ax.set_title(f"Termini di interazione SD × TD\n"
                     f"Concordance modello base={cph.concordance_index_:.3f}  "
                     f"con interazioni={cph_int.concordance_index_:.3f}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'cox_interaction_terms.png', dpi=150)
        plt.close()
    print(f"Interazioni salvate. Concordance: {cph_int.concordance_index_:.4f}")
except Exception as e:
    print(f"Interazioni error: {e}")
    import traceback; traceback.print_exc()


# Risultato: i cinque termini di interazione sono tutti non significativi:

# interazione              HR       p
# SD_comm_x_TD_trad       ~1,0     0,75
# SD_smelly_x_TD_vuln     ~1,0     0,54
# SD_turnover_x_TD_churn  ~1,0     0,68
# SD_turnover_x_TD_trad   ~1,0     0,57
# SD_quitters_x_TD_bugs   ~1,0     0,79
# 
# La concordance resta 0,888 (le interazioni SD x TD non aggiungono potere).
# 
# Su questi dati non emerge alcuna sinergia SD × TD. L'ipotesi teorica non trova supporto

# ─────────────────────────────────────────────
# STEP 10 — ANALISI PROGETTO: ACTIVE vs FAILED
# ─────────────────────────────────────────────
print("\n")
print("STEP 10 — Analisi a livello progetto: ACTIVE vs FAILED")

# SI sale dal livello developer al livello progetto, calcolando metriche medie per progetto e confrontando i 166 ACTIVE contro i 166 FAILED (definizione v2) con il test di Mann-Whitney U

proj_rows = []
for proj, grp in df.groupby('project_url'):
    # Un progetto è FAILED se tutti i suoi developer sono failed
    proj_failed = int(grp['proj_failed'].max())
    proj_rows.append({
        'project_url'         : proj,
        'proj_failed'         : proj_failed,
        'n_developers'        : grp['developer_id'].nunique(),
        'n_windows'           : grp['time_window_id'].nunique(),
        'avg_community_smells': grp['community_smell_count'].mean(),
        'avg_trad_smells'     : grp['traditional_smell_count'].mean(),
        'avg_ml_smells'       : grp['ml_smell_count'].mean(),
        'avg_vuln'            : grp['vulnerability_count'].mean(),
        'avg_turnover'        : grp['stqf_global_turnover'].mean(),
        'avg_truck'           : grp['stqf_global_truck'].mean(),
        'avg_churn'           : grp['code_churn'].mean(),
        'avg_commits'         : grp['commits_count'].mean(),
        'avg_smelly_devs'     : grp['stqf_ratio_smelly_devs'].mean(),
        'avg_st_congruence'   : grp['stqf_st_congruence'].mean(),
        'avg_density'         : grp['stqf_density'].mean(),
        'avg_smelly_quitters' : grp['stqf_ratio_smelly_quitters'].mean(),
    })
proj_level = pd.DataFrame(proj_rows)

print(f"Progetti ACTIVE: {(proj_level['proj_failed']==0).sum()}")
print(f"Progetti FAILED: {(proj_level['proj_failed']==1).sum()}")
proj_level.to_csv(OUT_DIR / 'project_level_outcomes.csv', index=False)

# Confronto statistico ACTIVE vs FAILED
active_p = proj_level[proj_level['proj_failed'] == 0]
failed_p = proj_level[proj_level['proj_failed'] == 1]
compare_cols = [
    'avg_community_smells', 'avg_trad_smells', 'avg_ml_smells',
    'avg_vuln', 'avg_turnover', 'avg_churn', 'n_developers',
    'avg_smelly_devs', 'avg_st_congruence', 'avg_density',
    'avg_smelly_quitters'
]

print("\nConfronto medie ACTIVE vs FAILED (livello progetto):")
compare_df = pd.DataFrame({
    'ACTIVE': active_p[compare_cols].mean(),
    'FAILED': failed_p[compare_cols].mean(),
    'diff%' : ((failed_p[compare_cols].mean() - active_p[compare_cols].mean())
               / active_p[compare_cols].mean().replace(0, np.nan) * 100)
}).round(3)
print(compare_df.to_string())
compare_df.to_csv(OUT_DIR / 'active_vs_failed_comparison.csv')

print("\nMann-Whitney U (ACTIVE vs FAILED a livello progetto):")
mw_rows = []
for col in compare_cols:
    stat, p = scipy_stats.mannwhitneyu(
        active_p[col].dropna(), failed_p[col].dropna(), alternative='two-sided')
    pstar = ('***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns')
    print(f"  {col:35s}: p={p:.4f} {pstar}")
    mw_rows.append({'feature': col, 'U': stat, 'p_value': p, 'sig': pstar,
                    'mean_active': active_p[col].mean(),
                    'mean_failed': failed_p[col].mean()})
mw_df = pd.DataFrame(mw_rows)
mw_df.to_csv(OUT_DIR / 'active_vs_failed_mw.csv', index=False)

# Boxplot confronto
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plot_cols = ['avg_community_smells', 'avg_trad_smells',
             'avg_turnover', 'avg_smelly_devs',
             'avg_st_congruence', 'avg_density']
for ax, col in zip(axes.flat, plot_cols):
    data   = [active_p[col].dropna(), failed_p[col].dropna()]
    labels = ['ACTIVE', 'FAILED']
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                medianprops={'color': 'black', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], ['steelblue', 'tomato']):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    # Aggiungi p-value
    row_mw = mw_df[mw_df['feature'] == col]
    if not row_mw.empty:
        ax.set_title(f"{col.replace('avg_','').replace('_',' ').title()}\n"
                     f"p={row_mw['p_value'].values[0]:.4f} {row_mw['sig'].values[0]}")
    ax.grid(alpha=0.3)
plt.suptitle("Confronto ACTIVE vs FAILED — metriche SD/TD (livello progetto)", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'active_vs_failed_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Boxplot salvati.")


# Risultato:
# 
# metrica              ACTIVE          FAILED         diff%      p(MWU)
# avg_st_congruence     0,426           0,315         −26%      <0,0001***
# avg_smelly_devs       0,520           0,618         +19%       0,0005***
# avg_density           0,601           0,519         −14%       0,0004***
# avg_community_smells  0,617           0,717         +16%       0,0076**
# avg_trad_smells       2,269           1,810         −20%       0,0244*
# n_developers          42,4            26,1          −38%       0,607 ns
# avg_vuln              0,831           0,641         −23%       0,234 ns
# avg_ml_smells         0,192           0,132         −31%       0,344 ns
# avg_turnover          0,256           0,264         +3%        0,541 ns
# avg_smelly_quitters   0,184           0,189         +2%        0,495 ns
# 
#
# Interpretazione (definizione v2):
# A differenza della v1, la dimensione del team NON discrimina più (mediane uguali 13 vs 13, MWU ns):
# la v2 bilancia i gruppi sull'intensità di commit, non sulla taglia. I discriminanti significativi sono
# ora strutturali/di attività: i progetti ATTIVI (mantenuti intensamente) hanno più st_congruence, più
# density e più smell tradizionali (più codice = più smell); gli INATTIVI hanno relativamente più
# community smells e smelly_devs. Turnover, smelly_quitters e vulnerabilità non distinguono i gruppi.
# Rispetto alla v1 la lettura cambia: sotto v2 gli inattivi appaiono semplicemente meno attivi nel complesso.
# più che dalla salute socio-tecnica.

# ─────────────────────────────────────────────
# STEP 11 — HEATMAP CORRELAZIONE
# ─────────────────────────────────────────────
print("\n")
print("STEP 11 — Heatmap correlazione")

# matrice di correlazione di Pearson tra le principali metriche e has_failed, con maschera triangolare per leggibilità


corr_cols = [
    'community_smell_count', 'traditional_smell_count', 'ml_smell_count',
    'vulnerability_count', 'stqf_global_turnover', 'stqf_ratio_smelly_devs',
    'stqf_density', 'stqf_st_congruence', 'stqf_global_truck',
    'code_churn', 'commits_count', 'se_score', 'has_failed'
]
corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].corr() # Pearson

fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, ax=ax,
            annot_kws={'size': 9},
            cbar_kws={'label': 'Pearson r'})
ax.set_title("Matrice di correlazione — SD, TD e has_failed\n"
             "(332 progetti, 10K developer, ACTIVE vs FAILED)")
plt.tight_layout()
plt.savefig(OUT_DIR / 'correlation_heatmap.png', dpi=150)
plt.close()
print("Heatmap salvata.")


# Risultato:
# stqf_ratio_smelly_devs <-> community_smell_count: r = 0,63 (le due misure di "smell sociale" catturano lo stesso costrutto).
# traditional_smell_count <-> vulnerability_count: r = 0,57; <-> commits_count: r = 0,51 (più codice -> più smell e più vulnerabilità: relazione di volume).
# stqf_st_congruence <-> stqf_density: r = −0,66 (trade-off tra congruenza e densità di rete).
# Riga has_failed: correlazioni prossime a zero con tutto (|r| ≤ 0,10; la più forte è commits_count con −0,10).
# 
# Interpretazione: a livello di singola osservazione, nessuna metrica SD/TD è linearmente correlata con il fallimento. 
# Il segnale predittivo dei modelli viene quasi tutto dalle variabili temporali/di attività, non dal debito socio-tecnico.



# ─────────────────────────────────────────────
# STEP 12 — TRUCK FACTOR & OWNERSHIP 
# ─────────────────────────────────────────────
print("\n")
print("STEP 12 — Truck factor & ownership ")

# Non esiste ownership per-file: files_touched_count, lines_added/deleted, code_churn sono aggregati per developer.
# E' stato scelto il proxy standard. Per ogni progetto ordino i developer per commit totali e definisco core / truck-factor il gruppo minimo che copre ≥50% dei commit
# del progetto. Poi confronto core vs periferici e misuro la concentrazione dei contributi (indice di Gini) per progetto.


dev_proj = (df.groupby(['project_url', 'developer_id'])
              .agg(commits=('commits_count', 'sum'), churn=('code_churn', 'sum'),
                   files=('files_touched_count', 'sum'), failed=('has_failed', 'max'), pfail=('proj_failed', 'max')).reset_index())
_flags, _rows = [], []
for proj, g in dev_proj.groupby('project_url'):
    g = g.sort_values('commits', ascending=False); tot = g['commits'].sum()
    if tot == 0:
        g['is_core'] = 0
    else:
        cutoff = (g['commits'].cumsum() / tot >= 0.50).values.argmax()  # bus factor: min set >=50% commit
        g['is_core'] = 0; g.iloc[:cutoff + 1, g.columns.get_loc('is_core')] = 1
    _flags.append(g[['project_url', 'developer_id', 'is_core']])
    _rows.append({'project_url': proj, 'n_dev': len(g), 'truck_factor': int(g['is_core'].sum()),
                  'gini_commits': gini(g['commits']), 'proj_failed': int(g['pfail'].max())})
dev_proj = dev_proj.merge(pd.concat(_flags), on=['project_url', 'developer_id'])
proj_conc = pd.DataFrame(_rows)
c_core = dev_proj[dev_proj.is_core == 1]; c_peri = dev_proj[dev_proj.is_core == 0]
print(f"Core (truck-factor): {len(c_core):,} | Periferici: {len(c_peri):,}")
for m, lbl in [('commits', 'Commit'), ('churn', 'Churn'), ('files', 'File')]:
    _, p = _ss.mannwhitneyu(c_core[m], c_peri[m]); d = cliffs_delta(c_core[m].values, c_peri[m].values)
    print(f"  {lbl:8s} core_med={c_core[m].median():9.1f} peri_med={c_peri[m].median():7.1f} p={p:.1e} d={d:+.2f} ({eff_label(d)})")
print(f"  Fallimento: core={c_core['failed'].mean():.1%} periferici={c_peri['failed'].mean():.1%}")
g_act = proj_conc[proj_conc.proj_failed == 0]['gini_commits'].dropna()
g_fai = proj_conc[proj_conc.proj_failed == 1]['gini_commits'].dropna()
_, p_g = _ss.mannwhitneyu(g_act, g_fai)
print(f"  Gini contributi: ACTIVE={g_act.median():.3f} FAILED={g_fai.median():.3f} p={p_g:.1e}")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].boxplot([np.log1p(c_core['commits']), np.log1p(c_peri['commits'])], tick_labels=['Core', 'Periferici']); ax[0].set_title('Commit (log)')
ax[1].boxplot([g_act, g_fai], tick_labels=['ACTIVE', 'FAILED']); ax[1].set_title(f'Gini per progetto (p={p_g:.1e})')
ax[2].bar(['Core', 'Periferici'], [c_core['failed'].mean(), c_peri['failed'].mean()], color=['#c0392b', '#2980b9']); ax[2].set_ylim(0, 1); ax[2].set_title('Tasso fallimento')
plt.tight_layout(); plt.savefig(OUT_DIR / 'truck_factor_ownership.png', dpi=110); plt.close()
proj_conc.to_csv(OUT_DIR / 'truck_factor_concentration.csv', index=False)
print("Truck factor salvato.")


# Risultati:
# Confronto core vs periferici       Core (mediana)     Periferici (mediana)     Cliff's d
# Commit                                183,5                 2,0              +0,96 (grande)
# Code churn                           58.222                 58               +0,92 (grande)
# File toccati                          379                    3               +0,93 (grande)

# In sintesi: l'ownership è estremamente concentrata; i core contribuiscono ~90× i periferici e sopravvivono un po' di più.


# ─────────────────────────────────────────────
# STEP 13 — SOGLIE SD/TD
# ─────────────────────────────────────────────
print("\n")
print("STEP 13 — Soglie SD/TD")

# 1. ROC + indice di Youden per il cutpoint ottimale e l'AUC univariata di ogni metrica;
# 2. tasso di fallimento per decile;
# 3. albero decisionale (max_depth=3, solo SD/TD, senza tenure/età). 



SDTD = {'community_smell_count': 'SD community', 'stqf_ratio_smelly_devs': 'SD smelly devs',
        'stqf_global_turnover': 'SD turnover', 'stqf_st_congruence': 'SD st-congruence',
        'traditional_smell_count': 'TD trad', 'ml_smell_count': 'TD ML', 'vulnerability_count': 'TD vuln'}
for c in SDTD: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
_rr = []
print(f"  {'metrica':16s} {'soglia':>9s} {'AUC':>6s} {'sens':>5s} {'spec':>5s}")
for c, lbl in SDTD.items():
    thr, a, se, sp = youden(df['has_failed'], df[c])
    print(f"  {lbl:16s} {thr:9.3f} {a:6.3f} {se:5.2f} {sp:5.2f}"); _rr.append({'metric': lbl, 'threshold': thr, 'auc': a})
pd.DataFrame(_rr).to_csv(OUT_DIR / 'thresholds_roc.csv', index=False)
_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, class_weight='balanced', random_state=0)
_tree.fit(df[list(SDTD)], df['has_failed'])
print(f"  Albero (solo SD/TD) AUC={roc_auc_score(df['has_failed'], _tree.predict_proba(df[list(SDTD)])[:,1]):.3f}")
print(export_text(_tree, feature_names=[SDTD[c] for c in SDTD]))
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
for ax, (c, lbl) in zip(axs.ravel(), list(SDTD.items())[:4]):
    q = pd.qcut(df[c].rank(method='first'), 10, labels=False)
    ax.plot(range(1, 11), df.groupby(q)['has_failed'].mean().values, 'o-'); ax.set_title(lbl); ax.grid(alpha=.3)
plt.suptitle('Tasso di fallimento per decile'); plt.tight_layout()
plt.savefig(OUT_DIR / 'thresholds_failure_rate.png', dpi=110); plt.close()
print("Thresholds salvato.")


# metrica               soglia (Youden)   AUC
# SD community smells     ≤ 3,00         0,492
# SD ratio smelly devs    ≥ 0,03         0,519
# SD turnover             ≤ 0,50         0,497
# SD st-congruence        ≥ 0,50         0,526
# TD trad smells          ≤ 14,0         0,489
# TD ML smells            ≤ 8,0          0,479
# TD vulnerabilità        ≤ 15,0         0,457




# ─────────────────────────────────────────────
# STEP 14 — SENTIMENT & EMOZIONI
# ─────────────────────────────────────────────
print("\n")
print("STEP 14 — Sentiment & Emozioni")

# sentiment_label è degenere: 24.912 "Neutral", 4 "Positive", 0 "Negative".
# Nel JSON però c'è sentiment_emotions con 6 emozioni (Anger, Fear, Sadness, Love, Joy, Surprise) è stato estratto e usato insieme a sentiment_score e se_score. 
# Confronto con Mann-Whitney + Cliff's delta.

print(
    "  sentiment_label degenere (>99.9% Neutral) -> "
    "escluso. Uso emozioni + score."
)

# Le emozioni sono contenute all'interno di developer_raw_json
RAW_JSON_COLUMN = 'developer_raw_json'

if RAW_JSON_COLUMN not in df.columns:
    raise KeyError(
        f"Colonna '{RAW_JSON_COLUMN}' non trovata nel dataset."
    )

# Converte ogni JSON testuale in un dizionario Python.
# safe_json è già definita nello STEP 2.
developer_json = df[RAW_JSON_COLUMN].apply(safe_json)


def extract_emotion(raw_data, emotion):
    """
    Estrae il valore di un'emozione dal JSON del developer.
    Restituisce NaN quando il dato non è disponibile.
    """
    if not isinstance(raw_data, dict):
        return np.nan

    emotions = raw_data.get('sentiment_emotions', {})

    if not isinstance(emotions, dict):
        return np.nan

    return emotions.get(emotion, np.nan)


# Estrazione delle sei emozioni
for emotion in EMO:
    column_name = f'emo_{emotion}'

    df[column_name] = pd.to_numeric(
        developer_json.map(
            lambda raw_data, current_emotion=emotion:
            extract_emotion(raw_data, current_emotion)
        ),
        errors='coerce'
    )

print("  Colonne emozionali estratte:")

for emotion in EMO:
    column_name = f'emo_{emotion}'

    print(
        f"    {column_name:15s}: "
        f"{df[column_name].notna().sum():,}/{len(df):,}"
    )

# Elenco dei segnali analizzati
SENT = {
    'sentiment_score': 'Sentiment',
    'se_score': 'SE score'
}

for emotion in EMO:
    column_name = f'emo_{emotion}'

    if (
        column_name in df.columns
        and df[column_name].notna().sum() >= 10
    ):
        SENT[column_name] = emotion

# Divisione tra osservazioni ACTIVE e FAILED
act = df[df['has_failed'] == 0].copy()
fai = df[df['has_failed'] == 1].copy()

sentiment_results = []

for column_name, label in SENT.items():

    active_values = pd.to_numeric(
        act[column_name],
        errors='coerce'
    ).dropna()

    failed_values = pd.to_numeric(
        fai[column_name],
        errors='coerce'
    ).dropna()

    if len(active_values) < 5 or len(failed_values) < 5:
        print(
            f"  {label:12s}: dati insufficienti "
            f"(ACTIVE={len(active_values)}, "
            f"FAILED={len(failed_values)})"
        )
        continue

    _, p_value = _ss.mannwhitneyu(
        active_values,
        failed_values,
        alternative='two-sided'
    )

    delta = cliffs_delta(
        active_values.values,
        failed_values.values
    )

    print(
        f"  {label:12s} "
        f"ACT={active_values.median():7.3f} "
        f"FAIL={failed_values.median():7.3f} "
        f"p={p_value:.1e} "
        f"d={delta:+.2f} ({eff_label(delta)})"
    )

    sentiment_results.append({
        'signal': label,
        'active_med': active_values.median(),
        'failed_med': failed_values.median(),
        'p_value': p_value,
        'cliffs_d': delta,
        'effect_size': eff_label(delta),
        'n_active': len(active_values),
        'n_failed': len(failed_values)
    })

pd.DataFrame(sentiment_results).to_csv(
    OUT_DIR / 'sentiment_active_vs_failed.csv',
    index=False
)

# ── Grafici ──

available_emotions = [
    emotion
    for emotion in EMO
    if (
        f'emo_{emotion}' in df.columns
        and df[f'emo_{emotion}'].notna().sum() >= 10
    )
]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Profilo emotivo medio
if available_emotions:

    x = np.arange(len(available_emotions))
    width = 0.38

    active_means = [
        act[f'emo_{emotion}'].mean()
        for emotion in available_emotions
    ]

    failed_means = [
        fai[f'emo_{emotion}'].mean()
        for emotion in available_emotions
    ]

    axes[0].bar(
        x - width / 2,
        active_means,
        width,
        label='ACTIVE',
        color='#2980b9'
    )

    axes[0].bar(
        x + width / 2,
        failed_means,
        width,
        label='FAILED',
        color='#c0392b'
    )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        available_emotions,
        rotation=20
    )

    axes[0].set_ylabel('Valore medio')
    axes[0].set_title('Profilo emotivo medio')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')

else:
    axes[0].text(
        0.5,
        0.5,
        'Dati emozionali non disponibili',
        ha='center',
        va='center'
    )
    axes[0].set_axis_off()

# Boxplot SE score
axes[1].boxplot(
    [
        pd.to_numeric(
            act['se_score'],
            errors='coerce'
        ).dropna(),

        pd.to_numeric(
            fai['se_score'],
            errors='coerce'
        ).dropna()
    ],
    tick_labels=['ACTIVE', 'FAILED']
)

axes[1].set_title('SE score')
axes[1].set_ylabel('Valore')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()

plt.savefig(
    OUT_DIR / 'sentiment_emotions.png',
    dpi=110
)

plt.close()

print("Sentiment analysis salvata.")


# segnale         ACTIVE (mediana)       FAILED (mediana)      Cliff's d
# SE score             2,50                   0,90               +0,20 
# Emo: Joy             0,240                  0,236              +0,19 
# Emo: Fear            0,126                  0,123              +0,15 
# Emo: Love            0,098                  0,096              +0,15 
# Emo: Surprise        0,093                  0,096              −0,14 
# Emo: Sadness         0,157                  0,154              +0,12 
# Sentiment score      0,013                  0,011              +0,10 
# Emo: Anger           0,183                  0,185              −0,09


# ─────────────────────────────────────────────
# STEP 15 — COX TEMPO-VARIANTE
# ─────────────────────────────────────────────
print("\n")
print("STEP 15 — Cox tempo-variante (pre-fallimento)")

CENSOR_GAP = INACTIVITY_THRESHOLD
if 'log_commits_count' not in df.columns:
    df['log_commits_count'] = np.log1p(pd.to_numeric(df['commits_count'], errors='coerce').fillna(0))

# esito canonico a livello (progetto, developer)
_ev = df.groupby(['project_url', 'developer_id']).agg(
    event=('has_failed', 'max'), n_win=('time_window_rank', 'nunique')).reset_index()

ADV = {'community_smell_count': 'SD community', 'stqf_global_turnover': 'SD turnover',
       'stqf_st_congruence': 'SD st-congr', 'traditional_smell_count': 'TD trad',
       'vulnerability_count': 'TD vuln', 'log_commits_count': 'Commit(log)',
       'se_score': 'SE score', 'sentiment_score': 'Sentiment'}
for c in ADV:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)


print(f"Esito canonico (censura a destra): eventi={_ev['event'].mean():.1%} censurati={1-_ev['event'].mean():.1%}")
_tv = _ev[_ev.n_win >= 3][['project_url', 'developer_id', 'event']]
_sub = df.merge(_tv[['project_url', 'developer_id']], on=['project_url', 'developer_id'])
_rows = []
for (proj, dev), grp in _sub.groupby(['project_url', 'developer_id']):
    grp = grp.sort_values('developer_tenure')
    ev = int(_tv[(_tv.project_url == proj) & (_tv.developer_id == dev)]['event'].iloc[0])
    ten = grp['developer_tenure'].values
    for j in range(len(grp)):
        start = ten[j]; stop = ten[j + 1] if j < len(grp) - 1 else ten[j] + 1
        if stop <= start: stop = start + 1
        r = {'id': hash((proj, dev)) & 0xffffffff, 'start': start, 'stop': stop,
             'event': ev if j == len(grp) - 1 else 0}
        for c in ADV: r[c] = grp[c].values[j]
        _rows.append(r)
_long = pd.DataFrame(_rows)
for c in ADV:
    s = _long[c].std()
    if s > 0: _long[c] = (_long[c] - _long[c].mean()) / s
_ctv = CoxTimeVaryingFitter(penalizer=0.1)
_ctv.fit(_long, id_col='id', event_col='event', start_col='start', stop_col='stop')
_res = _ctv.summary[['exp(coef)', 'p']].copy()
_res.index = [ADV.get(i, i) for i in _res.index]; _res = _res.reindex(_res['p'].sort_values().index)
print(f"  Dev={_long['id'].nunique():,} intervalli={len(_long):,} | HR (standardizzate, >1 = piu' rischio):")
for name, row in _res.iterrows():
    sig = '***' if row['p'] < .001 else '**' if row['p'] < .01 else '*' if row['p'] < .05 else ''
    print(f"   {name:16s} HR={row['exp(coef)']:.3f} p={row['p']:.4f} {sig}")
_res.to_csv(OUT_DIR / 'timevarying_cox.csv')
print("Analisi Cox tempo-variante conclusa")



# ─────────────────────────────────────────────
# STEP 16 — FIRMA PRE-FALLIMENTO
# ─────────────────────────────────────────────
print("\n")
print("STEP 16 — Firma pre-fallimento (traiettoria finestre finali)")


_aid = _ev[(_ev.n_win >= 3) & (_ev.event == 1)][['project_url', 'developer_id']]
_traj = df.merge(_aid, on=['project_url', 'developer_id']).sort_values(
    ['project_url', 'developer_id', 'time_window_rank']).copy()
_seq = _traj.groupby(['project_url', 'developer_id']).cumcount()
_tot = _traj.groupby(['project_url', 'developer_id'])['time_window_rank'].transform('size')
_traj['pos'] = _seq - (_tot - 1)                       # 0 = ultima finestra, negativi = precedenti
SIG = ['community_smell_count', 'traditional_smell_count', 'stqf_global_turnover', 'log_commits_count', 'se_score']
_prof = _traj[_traj['pos'] >= -4].groupby('pos')[SIG].mean().round(3)
print(f"  Dev in abbandono con traiettoria: {_aid.shape[0]:,}")
print("  Media per posizione (0 = ultima finestra prima di sparire):")
print(_prof.to_string())
_first = _traj[_seq == 0].set_index(['project_url', 'developer_id'])[SIG]
_last  = _traj[_traj['pos'] == 0].set_index(['project_url', 'developer_id'])[SIG]
_pair = _first.add_suffix('_f').join(_last.add_suffix('_l')).dropna()
print(f"  Prima vs ultima finestra (Wilcoxon appaiato, n={len(_pair):,}):")
_sr = []
for c in SIG:
    a, b = _pair[c + '_f'], _pair[c + '_l']
    try: _, p = _ss.wilcoxon(a, b)
    except Exception: p = np.nan
    dpc = (b.mean() - a.mean()) / (abs(a.mean()) + 1e-9) * 100
    arrow = 'su' if b.median() > a.median() else 'giu' if b.median() < a.median() else '='
    print(f"   {c:26s} {arrow} prima={a.mean():6.3f} ultima={b.mean():6.3f} ({dpc:+.0f}%) p={p:.1e}")
    _sr.append({'metric': c, 'first_mean': a.mean(), 'last_mean': b.mean(), 'delta_pct': dpc, 'p': p})
pd.DataFrame(_sr).to_csv(OUT_DIR / 'prefailure_signature.csv', index=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
for c in ['traditional_smell_count', 'stqf_global_turnover', 'log_commits_count', 'se_score']:
    v = _prof[c] / (_prof[c].abs().max() + 1e-9)
    ax[0].plot(_prof.index, v, 'o-', label=c)
ax[0].set_xlabel('finestre prima della sparizione (0 = ultima)'); ax[0].set_ylabel('valore normalizzato')
ax[0].legend(fontsize=8); ax[0].set_title('Firma pre-fallimento (traiettoria)'); ax[0].grid(alpha=.3)
hr = _res['exp(coef)']; ax[1].barh(range(len(hr)), hr.values, color=['#c0392b' if v > 1 else '#2980b9' for v in hr.values])
ax[1].set_yticks(range(len(hr))); ax[1].set_yticklabels(hr.index); ax[1].axvline(1, color='k', ls='--')
ax[1].set_title('Cox tempo-variante — Hazard Ratio')
plt.tight_layout(); plt.savefig(OUT_DIR / 'prefailure_timevarying.png', dpi=110); plt.close()
print("Analisi pre-fallimento conclusa")


# ─────────────────────────────────────────────
# STEP 17 — LIVELLO PROGETTO, CONTROLLO PER DIMENSIONE
# ─────────────────────────────────────────────
print("\n")
print("STEP 17 — Livello progetto, controllo per dimensione del team")


_p = df.groupby('project_url').agg(
    n_dev=('developer_id', 'nunique'), SD_community=('community_smell_count', 'mean'),
    SD_turnover=('stqf_global_turnover', 'mean'), SD_stcongr=('stqf_st_congruence', 'mean'),
    SD_smellyq=('stqf_ratio_smelly_quitters', 'mean'), TD_trad=('traditional_smell_count', 'mean'),
    TD_ml=('ml_smell_count', 'mean'), TD_vuln=('vulnerability_count', 'mean'),
    proj_failed=('proj_failed', 'max')).reset_index()
_p['log_n_dev'] = np.log1p(_p['n_dev'])
import statsmodels.api as _sm
from statsmodels.discrete.discrete_model import Logit as _Logit
SDTD = ['SD_community', 'SD_turnover', 'SD_stcongr', 'SD_smellyq', 'TD_trad', 'TD_ml', 'TD_vuln']
print(f"  ACTIVE={int((_p.proj_failed==0).sum())} FAILED={int((_p.proj_failed==1).sum())} | "
      f"n_dev mediana ACTIVE={_p[_p.proj_failed==0]['n_dev'].median():.0f} FAILED={_p[_p.proj_failed==1]['n_dev'].median():.0f}")
print("  Coefficiente logit: GREZZO vs AGGIUSTATO per log(n_dev):")
_A = []
for m in SDTD:
    try:
        mr = _Logit(_p['proj_failed'], _sm.add_constant(_p[[m]])).fit(disp=0)
        ma = _Logit(_p['proj_failed'], _sm.add_constant(_p[[m, 'log_n_dev']])).fit(disp=0)
        flip = '  <-- CAMBIA SEGNO' if np.sign(mr.params[m]) != np.sign(ma.params[m]) else ''
        print(f"   {m:13s} grezzo={mr.params[m]:+6.2f}(p={mr.pvalues[m]:.3f}) | +taglia={ma.params[m]:+6.2f}(p={ma.pvalues[m]:.3f}){flip}")
        _A.append({'metric': m, 'coef_raw': mr.params[m], 'p_raw': mr.pvalues[m], 'coef_adj': ma.params[m], 'p_adj': ma.pvalues[m]})
    except Exception as e:
        print(f"   {m}: {e}")
print("   [log(n_dev) e' sempre il predittore forte e protettivo: la taglia e' il confondente]")
pd.DataFrame(_A).to_csv(OUT_DIR / 'project_size_controlled_logit.csv', index=False)
_p['size_bin'] = pd.cut(_p['n_dev'], [0, 10, 25, 60, 150, 1e9], labels=['1-10', '11-25', '26-60', '61-150', '150+'])
print("  Confronto ACTIVE-FAILED ENTRO fascia (diff mediana; * p<0.05):")
_B = []
for m in SDTD:
    ln = f"   {m:13s} "
    for b in _p['size_bin'].cat.categories:
        sb = _p[_p.size_bin == b]; a = sb[sb.proj_failed == 0][m]; f = sb[sb.proj_failed == 1][m]
        if len(a) >= 3 and len(f) >= 3:
            try: _, pp = _ss.mannwhitneyu(a, f)
            except Exception: pp = 1.0
            ln += f"{b}:{(a.median()-f.median()):+.2f}{'*' if pp<.05 else ''} "; _B.append({'metric': m, 'bin': b, 'diff': a.median()-f.median(), 'p': pp})
    print(ln)
pd.DataFrame(_B).to_csv(OUT_DIR / 'project_size_matched.csv', index=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8)); _dA = pd.DataFrame(_A).set_index('metric'); xx = np.arange(len(_dA))
ax[0].bar(xx - .19, -np.log10(_dA['p_raw'] + 1e-300), .38, label='grezzo', color='#7f8c8d')
ax[0].bar(xx + .19, -np.log10(_dA['p_adj'] + 1e-300), .38, label='+ n_dev', color='#c0392b')
ax[0].axhline(-np.log10(.05), ls='--', color='k'); ax[0].set_xticks(xx); ax[0].set_xticklabels(_dA.index, rotation=30, ha='right')
ax[0].set_ylabel('-log10(p)'); ax[0].legend(); ax[0].set_title("Significativita' prima/dopo controllo taglia")
ax[1].boxplot([np.log1p(_p[_p.proj_failed == 0]['n_dev']), np.log1p(_p[_p.proj_failed == 1]['n_dev'])], tick_labels=['ACTIVE', 'FAILED'])
ax[1].set_ylabel('log(1+n_dev)'); ax[1].set_title('Dimensione team (il confondente)')
plt.tight_layout(); plt.savefig(OUT_DIR / 'project_size_control.png', dpi=110); plt.close()
print("Analisi a livello progetto conclusa")


# ─────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# FALLIMENTO A LIVELLO PROGETTO (multivariato)
#   RQ2 = "quali fattori socio-tecnici distinguono i progetti che sopravvivono
#          da quelli che cessano (assenza di attivita' nell'ultima finestra osservabile)".
#   proj_failed e' di fatto la cessazione COLLETTIVA dell'attivita' (tutti i developer
#   che smettono). Quindi i predittori che DESCRIVONO DIRETTAMENTE l'abbandono, o che
#   usano informazione post-abbandono, sono ESCLUSI perche' renderebbero il modello
#   circolare: in particolare 'global.turnover' (turnover = developer che se ne vanno)
#   e 'ratio.smelly.quitters' (developer che hanno abbandonato). Il modello di progetto
#   testa quindi solo fattori NON derivati dall'abbandono: debito tecnico (smell,
#   vulnerabilita'), community smells, struttura della rete (st-congruence, truck) e
#   dimensione. Non si potra' concludere "l'abbandono causa il fallimento" (tautologico):
#   RQ1 e RQ2 restano due domande distinte, non un nesso causale diretto.
#   18) Logit multivariato  proj_failed ~ predittori PULITI (leakage-free) + log(n_dev)
#   19) Cox di sopravvivenza del progetto (medie fisse; durata=vita, evento=morte)
#   20) Cox tempo-variante del progetto (metriche e dimensione per finestra; specchio STEP 15)
#   Con la nuova definizione (intensita' di commit, binaria), il modello di
#   RIFERIMENTO per RQ2 e' il LOGIT (18); i Cox (19-20) restano come lente secondaria
#   (trattano gli "attivi v2" come censurati, meno appropriato per una def. non temporale).
# ═══════════════════════════════════════════════════════════════════════════
import statsmodels.api as _sm2
from statsmodels.discrete.discrete_model import Logit as _Logit2
from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif
from lifelines import CoxPHFitter as _CoxPH

# predittori PULITI (no turnover/quitters): solo fattori non derivati dall'abbandono, VIF < soglia
PREDS = ['SD_community', 'SD_stcongr', 'SD_truck', 'TD_trad', 'TD_ml', 'TD_vuln', 'log_n_dev']

_pl = df.groupby('project_url').agg(
    n_dev=('developer_id', 'nunique'),
    SD_community=('community_smell_count', 'mean'), SD_stcongr=('stqf_st_congruence', 'mean'),
    SD_truck=('stqf_global_truck', 'mean'), TD_trad=('traditional_smell_count', 'mean'),
    TD_ml=('ml_smell_count', 'mean'), TD_vuln=('vulnerability_count', 'mean'),
    proj_first=('time_window_rank', 'min'), proj_last=('time_window_rank', 'max'),
    proj_failed=('proj_failed', 'max')).reset_index()
_pl['log_n_dev'] = np.log1p(_pl['n_dev'])
_pl['duration'] = _pl['proj_last'] - _pl['proj_first'] + 1
_Z = _pl[PREDS].copy()                                     # standardizzo -> OR/HR per +1 dev. std.
for c in PREDS:
    s = _Z[c].std(); _Z[c] = (_Z[c] - _Z[c].mean()) / s if s > 0 else 0.0
_vif_df = pd.DataFrame({'f': PREDS, 'VIF': [round(_vif(_Z.values, i), 2) for i in range(len(PREDS))]})


# ─────────────────────────────────────────────
# STEP 18 — LOGIT MULTIVARIATO A LIVELLO PROGETTO
# ─────────────────────────────────────────────

print("\n")
print("STEP 18 — RQ2: Logit multivariato a livello progetto (leakage-free)")

print(f"Progetti: {len(_pl)} | ACTIVE (sopravvivono) {(_pl.proj_failed==0).sum()} | FAILED (cessano) {(_pl.proj_failed==1).sum()}")
print("  ESCLUSI per tautologia/leakage: SD_turnover (turnover), SD_smellyq (quitters)")
print(f"  VIF (tutti < {VIF_THRESHOLD}): " + ", ".join(f"{r.f}={r.VIF}" for r in _vif_df.itertuples()))
_ml = _Logit2(_pl['proj_failed'], _sm2.add_constant(_Z[PREDS])).fit(disp=0, maxiter=200)
print(f"  McFadden R2 = {1 - _ml.llf/_ml.llnull:.3f}  (OR per +1 dev. std.; OR>1 -> piu' rischio di morte)")
_rowsL = []
for p in PREDS:
    orr = np.exp(_ml.params[p]); pv = _ml.pvalues[p]
    sig = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else 'ns'
    print(f"   {p:13s} OR={orr:5.2f}  p={pv:.3f}  {sig}")
    _rowsL.append({'predictor': p, 'coef': _ml.params[p], 'odds_ratio': orr, 'p_value': pv})
pd.DataFrame(_rowsL).to_csv(OUT_DIR / 'rq2_project_logit_multivariate.csv', index=False)


# ─────────────────────────────────────────────
# STEP 19 — COX DI SOPRAVVIVENZA DEL PROGETTO
# ─────────────────────────────────────────────

print("\n")
print("STEP 19 — RQ2: Cox di sopravvivenza a livello progetto (leakage-free)")

_cox_pl = pd.concat([_pl[['duration', 'proj_failed']].rename(columns={'proj_failed': 'event'}), _Z[PREDS]], axis=1)
_cph_pl = _CoxPH(penalizer=0.1); _cph_pl.fit(_cox_pl, duration_col='duration', event_col='event')
print(f"  Progetti={len(_cox_pl)} | morti={int(_cox_pl['event'].sum())} | vivi al cutoff={int((_cox_pl['event']==0).sum())}")
print(f"  Concordance = {_cph_pl.concordance_index_:.3f}  (HR per +1 dev. std.; HR>1 -> muore prima)")
_sm_pl = _cph_pl.summary[['exp(coef)', 'p']].reindex(_cph_pl.summary['p'].sort_values().index)
for name, row in _sm_pl.iterrows():
    sig = '***' if row['p'] < .001 else '**' if row['p'] < .01 else '*' if row['p'] < .05 else 'ns'
    print(f"   {name:13s} HR={row['exp(coef)']:5.2f}  p={row['p']:.3f}  {sig}")
_sm_pl.to_csv(OUT_DIR / 'rq2_project_cox.csv')


# ─────────────────────────────────────────────
# STEP 20 — COX TEMPO-VARIANTE A LIVELLO PROGETTO
# ─────────────────────────────────────────────

print("\n")
print("STEP 20 — RQ2: Cox tempo-variante a livello progetto (leakage-free)")

_pw = df.groupby('project_url_x' if 'project_url_x' in df.columns else 'project_url')
_pw = df.groupby(['project_url', 'time_window_rank']).agg(
    n_dev=('developer_id', 'nunique'), SD_community=('community_smell_count', 'mean'),
    SD_stcongr=('stqf_st_congruence', 'mean'), SD_truck=('stqf_global_truck', 'mean'),
    TD_trad=('traditional_smell_count', 'mean'), TD_ml=('ml_smell_count', 'mean'),
    TD_vuln=('vulnerability_count', 'mean')).reset_index()
_pw['log_n_dev'] = np.log1p(_pw['n_dev'])
_pw['age'] = _pw['time_window_rank'] - _pw.groupby('project_url')['time_window_rank'].transform('min')
_pwin = df.groupby('project_url')['time_window_rank'].nunique()
_pfail = df.groupby('project_url')['proj_failed'].max()
_rows_tv = []
for proj, g in _pw[_pw['project_url'].isin(_pwin[_pwin >= 3].index)].groupby('project_url'):
    g = g.sort_values('age'); ev = int(_pfail[proj]); ages = g['age'].values
    for j in range(len(g)):
        start = ages[j]; stop = ages[j + 1] if j < len(g) - 1 else ages[j] + 1
        if stop <= start: stop = start + 1
        r = {'id': hash(proj) & 0xffffffff, 'start': start, 'stop': stop, 'event': ev if j == len(g) - 1 else 0}
        for c in PREDS: r[c] = g[c].values[j]
        _rows_tv.append(r)
_ltv = pd.DataFrame(_rows_tv)
for c in PREDS:
    s = _ltv[c].std()
    if s > 0: _ltv[c] = (_ltv[c] - _ltv[c].mean()) / s
_ctv_pl = CoxTimeVaryingFitter(penalizer=0.1)
_ctv_pl.fit(_ltv, id_col='id', event_col='event', start_col='start', stop_col='stop')
print(f"  Progetti={_ltv['id'].nunique():,} | intervalli={len(_ltv):,} (metriche e dimensione VARIANO per finestra)")
_res_tv = _ctv_pl.summary[['exp(coef)', 'p']].reindex(_ctv_pl.summary['p'].sort_values().index)
for name, row in _res_tv.iterrows():
    sig = '***' if row['p'] < .001 else '**' if row['p'] < .01 else '*' if row['p'] < .05 else 'ns'
    print(f"   {name:13s} HR={row['exp(coef)']:5.2f}  p={row['p']:.3f}  {sig}")
_res_tv.to_csv(OUT_DIR / 'rq2_project_cox_timevarying.csv')


# ─────────────────────────────────────────────
# grafici RQ2
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
_dl = pd.DataFrame(_rowsL).set_index('predictor')['odds_ratio']
ax[0].barh(range(len(_dl)), _dl.values, color=['#c0392b' if v > 1 else '#2980b9' for v in _dl.values])
ax[0].set_yticks(range(len(_dl))); ax[0].set_yticklabels(_dl.index); ax[0].axvline(1, color='k', ls='--')
ax[0].set_title('STEP 18 — Logit progetto leakage-free (OR, +1 SD)')
_hr = _sm_pl['exp(coef)']
ax[1].barh(range(len(_hr)), _hr.values, color=['#c0392b' if v > 1 else '#2980b9' for v in _hr.values])
ax[1].set_yticks(range(len(_hr))); ax[1].set_yticklabels(_hr.index); ax[1].axvline(1, color='k', ls='--')
ax[1].set_title('STEP 19 — Cox progetto leakage-free (HR, +1 SD)')
plt.tight_layout(); plt.savefig(OUT_DIR / 'rq2_project_models.png', dpi=110); plt.close()
try:
    _fix = pd.read_csv(OUT_DIR / 'rq2_project_cox.csv', index_col=0)['exp(coef)']
    _idx = [i for i in PREDS if i in _res_tv.index and i in _fix.index]
    fig, ax = plt.subplots(figsize=(9, 4.6)); yy = np.arange(len(_idx)); w = .38
    ax.barh(yy + w/2, [_fix[i] for i in _idx], w, label='STEP 19 (medie fisse)', color='#7f8c8d')
    ax.barh(yy - w/2, [_res_tv['exp(coef)'][i] for i in _idx], w, label='STEP 20 (tempo-variante)', color='#c0392b')
    ax.axvline(1, color='k', ls='--'); ax.set_yticks(yy); ax.set_yticklabels(_idx)
    ax.set_xlabel('Hazard Ratio (+1 dev. std.)'); ax.legend(); ax.set_title('RQ2 leakage-free: covariate fisse vs tempo-varianti')
    plt.tight_layout(); plt.savefig(OUT_DIR / 'rq2_timefixed_vs_timevarying.png', dpi=110); plt.close()
except Exception as _e:
    print("  (grafico confronto saltato)")
print("Step 18-19-20 (RQ2 leakage-free) completati.")



# ═══════════════════════════════════════════════════════════════════════════
# La definizione di fallimento (cessazione collettiva) rende la DIMENSIONE del
# team un discriminante quasi meccanico. Quindi controlliamo esplicitamente la SCALA del progetto
# con piu' indicatori strutturali e verifichiamo se i fattori socio-tecnici
# (bus factor, debito tecnico) mantengono un effetto AL NETTO della scala.
# Controlli di scala (VIF < soglia, non derivati dall'abbandono):
#   - log_n_dev            : dimensione del team (n. contributori)
#   - log_commits_per_win  : intensita' di attivita' (commit/finestra, norm. per vita)
#   - log_loc              : dimensione del codice (lines of code)
# L'eta' del progetto e' controllata dal Cox tramite l'asse del tempo (durata).
# ═══════════════════════════════════════════════════════════════════════════
import statsmodels.api as _sm3
from statsmodels.discrete.discrete_model import Logit as _Logit3
from lifelines import CoxPHFitter as _CoxPH3

_pl2 = df.groupby('project_url').agg(
    n_dev=('developer_id', 'nunique'), n_win=('time_window_rank', 'nunique'),
    tot_commits=('commits_count', 'sum'), loc=('project_loc', 'max') if 'project_loc' in df.columns else ('commits_count', 'sum'),
    SD_truck=('stqf_global_truck', 'mean'), SD_community=('community_smell_count', 'mean'),
    TD_trad=('traditional_smell_count', 'mean'), TD_vuln=('vulnerability_count', 'mean'),
    proj_first=('time_window_rank', 'min'), proj_last=('time_window_rank', 'max'),
    proj_failed=('proj_failed', 'max')).reset_index()
_pl2['log_n_dev'] = np.log1p(_pl2['n_dev'])
_pl2['log_commits_per_win'] = np.log1p(pd.to_numeric(_pl2['tot_commits'], errors='coerce').fillna(0) / _pl2['n_win'])
_pl2['log_loc'] = np.log1p(pd.to_numeric(_pl2['loc'], errors='coerce').fillna(0))
_pl2['duration'] = _pl2['proj_last'] - _pl2['proj_first'] + 1

SCALE = ['log_n_dev', 'log_commits_per_win', 'log_loc']       # controlli strutturali / scala
SOCTEC = ['SD_truck', 'SD_community', 'TD_trad', 'TD_vuln']    # fattori socio-tecnici (leakage-free)
ALLP = SOCTEC + SCALE
_Z2 = _pl2[ALLP].copy()
for c in ALLP:
    s = _Z2[c].std(); _Z2[c] = (_Z2[c] - _Z2[c].mean()) / s if s > 0 else 0.0



# ─────────────────────────────────────────────
# STEP 21 — RQ2 ROBUSTEZZA: CONTROLLI STRUTTURALI / DI SCALA
# ─────────────────────────────────────────────

print("\n")
print("STEP 21 — RQ2 robustezza: fattori socio-tecnici al netto della SCALA del progetto")

print(f"  Controlli di scala: {SCALE}")
_ml2 = _Logit3(_pl2['proj_failed'], _sm3.add_constant(_Z2[ALLP])).fit(disp=0, maxiter=200)
print(f"  [Logit] McFadden R2 = {1 - _ml2.llf/_ml2.llnull:.3f}  (OR per +1 dev. std.)")
for p in ALLP:
    orr = np.exp(_ml2.params[p]); pv = _ml2.pvalues[p]
    sig = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else 'ns'
    tag = '(socio-tecnico)' if p in SOCTEC else '(scala)'
    print(f"    {p:20s} OR={orr:5.2f}  p={pv:.3f}  {sig:3s} {tag}")

# ── STEP 21b — Cox con controlli di scala estesi (eta' via asse del tempo) ──
_cox2 = pd.concat([_pl2[['duration', 'proj_failed']].rename(columns={'proj_failed': 'event'}), _Z2[ALLP]], axis=1)
_cph2 = _CoxPH3(penalizer=0.1); _cph2.fit(_cox2, duration_col='duration', event_col='event')
print(f"  [Cox] concordance = {_cph2.concordance_index_:.3f}  (HR per +1 dev. std.; eta' controllata dall'asse tempo)")
_s2 = _cph2.summary[['exp(coef)', 'p']]
for p in ALLP:
    hr = _s2.loc[p, 'exp(coef)']; pv = _s2.loc[p, 'p']
    sig = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else 'ns'
    tag = '(socio-tecnico)' if p in SOCTEC else '(scala)'
    print(f"    {p:20s} HR={hr:5.2f}  p={pv:.3f}  {sig:3s} {tag}")
_s2.to_csv(OUT_DIR / 'rq2_robustness_scale_cox.csv')
pd.DataFrame([{'predictor': p, 'odds_ratio': np.exp(_ml2.params[p]), 'p_logit': _ml2.pvalues[p],
               'HR': _s2.loc[p, 'exp(coef)'], 'p_cox': _s2.loc[p, 'p'],
               'tipo': 'socio-tecnico' if p in SOCTEC else 'scala'} for p in ALLP]
            ).to_csv(OUT_DIR / 'rq2_robustness_scale.csv', index=False)


# ─────────────────────────────────────────────
# confronto: SD/TD con controllo MINIMO (solo n_dev, STEP 19) vs ESTESO (STEP 21)
print("  --- I fattori socio-tecnici SOPRAVVIVONO al controllo di scala esteso? ---")
try:
    _base = pd.read_csv(OUT_DIR / 'rq2_project_cox.csv', index_col=0)['exp(coef)']
    for p in SOCTEC:
        b = _base.get(p, np.nan); f = _s2.loc[p, 'exp(coef)']; pv = _s2.loc[p, 'p']
        verdetto = 'resta' if pv < .05 else 'si attenua/sparisce'
        print(f"    {p:13s} HR: n_dev-only={b:.2f}  ->  scala-estesa={f:.2f} (p={pv:.3f}) -> {verdetto}")
except Exception:
    pass

fig, ax = plt.subplots(figsize=(9, 4.6)); _hr2 = _s2.loc[ALLP, 'exp(coef)']
cols = ['#c0392b' if _s2.loc[p, 'p'] < .05 and _hr2[p] > 1 else ('#2980b9' if _s2.loc[p, 'p'] < .05 else '#bdc3c7') for p in ALLP]
ax.barh(range(len(ALLP)), [_hr2[p] for p in ALLP], color=cols)
ax.set_yticks(range(len(ALLP))); ax.set_yticklabels([f"{p} {'(ST)' if p in SOCTEC else '(scala)'}" for p in ALLP])
ax.axvline(1, color='k', ls='--'); ax.set_xlabel('Hazard Ratio (+1 dev. std.)')
ax.set_title('STEP 21 — RQ2: socio-tecnico al netto della scala (rosso=rischio sign., grigio=ns)')
plt.tight_layout(); plt.savefig(OUT_DIR / 'rq2_robustness_scale.png', dpi=110); plt.close()
print("Step 21 (RQ2 robustezza scala) completato.")



# ═══════════════════════════════════════════════════════════════════════════
# STEP 22 — SENSIBILITA': DEFINIZIONE ALTERNATIVA TEMPORALE 
# La definizione PRINCIPALE fatta nello STEP 3 e' ora: inattivo se < 1 commit/mese in
# media nell'ultimo anno di attivita' (v2). Quindi verifichiamo la robustezza con la
# definizione ALTERNATIVA TEMPORALE (v1): un progetto e' inattivo se non ha attivita'
# nell'ultima finestra osservabile (ultima attivita' >= soglia finestre prima del
# cutoff). Rieseguiamo il modello RQ2 sotto v1 e confrontiamo con la principale v2.
# NB: v1 e' temporale -> il Cox di sopravvivenza e' il modello appropriato per essa.
# ═══════════════════════════════════════════════════════════════════════════
import statsmodels.api as _sm4
from lifelines import CoxPHFitter as _CoxPH4

# aggregazione progetto + definizione alternativa v1 (temporale) e esito principale v2
_agg = df.groupby('project_url').agg(
    n_dev=('developer_id', 'nunique'), n_win=('time_window_rank', 'nunique'),
    tot_commits=('commits_count', 'sum'), loc=('project_loc', 'max') if 'project_loc' in df.columns else ('commits_count', 'sum'),
    SD_truck=('stqf_global_truck', 'mean'), SD_community=('community_smell_count', 'mean'),
    TD_trad=('traditional_smell_count', 'mean'), TD_vuln=('vulnerability_count', 'mean'),
    proj_first=('time_window_rank', 'min'), proj_last=('time_window_rank', 'max'),
    proj_failed_v2=('proj_failed', 'max')).reset_index()
_agg['inactive_v1'] = ((GLOBAL_LAST - _agg['proj_last']) >= INACTIVITY_THRESHOLD).astype(int)
_agg['life'] = _agg['proj_last'] - _agg['proj_first'] + 1
_agg['log_n_dev'] = np.log1p(_agg['n_dev'])
_agg['log_commits_per_win'] = np.log1p(pd.to_numeric(_agg['tot_commits'], errors='coerce').fillna(0) / _agg['n_win'])
_agg['log_loc'] = np.log1p(pd.to_numeric(_agg['loc'], errors='coerce').fillna(0))


# ─────────────────────────────────────────────
# STEP 22 — SENSIBILITA': DEFINIZIONE ALTERNATIVA TEMPORALE 
# ─────────────────────────────────────────────

print("\n")
print("STEP 22 — Sensibilita': definizione alternativa TEMPORALE (v1) vs principale (v2)")

print(f"  principale v2 (<1 commit/mese):    INATTIVI {int(_agg.proj_failed_v2.sum())} | ATTIVI {int((_agg.proj_failed_v2==0).sum())}")
print(f"  alternativa v1 (ultima finestra):  INATTIVI {int(_agg.inactive_v1.sum())} | ATTIVI {int((_agg.inactive_v1==0).sum())}")
print(f"  progetti che cambiano etichetta v2<->v1: {int((_agg.proj_failed_v2!=_agg.inactive_v1).sum())} / {len(_agg)}")

SOC = ['SD_truck', 'SD_community', 'TD_trad', 'TD_vuln']
SCA = ['log_n_dev', 'log_commits_per_win', 'log_loc']
_Zv = _agg[SOC + SCA].copy()
for c in SOC + SCA:
    s2 = _Zv[c].std(); _Zv[c] = (_Zv[c] - _Zv[c].mean()) / s2 if s2 > 0 else 0.0

# Cox di sopravvivenza sotto v1 (temporale) con controlli di scala estesi
_coxv = pd.concat([_agg[['life', 'inactive_v1']].rename(columns={'life': 'duration', 'inactive_v1': 'event'}), _Zv[SOC + SCA]], axis=1)
_cphv = _CoxPH4(penalizer=0.1); _cphv.fit(_coxv, duration_col='duration', event_col='event')
print(f"  [Cox v1 + scala estesa] concordance = {_cphv.concordance_index_:.3f}  (HR per +1 dev. std.)")
_sv = _cphv.summary[['exp(coef)', 'p']]
for p in SOC + SCA:
    hr = _sv.loc[p, 'exp(coef)']; pv = _sv.loc[p, 'p']
    sig = '***' if pv < .001 else '**' if pv < .01 else '*' if pv < .05 else 'ns'
    tag = '(socio-tecnico)' if p in SOC else '(scala)'
    print(f"    {p:20s} HR={hr:5.2f}  p={pv:.3f}  {sig:3s} {tag}")
_sv.to_csv(OUT_DIR / 'rq2_sensitivity_v1_cox.csv')

# confronto: principale v2 (STEP 21) vs alternativa v1
print("  --- I fattori socio-tecnici sono PIU' forti sotto la definizione temporale v1? ---")
try:
    _main = pd.read_csv(OUT_DIR / 'rq2_robustness_scale_cox.csv', index_col=0)['exp(coef)']
    for p in SOC:
        m = _main.get(p, np.nan); v = _sv.loc[p, 'exp(coef)']; pv = _sv.loc[p, 'p']
        print(f"    {p:13s} HR v2(principale)={m:.2f}  ->  v1(alternativa)={v:.2f} (p={pv:.3f})")
except Exception:
    pass

fig, ax = plt.subplots(figsize=(9, 4.6)); _hrv = _sv.loc[SOC + SCA, 'exp(coef)']
cols = ['#c0392b' if _sv.loc[p, 'p'] < .05 and _hrv[p] > 1 else ('#2980b9' if _sv.loc[p, 'p'] < .05 else '#bdc3c7') for p in SOC + SCA]
ax.barh(range(len(SOC + SCA)), [_hrv[p] for p in SOC + SCA], color=cols)
ax.set_yticks(range(len(SOC + SCA))); ax.set_yticklabels([f"{p} {'(ST)' if p in SOC else '(scala)'}" for p in SOC + SCA])
ax.axvline(1, color='k', ls='--'); ax.set_xlabel('Hazard Ratio (+1 dev. std.)')
ax.set_title('STEP 22 - RQ2 sensibilita: definizione temporale alternativa (v1)')
plt.tight_layout(); plt.savefig(OUT_DIR / 'rq2_sensitivity_v1.png', dpi=110); plt.close()
print("Step 22 (sensibilita' definizione alternativa v1) completato.")

# Sotto la definizione temporale v1 (inattivo se nessuna attività nell'ultima finestra; split 294/38, molto
# sbilanciato), gli effetti socio-tecnici appaiono molto più forti: bus factor HR=1,80 (p<0,001), smell
# tradizionali HR=1,27 (p=0,005), vulnerabilità HR=1,20 (p=0,01), al netto della scala. Il confronto è
# istruttivo: la forza dei risultati socio-tecnici a livello progetto dipende dalla definizione di
# fallimento. La v1 (timing) enfatizza i progetti "fermi da tempo" e fa emergere il bus factor; la v2
# (intensità), più robusta all'attività residua e ai progetti completati, ridimensiona quel segnale. L'unica
# evidenza stabile a entrambe le definizioni è il debito di sicurezza (vulnerabilità).


# Domande di ricerca
# 
# RQ1 (livello developer). "Quali fattori socio-tecnici aumentano il rischio di abbandono dei developer?" 
# unità = coppia (progetto, developer);
# evento = il developer smette ≥ 4 finestre prima della fine del suo progetto;
# censura = ancora attivo entro la soglia dall'orizzonte del progetto.
#
# RQ2 (livello progetto). "Considerando il fallimento del progetto come la cessazione completa della
# comunità di sviluppo, ossia l'assenza di attività nell'ultima finestra osservabile, quali 
# fattori socio-tecnici distinguono i progetti che sopravvivono da quelli che non sopravvivono?"
# unità = progetto;
# evento = il progetto muore ≥ 4 finestre prima del cutoff del dataset (set-2025);
# censura = ancora attivo al cutoff.

# Il fallimento del progetto è, per costruzione, la cessazione collettiva dell'attività — cioè, di
# fatto, tutti i developer che smettono. Ne segue che non si può concludere che "l'abbandono dei
# developer causa il fallimento del progetto": sarebbe tautologico, perché il secondo evento è costruito a
# partire dal primo.


# RQ1 — Cosa fa abbandonare i developer?

# I dati sono stati recuperati in quattro STEP:
#  ———— STEP 7 ->Logit early-stage (McFadden R²≈0,11): rischio significativo per ratio_smelly_devs (OR=2,01),
#       ratio_smelly_quitters (OR=2,26), global_truck (OR=2,38), communicability (OR=1,77); protettivi commit
#       (0,59) e anzianità (0,84). Debito tecnico non significativo;
# 
# ———— STEP 8 -> Cox PH (concordance 0,89): turnover HR=1,71 (IC 1,24–2,36), global_truck HR=2,13,
#      smelly_quitters HR=1,35. Anzianità protettiva ma non dominante. Debito tecnico non significativo;
# 
# ———— STEP 15 -> Cox tempo-variante (2.173 developer, 14.179 intervalli): con le metriche misurate finestra per
#      finestra, il turnover corrente alza il rischio istantaneo di abbandono (HR=1,11, p<0,001). Commit e
#      debito tecnico hanno HR<1 (proxy di attività);
# 
# ———— STEP 16 -> Firma pre-abbandono (1.460 abbandoni): avvicinandosi all'uscita, commit −30%, debito tecnico
#      prodotto −60%, engagement −8%, mentre il turnover sale +23% (tutti p ≪ 0,001);

# ———— Risposta a RQ1: il debito sociale, e in particolare il turnover del team, aumenta il rischio di abbandono del developer, in modo robusto e coerente.
# Il debito tecnico non predice l'abbandono individuale: si comporta da indicatore di attività.



# RQ2 — Cosa fa morire i progetti

# I dati sono stati recuperati in sette STEP:
# ———— STEP 10 — La dimensione del team NON discrimina (mediana 13 sia negli attivi sia negli inattivi, MWU p=0,61). I
#      discriminanti significativi sono strutturali/di attività: gli attivi (mantenuti intensamente) hanno più
#      st-congruence (p<0,001), più density (p<0,001) e più smell tradizionali (p=0,02, semplicemente più codice);
#      gli inattivi hanno relativamente più smelly-devs (p<0,001) e community smells (p<0,01). Turnover,
#      smelly-quitters e vulnerabilità non distinguono i gruppi;
# 
# ———— STEP 12 — Ownership. I core (truck-factor) contribuiscono ~90× i periferici e abbandonano molto meno
#      (43% vs 88%): sono i "reggenti" del progetto. La concentrazione (Gini) però non differisce più tra attivi e
#      inattivi sotto v2 (0,77 vs 0,77, ns).
# 
# ———— STEP 17 — Confondimento della dimensione. Controllando log(n_dev) una metrica alla volta, turnover e
#      smelly-quitters cambiano segno (da "protettivi" a rischio);
# 
# ———— STEP 18 — Logit multivariato leakage-free (modello di RIFERIMENTO per v2; McFadden 0,15). Con la
#      definizione per intensità, i fattori socio-tecnici sono in gran parte non significativi: l'unico effetto
#      forte è il bus factor, ma protettivo (OR=0,39, p<0,001), cioè i progetti con conoscenza più concentrata
#      risultano meno inattivi, probabilmente perché un core solido -> attività sostenuta. Smell tradizionali
#      protettivi (OR=0,68, p=0,04); dimensione, community, st-congruence, vulnerabilità: non significativi.;
# 
# ———— STEP 19 — Cox di sopravvivenza del progetto, medie fisse (concordance 0,76; durata=vita del progetto, evento=morte).
#      HR per +1 deviazione standard, HR>1 = muore prima;
# 
# ———— STEP 19-21 — Cox (lente secondaria). Sotto v2 dominano gli indicatori di scala: dimensione
#      (HR=0,67), intensità di attività (HR=0,74), codice (HR=0,86), tutti protettivi. L'unico fattore
#      socio-tecnico che sopravvive al controllo di scala è, debolmente, il debito di sicurezza (vulnerabilità)
#      (HR=1,27, p=0,04). Bus factor, smell tradizionali, community: non significativi. (Nel tempo-variante STEP 20
#      tutto collassa verso HR≈1, come per RQ1.)


# Risposta a RQ2:
# i fattori socio-tecnici NON distinguono in modo robusto i progetti attivi da quelli inattivi al netto della scala. La sopravvivenza è
# dominata da scala e intensità di attività; l'unico segnale socio-tecnico debole e stabile è il debito
# di sicurezza (vulnerabilità). Il bus factor non è un fattore di rischio robusto (anzi, nel logit è
# protettivo). È un risultato più sobrio ma più solido: quando "fallimento" non è solo "essersi fermati da
# poco" ma "non essere stati mantenuti", il debito socio-tecnico spiega poco oltre l'attività stessa.


# Sintesi:
# fattore                              RQ1(abbandono developer)                RQ2(morte progetto)
#
# Turnover(debito sociale)                aumenta il rischio                       tautologico
# Truck / bus factor                      aumenta il rischio             non robusto (protettivo nel logit)
# Debito tecnico(vulnerabilità)               non predice                       aumenta il rischio
# Debito tecnico(smell)                       non predice            non robusto (dipendente dalla definizione)
# Dimensione/scala/attività                   protettiva                         fattore dominante

# A livello developer, è il churn sociale (turnover)  guida l'abbandono individuale, coerentemente in più modelli;
# A livello progetto,  con una definizione robusta di inattività (v2), il debito socio-tecnico spiega
# poco oltre la scala e l'intensità di attività; solo le vulnerabilità mostrano un segnale debole e stabile.
# Gli effetti forti che si vedono sotto la definizione temporale (v1) non sono robusti al cambio di definizione.
# La conclusione complessiva è quindi asimmetrica: il debito sociale conta per l'attrito dei singoli, ma
# non si "somma" in modo semplice in un effetto socio-tecnico robusto sulla mortalità del progetto — che
# dipende soprattutto da quanto e quanto intensamente il progetto viene mantenuto.

# ─────────────────────────────────────────────
# TERMINE ANALISI
# ─────────────────────────────────────────────
print("\n")
print("ANALISI COMPLETATA")

print(f"Output in: {OUT_DIR}")
print("=" * 65)
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name:52s} ({f.stat().st_size:>9,} bytes)")
