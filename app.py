"""
Application de prédiction du risque d'échec virologique.

Interface Gradio autonome : charge le pipeline entraîné (modele_prediction_cv.joblib)
et les métadonnées des variables (metadonnees_variables.json), puis expose un
formulaire de saisie et affiche la probabilité de non-suppression de la prochaine
charge virale.

Ce fichier est déployé tel quel sur Hugging Face Spaces (voir section 10.3 du notebook).
"""
import json
import math
import os
import re
import tempfile
import unicodedata
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import gradio as gr

# ------------------------------------------------------------------
# ZeroGPU : le Space étant sur matériel ZeroGPU, HF exige qu'au moins une fonction
# soit décorée par @spaces.GPU, sinon erreur "No @spaces.GPU function detected".
# Ce modèle (Random Forest) tourne sur CPU : la fonction ci-dessous n'est jamais
# appelée, donc aucun quota GPU n'est consommé. Sans le module `spaces`
# (CPU basic, exécution locale), le bloc est simplement ignoré.
# ------------------------------------------------------------------
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_startup_marker():
        return None
except ImportError:
    pass

# ------------------------------------------------------------------
# Chargement du modèle et des métadonnées (une seule fois, au démarrage)
# ------------------------------------------------------------------
ARTEFACT = joblib.load("modele_prediction_cv.joblib")
PIPELINE = ARTEFACT["pipeline"]
SEUIL = ARTEFACT["seuil_optimal"]
COLONNES = ARTEFACT["colonnes_attendues"]

with open("metadonnees_variables.json", encoding="utf-8") as f:
    METADONNEES = json.load(f)


def _nombre(valeur, defaut):
    """Convertit en float ; renvoie `defaut` si la valeur est absente ou NaN."""
    try:
        v = float(valeur)
        return defaut if math.isnan(v) else v
    except (TypeError, ValueError):
        return defaut


def construire_composants_entree():
    """Génère un composant Gradio par variable, dans l'ordre attendu par le modèle."""
    composants = []
    for col in COLONNES:
        info = METADONNEES[col]
        if info["type"] == "numerique":
            # Champ numérique libre : on peut taper des décimales (ex. 8.06).
            # La plage observée à l'entraînement est affichée à titre indicatif.
            mini = _nombre(info.get("min"), 0.0)
            maxi = _nombre(info.get("max"), mini)
            defaut = round(_nombre(info.get("valeur_defaut"), mini), 2)
            composants.append(gr.Number(
                value=defaut, label=col, precision=None, step=0.1,
                info=f"Plage observée : {mini:.2f} à {maxi:.2f}"))
        else:
            choix = [m for m in info.get("modalites", []) if pd.notna(m)]
            defaut = info.get("valeur_defaut")
            if defaut not in choix:
                choix = choix + [defaut] if pd.notna(defaut) and not choix else choix
                defaut = choix[0] if choix else None
            composants.append(gr.Dropdown(choices=choix, value=defaut, label=col))
    return composants


def predire(*valeurs):
    """Assemble les valeurs saisies en une ligne de DataFrame et retourne la prédiction."""
    if any(v is None for v in valeurs):
        return None, "⚠️ Veuillez renseigner toutes les valeurs numériques."
    ligne = pd.DataFrame([dict(zip(COLONNES, valeurs))])
    probabilite = PIPELINE.predict_proba(ligne)[0, 1]

    decision = "NON SUPPRIME (risque d'échec)" if probabilite >= SEUIL else "SUPPRIME"
    etiquette = {
        "NON SUPPRIME (risque)": float(probabilite),
        "SUPPRIME": float(1 - probabilite),
    }
    message = (f"Probabilité d'échec virologique : {probabilite:.1%}\n"
               f"Seuil de décision retenu : {SEUIL:.1%}\n"
               f"Décision : {decision}")
    return etiquette, message


# ------------------------------------------------------------------
# Analyse par lot : lecture d'un fichier de patients, prédiction et analyse détaillée
# ------------------------------------------------------------------
COLONNES_NUM = [c for c in COLONNES if METADONNEES[c]["type"] == "numerique"]
COLONNES_CAT = [c for c in COLONNES if METADONNEES[c]["type"] != "numerique"]
REFERENCE = {c: METADONNEES[c]["valeur_defaut"] for c in COLONNES}
MAX_PATIENTS = 2000          # limite de sécurité pour rester rapide sur un Space gratuit
SEUIL_IMPACT = 0.5           # en points de pourcentage : en dessous, un facteur est jugé neutre
NOMS_ID = {"id", "id patient", "identifiant", "patient", "code patient", "code",
           "numero", "n", "matricule", "dossier", "num dossier", "numero dossier"}
COL_IMPACT = "Impact (points de %)"
COL_PROBA = "Probabilité d'échec (%)"


def _norm(texte):
    """Minuscules, sans accents ni ponctuation : sert à rapprocher les noms de colonnes."""
    t = unicodedata.normalize("NFKD", str(texte)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _fmt(valeur):
    """Formate une valeur pour l'affichage (2 décimales max, 'manquant' si vide)."""
    if valeur is None or (not isinstance(valeur, str) and pd.isna(valeur)):
        return "manquant"
    if isinstance(valeur, (int, float, np.integer, np.floating)):
        return f"{float(valeur):.2f}".rstrip("0").rstrip(".")
    return str(valeur)


def _lire_fichier(chemin):
    ext = os.path.splitext(chemin)[1].lower()
    if ext not in (".xlsx", ".xlsm", ".csv", ".txt", ".tsv"):
        raise gr.Error("Format non pris en charge : utilisez un fichier .xlsx ou .csv.")
    try:
        if ext in (".xlsx", ".xlsm"):
            return pd.read_excel(chemin, sheet_name=0)   # première feuille
        try:
            return pd.read_csv(chemin, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(chemin, sep=None, engine="python", encoding="latin-1")
    except Exception as erreur:
        raise gr.Error(f"Impossible de lire le fichier : {erreur}")


def _convertir_numerique(serie):
    """Convertit en nombres (accepte la virgule décimale). Renvoie (valeurs, invalides)."""
    texte = serie.map(lambda v: np.nan if pd.isna(v) else str(v).strip().replace(",", "."))
    texte = texte.replace("", np.nan)
    valeurs = pd.to_numeric(texte, errors="coerce")
    return valeurs, texte.notna() & valeurs.isna()


def _convertir_categoriel(serie, modalites):
    """Rapproche les valeurs des modalités connues (sans tenir compte casse/accents)."""
    canon = {_norm(m): m for m in modalites}
    resultat, inconnues = [], []
    for v in serie:
        if pd.isna(v) or str(v).strip() == "":
            resultat.append(np.nan)
            inconnues.append(False)
        elif _norm(v) in canon:
            resultat.append(canon[_norm(v)])
            inconnues.append(False)
        else:
            resultat.append(str(v).strip())
            inconnues.append(True)
    return pd.Series(resultat, index=serie.index, dtype=object), pd.Series(inconnues, index=serie.index)


def _preparer(brut):
    """Valide le fichier, convertit les colonnes. Renvoie (X, identifiants, lignes, alertes)."""
    correspondance = {_norm(c): c for c in COLONNES}
    renommage, col_id = {}, None
    for c in brut.columns:
        n = _norm(c)
        if n in correspondance and correspondance[n] not in renommage.values():
            renommage[c] = correspondance[n]
        elif col_id is None and n in NOMS_ID:
            col_id = c
    df = brut.rename(columns=renommage)

    manquantes = [c for c in COLONNES if c not in df.columns]
    if manquantes:
        raise gr.Error(
            f"{len(manquantes)} colonne(s) manquante(s) dans le fichier : "
            + ", ".join(manquantes)
            + ". Téléchargez le modèle de fichier pour obtenir la structure attendue.")

    df = df.copy()
    df["__ligne"] = df.index + 2                 # numéro de ligne dans Excel (1 = en-têtes)
    df = df[df[COLONNES].notna().any(axis=1)]    # ignore les lignes entièrement vides
    if df.empty:
        raise gr.Error("Aucun patient trouvé dans le fichier.")
    if len(df) > MAX_PATIENTS:
        raise gr.Error(f"Le fichier contient {len(df)} patients ; la limite est de {MAX_PATIENTS}.")

    lignes = df["__ligne"].tolist()
    if col_id is not None:
        ids = [str(v).strip() if pd.notna(v) and str(v).strip() else f"Ligne {l}"
               for v, l in zip(df[col_id], lignes)]
    else:
        ids = [f"Ligne {l}" for l in lignes]
    doublons = pd.Series(ids).duplicated(keep=False).tolist()
    ids = [f"{i} (ligne {l})" if d else i for i, l, d in zip(ids, lignes, doublons)]

    alertes = [[] for _ in range(len(df))]
    X = pd.DataFrame(index=df.index)
    manquants = [[] for _ in range(len(df))]
    invalides = [[] for _ in range(len(df))]
    hors_plage = [[] for _ in range(len(df))]
    inconnus = [[] for _ in range(len(df))]

    for col in COLONNES_NUM:
        valeurs, invalide = _convertir_numerique(df[col])
        X[col] = valeurs
        mini, maxi = METADONNEES[col]["min"], METADONNEES[col]["max"]
        for k, (v, inv) in enumerate(zip(valeurs, invalide)):
            if inv:
                invalides[k].append(f"{col} = « {df[col].iloc[k]} »")
            elif pd.isna(v):
                manquants[k].append(col)
            elif v < mini or v > maxi:
                hors_plage[k].append(f"{col} = {_fmt(v)}")
    for col in COLONNES_CAT:
        valeurs, inconnue = _convertir_categoriel(df[col], METADONNEES[col]["modalites"])
        X[col] = valeurs
        for k, (v, inc) in enumerate(zip(valeurs, inconnue)):
            if inc:
                inconnus[k].append(f"{col} = « {v} »")
            elif pd.isna(v):
                manquants[k].append(col)

    for k in range(len(df)):
        if manquants[k]:
            alertes[k].append("Valeur manquante (imputée par le modèle) : " + ", ".join(manquants[k]))
        if invalides[k]:
            alertes[k].append("Valeur non numérique (traitée comme manquante) : " + ", ".join(invalides[k]))
        if inconnus[k]:
            alertes[k].append("Modalité inconnue (ignorée par le modèle) : " + ", ".join(inconnus[k]))
        if hors_plage[k]:
            alertes[k].append("Hors plage observée à l'entraînement : " + ", ".join(hors_plage[k]))

    return X[COLONNES], ids, lignes, [" | ".join(a) if a else "" for a in alertes]


def _calculer_impacts(X):
    """Probabilité de chaque patient + impact de chaque variable.

    Impact d'une variable = probabilité du patient moins la probabilité obtenue en
    remplaçant cette seule variable par la valeur de référence (valeur par défaut
    des métadonnées). Positif = la variable augmente le risque par rapport au
    profil de référence. Analyse de sensibilité indicative (interactions ignorées).
    """
    base = PIPELINE.predict_proba(X)[:, 1]
    impacts = np.zeros((len(X), len(COLONNES)))
    for j, col in enumerate(COLONNES):
        X_ref = X.copy()
        X_ref[col] = REFERENCE[col]
        impacts[:, j] = base - PIPELINE.predict_proba(X_ref)[:, 1]
    return base, impacts * 100


def _niveau(p):
    if p >= SEUIL:
        return "Élevé"
    return "Modéré" if p >= SEUIL / 2 else "Faible"


def _facteurs(impacts_patient, valeurs_patient, sens, n=3):
    ordre = np.argsort(-impacts_patient) if sens > 0 else np.argsort(impacts_patient)
    textes = [f"{COLONNES[j]} = {_fmt(valeurs_patient[j])} ({impacts_patient[j]:+.1f} pt)"
              for j in ordre[:n] if sens * impacts_patient[j] >= SEUIL_IMPACT]
    return " ; ".join(textes) if textes else "—"


def _construire_tables(X, ids, lignes, alertes, base, impacts):
    lignes_resume, lignes_detail = [], []
    for i, patient in enumerate(ids):
        valeurs = X.iloc[i].tolist()
        p = float(base[i])
        lignes_resume.append({
            "Patient": patient,
            "Ligne": lignes[i],
            COL_PROBA: round(p * 100, 1),
            "Décision": "NON SUPPRIME (risque d'échec)" if p >= SEUIL else "SUPPRIME",
            "Niveau de risque": _niveau(p),
            "Facteurs qui augmentent le risque": _facteurs(impacts[i], valeurs, +1),
            "Facteurs qui réduisent le risque": _facteurs(impacts[i], valeurs, -1),
            "Alertes sur les données": alertes[i],
        })
        for j, col in enumerate(COLONNES):
            imp = float(impacts[i, j])
            lignes_detail.append({
                "Patient": patient,
                "Variable": col,
                "Valeur": _fmt(valeurs[j]),
                "Valeur de référence": _fmt(REFERENCE[col]),
                COL_IMPACT: round(imp, 2),
                "Effet": ("↑ Augmente le risque" if imp >= SEUIL_IMPACT
                          else "↓ Réduit le risque" if imp <= -SEUIL_IMPACT
                          else "≈ Neutre"),
            })
    resume = pd.DataFrame(lignes_resume).sort_values(
        COL_PROBA, ascending=False, kind="stable").reset_index(drop=True)
    return resume, pd.DataFrame(lignes_detail)


def _ajuster_largeurs(feuille):
    for colonne in feuille.columns:
        largeur = max(len(str(c.value)) if c.value is not None else 0 for c in colonne[:60])
        feuille.column_dimensions[colonne[0].column_letter].width = min(max(largeur + 2, 10), 70)


def _exporter(resume, detail):
    dossier = tempfile.mkdtemp(prefix="analyse_cv_")
    chemin = os.path.join(dossier, f"analyse_patients_{datetime.now():%Y%m%d_%H%M}.xlsx")
    auc = ARTEFACT.get("roc_auc_test")
    parametres = pd.DataFrame({
        "Paramètre": ["Date de l'analyse", "Nombre de patients", "Modèle",
                      "Date d'entraînement du modèle", "ROC-AUC (test)",
                      "Seuil de décision (NON SUPPRIME si probabilité ≥ seuil)",
                      "Niveaux de risque", "Méthode d'analyse des facteurs"],
        "Valeur": [f"{datetime.now():%d/%m/%Y %H:%M}", len(resume),
                   str(ARTEFACT.get("nom_modele", "n/d")),
                   str(ARTEFACT.get("date_entrainement", "n/d")),
                   f"{auc:.3f}" if isinstance(auc, (int, float)) else str(auc or "n/d"),
                   f"{SEUIL:.1%}",
                   f"Élevé : ≥ {SEUIL:.1%} ; Modéré : de {SEUIL / 2:.1%} à {SEUIL:.1%} ; "
                   f"Faible : < {SEUIL / 2:.1%}",
                   "Impact = probabilité du patient − probabilité en remplaçant la variable "
                   "par sa valeur de référence (valeur par défaut). Indicatif ; "
                   "n'établit pas de lien de cause à effet."],
    })
    with pd.ExcelWriter(chemin, engine="openpyxl") as ecrivain:
        resume.to_excel(ecrivain, sheet_name="Résultats", index=False)
        detail.to_excel(ecrivain, sheet_name="Détail par variable", index=False)
        parametres.to_excel(ecrivain, sheet_name="Paramètres", index=False)
        for feuille in ecrivain.book.worksheets:
            _ajuster_largeurs(feuille)
    return chemin


def creer_modele_fichier():
    """Génère le fichier Excel modèle (feuille « Patients » + feuille « Notice »)."""
    chemin = os.path.join(tempfile.mkdtemp(prefix="modele_cv_"), "modele_fichier_patients.xlsx")
    exemple = pd.DataFrame([{"ID_Patient": "Patient 1", **REFERENCE}])
    notice = []
    for col in COLONNES:
        info = METADONNEES[col]
        if info["type"] == "numerique":
            notice.append({"Variable": col, "Type": "Nombre (décimales acceptées)",
                           "Plage observée / valeurs autorisées":
                               f"{_fmt(info['min'])} à {_fmt(info['max'])}",
                           "Valeur d'exemple": _fmt(info["valeur_defaut"])})
        else:
            notice.append({"Variable": col, "Type": "Texte (liste de choix)",
                           "Plage observée / valeurs autorisées": " / ".join(info["modalites"]),
                           "Valeur d'exemple": str(info["valeur_defaut"])})
    with pd.ExcelWriter(chemin, engine="openpyxl") as ecrivain:
        exemple.to_excel(ecrivain, sheet_name="Patients", index=False)
        pd.DataFrame(notice).to_excel(ecrivain, sheet_name="Notice", index=False)
        for feuille in ecrivain.book.worksheets:
            _ajuster_largeurs(feuille)
    return chemin


def _vue_patient(patient, etat):
    """Entête, tableau et graphique de l'analyse détaillée d'un patient."""
    resume, detail = etat["resume"], etat["detail"]
    ligne = resume[resume["Patient"] == patient].iloc[0]
    entete = (f"### {patient}\n"
              f"**Probabilité d'échec virologique : {ligne[COL_PROBA]:.1f} %** "
              f"(seuil : {SEUIL:.1%}) — {ligne['Décision']} — niveau de risque **{ligne['Niveau de risque']}**")
    if ligne["Alertes sur les données"]:
        entete += f"\n\n⚠️ {ligne['Alertes sur les données']}"
    d = detail[detail["Patient"] == patient].drop(columns="Patient")
    ordre = d[COL_IMPACT].abs().sort_values(ascending=False, kind="stable").index
    d = d.loc[ordre].reset_index(drop=True)
    return entete, d, d.head(10)[["Variable", COL_IMPACT]]


def analyser_lot(chemin):
    """Point d'entrée de l'onglet « Analyse par lot »."""
    if not chemin:
        raise gr.Error("Veuillez d'abord charger un fichier de patients (.xlsx ou .csv).")
    X, ids, lignes, alertes = _preparer(_lire_fichier(chemin))
    base, impacts = _calculer_impacts(X)
    resume, detail = _construire_tables(X, ids, lignes, alertes, base, impacts)
    fichier_resultats = _exporter(resume, detail)

    n = len(resume)
    n_risque = int((resume["Décision"] != "SUPPRIME").sum())
    n_alertes = int((resume["Alertes sur les données"] != "").sum())
    texte = (f"**{n} patient(s) analysé(s)** — **{n_risque}** ({n_risque / n:.0%}) prédit(s) "
             f"NON SUPPRIME (probabilité ≥ {SEUIL:.1%}). Probabilité moyenne d'échec : "
             f"{resume[COL_PROBA].mean():.1f} %.")
    if n_alertes:
        texte += f"\n\n⚠️ {n_alertes} ligne(s) avec des alertes sur les données (voir la dernière colonne)."
    niveaux = pd.DataFrame({
        "Niveau de risque": ["Élevé", "Modéré", "Faible"],
        "Patients": [int((resume["Niveau de risque"] == k).sum()) for k in ("Élevé", "Modéré", "Faible")]})

    etat = {"resume": resume, "detail": detail}
    choix = resume["Patient"].tolist()
    entete, tableau, graphique = _vue_patient(choix[0], etat)
    return (texte, niveaux, resume, fichier_resultats,
            gr.Dropdown(choices=choix, value=choix[0]), entete, tableau, graphique, etat)


def afficher_detail(patient, etat):
    if not etat or not patient:
        return "", pd.DataFrame(), pd.DataFrame()
    return _vue_patient(patient, etat)


MODELE_FICHIER = creer_modele_fichier()


with gr.Blocks(title="Prédiction de la charge virale") as demo:
    gr.Markdown(
        "# Prédiction du risque d'échec virologique\n"
        "Cet outil estime la probabilité que la **prochaine charge virale** d'un "
        "patient sous ARV soit **NON SUPPRIME**, à partir de son historique "
        "virologique et de son profil clinique.\n\n"
        "⚠️ Outil d'aide à la décision réalisé dans un cadre de formation — "
        "ne remplace pas un avis médical.")

    with gr.Tabs():
        # ---------------------------- Patient unique ----------------------------
        with gr.Tab("Patient unique"):
            with gr.Row():
                with gr.Column():
                    entrees = construire_composants_entree()
                    bouton = gr.Button("Prédire", variant="primary")
                with gr.Column():
                    sortie_probabilite = gr.Label(label="Résultat")
                    sortie_message = gr.Textbox(label="Détail", lines=3)

            bouton.click(fn=predire, inputs=entrees, outputs=[sortie_probabilite, sortie_message])

        # ------------------------- Analyse par lot (fichier) -------------------------
        with gr.Tab("Analyse par lot (fichier)"):
            gr.Markdown(
                "Chargez un fichier **Excel (.xlsx)** ou **CSV** contenant un patient par ligne. "
                "Les colonnes attendues sont celles du **modèle de fichier** (première feuille lue ; "
                "une colonne d'identifiant `ID_Patient` est facultative). "
                "Les décimales avec virgule ou point sont acceptées.")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.DownloadButton("📥 Télécharger le modèle de fichier", value=MODELE_FICHIER)
                    fichier = gr.File(label="Fichier des patients", file_types=[".xlsx", ".csv", ".txt"])
                    bouton_lot = gr.Button("Analyser les patients", variant="primary")
                with gr.Column(scale=2):
                    resume_lot = gr.Markdown()
                    graphique_niveaux = gr.BarPlot(
                        x="Niveau de risque", y="Patients", title="Répartition par niveau de risque",
                        x_title="Niveau de risque", y_title="Nombre de patients", height=260)

            table_lot = gr.Dataframe(
                label="Résultats, du risque le plus élevé au plus faible",
                interactive=False, wrap=True, max_height=420)
            export_lot = gr.File(label="Télécharger les résultats (Excel : résultats, détail par variable, paramètres)")

            gr.Markdown("### Analyse détaillée d'un patient")
            choix_patient = gr.Dropdown(label="Choisir un patient", choices=[], interactive=True)
            detail_entete = gr.Markdown()
            with gr.Row():
                detail_table = gr.Dataframe(
                    label="Effet de chaque variable (triées par importance)",
                    interactive=False, wrap=True, max_height=420)
                detail_graphique = gr.BarPlot(
                    x="Variable", y=COL_IMPACT, title="10 variables les plus influentes",
                    x_title="Variable", y_title="Impact (points de %)", x_label_angle=-45, height=420)
            with gr.Accordion("Comment lire cette analyse ?", open=False):
                gr.Markdown(
                    f"- **Décision** : NON SUPPRIME si la probabilité d'échec est ≥ **{SEUIL:.1%}** "
                    f"(seuil retenu à l'entraînement).\n"
                    f"- **Niveau de risque** : Élevé ≥ {SEUIL:.1%} ; Modéré entre {SEUIL / 2:.1%} et "
                    f"{SEUIL:.1%} ; Faible < {SEUIL / 2:.1%}.\n"
                    "- **Impact d'une variable** : différence de probabilité (en points de %) entre le "
                    "patient et le même patient dont **cette seule variable** est remplacée par la valeur "
                    "de référence (valeur par défaut). Positif = augmente le risque, négatif = le réduit.\n"
                    "- C'est une analyse de **sensibilité indicative** : elle ne prouve pas de lien de "
                    "cause à effet et ignore les interactions entre variables.\n"
                    "- Les valeurs manquantes sont imputées par le modèle (médiane / « Inconnu »), "
                    "comme à l'entraînement ; elles sont signalées dans la colonne « Alertes ».")

            etat = gr.State(None)
            bouton_lot.click(
                fn=analyser_lot, inputs=fichier,
                outputs=[resume_lot, graphique_niveaux, table_lot, export_lot,
                         choix_patient, detail_entete, detail_table, detail_graphique, etat])
            choix_patient.input(
                fn=afficher_detail, inputs=[choix_patient, etat],
                outputs=[detail_entete, detail_table, detail_graphique])

if __name__ == "__main__":
    demo.launch(        
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        ssr_mode=False
    )
