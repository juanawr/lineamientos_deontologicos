"""
Primeros experimentos supervisados (Entrega 2).

E1  Clasificar párrafos de las 9 leyes colombianas según la profesión (9 clases).
E1m Igual que E1, pero con los nombres de profesión enmascarados (control de fuga de información).
E3  Clasificar párrafos de las 9 profesiones usando todos sus documentos (ley + código
    internacional), con y sin enmascarado. Es el experimento principal de la propuesta.
E2  Entrenar con las leyes colombianas y evaluar con los códigos internacionales de las mismas
    9 profesiones (generalización a otro documento y otra redacción).

Validación: StratifiedGroupKFold (5 pliegues) agrupando por artículo/sección, para que
párrafos del mismo artículo no queden repartidos entre entrenamiento y prueba.
Métrica principal: F1-macro (las clases están desbalanceadas).

Uso: python src/experimentos.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from comun import cargar, enmascarar, PROFESIONES_CON_LEY

SEMILLA = 42


def modelos():
    tfidf = lambda: TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return {
        "Baseline (clase mayoritaria)": make_pipeline(tfidf(), DummyClassifier(strategy="most_frequent")),
        "TF-IDF + Complement NB": make_pipeline(tfidf(), ComplementNB()),
        "TF-IDF + Regresión logística": make_pipeline(
            tfidf(), LogisticRegression(max_iter=2000, C=10, class_weight="balanced")),
        "TF-IDF + SVM lineal": make_pipeline(tfidf(), LinearSVC(C=1, class_weight="balanced")),
    }


def validar(textos, y, grupos):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    filas, preds = [], {}
    for nombre, m in modelos().items():
        p = cross_val_predict(m, textos, y, groups=grupos, cv=cv)
        preds[nombre] = p
        filas.append({"modelo": nombre, "accuracy": accuracy_score(y, p),
                      "f1_macro": f1_score(y, p, average="macro")})
    return pd.DataFrame(filas), preds


def main():
    df = cargar()
    co = df[df.origen == "Colombia"].reset_index(drop=True)
    intl = df[(df.origen == "Internacional") & df.profesion.isin(PROFESIONES_CON_LEY)].reset_index(drop=True)
    resultados = {}

    # E1 y E1m
    tablas = []
    for exp, textos in [("E1 (texto original)", co.texto), ("E1m (profesión enmascarada)", co.texto.map(enmascarar))]:
        tabla, preds = validar(textos, co.profesion, co.grupo)
        tabla.insert(0, "experimento", exp)
        tablas.append(tabla)
        if exp.startswith("E1m"):
            mejor = tabla.sort_values("f1_macro").iloc[-1]["modelo"]
            rep = classification_report(co.profesion, preds[mejor], output_dict=True, zero_division=0)
            pd.DataFrame(rep).T.round(3).to_csv("results/e1m_informe_por_clase.csv")
            fig, ax = plt.subplots(figsize=(8, 7))
            ConfusionMatrixDisplay.from_predictions(co.profesion, preds[mejor], ax=ax, xticks_rotation=60,
                                                    normalize="true", values_format=".2f", colorbar=False)
            ax.set_xlabel("Predicción"); ax.set_ylabel("Profesión real")
            ax.set_title(f"Matriz de confusión normalizada, E1m ({mejor})")
            plt.tight_layout(); plt.savefig("figures/fig3_confusion_e1m.png", dpi=150); plt.close()

    # E3: 9 profesiones, todos los documentos
    nueve = pd.concat([co, intl]).reset_index(drop=True)
    for exp, textos in [("E3 (texto original)", nueve.texto), ("E3m (profesión enmascarada)", nueve.texto.map(enmascarar))]:
        tabla, _ = validar(textos, nueve.profesion, nueve.grupo)
        tabla.insert(0, "experimento", exp)
        tablas.append(tabla)

    # E2: leyes colombianas -> códigos internacionales
    filas = []
    for enm in (False, True):
        f = enmascarar if enm else (lambda t: t)
        for nombre, m in modelos().items():
            m.fit(co.texto.map(f), co.profesion)
            p = m.predict(intl.texto.map(f))
            filas.append({"experimento": "E2m (enmascarado)" if enm else "E2 (texto original)", "modelo": nombre,
                          "accuracy": accuracy_score(intl.profesion, p),
                          "f1_macro": f1_score(intl.profesion, p, average="macro")})
    tablas.append(pd.DataFrame(filas))

    res = pd.concat(tablas).round(3)
    res.to_csv("results/resultados_e1_e2.csv", index=False)
    print(f"Leyes colombianas: {len(co)} párrafos, {co.grupo.nunique()} grupos | internacionales: {len(intl)} párrafos | E3: {len(nueve)}")

    # Figura 4: comparación F1-macro
    mejores = res[res.modelo == "TF-IDF + SVM lineal"].copy()
    mejores["base"] = mejores.experimento.str.extract(r"^(E\d)")
    mejores["enmascarado"] = mejores.experimento.str.contains("m ")
    piv = mejores.pivot(index="base", columns="enmascarado", values="f1_macro").rename(
        columns={False: "Texto original", True: "Profesión enmascarada"}).reindex(["E1", "E3", "E2"])
    ax = piv.plot.bar(figsize=(7, 4), color=["#2a6f97", "#e09f3e"], rot=0)
    ax.axhline(1 / 9, ls="--", c="grey", lw=1, label="Azar (1/9)")
    ax.set_ylabel("F1-macro"); ax.set_xlabel("")
    ax.set_xticklabels(["E1: leyes CO\n(CV agrupada)", "E3: todos los docs\n(CV agrupada)", "E2: CO → internacional"])
    ax.set_title("F1-macro de SVM lineal por experimento"); ax.legend()
    plt.tight_layout(); plt.savefig("figures/fig4_f1_experimentos.png", dpi=150); plt.close()
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
