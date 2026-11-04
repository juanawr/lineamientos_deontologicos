"""
Análisis exploratorio del dataset de párrafos.
Genera tablas en results/ y figuras en figures/.

Uso: python src/eda.py
"""
from collections import Counter
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from comun import cargar, enmascarar, RE_PROFESION, PROFESIONES_CON_LEY

STOP_ES = set("""de la que el en y a los del se las por un para con no una su al lo como más o pero sus le ha me
si sin sobre este ya entre cuando todo esta ser son dos también fue había era muy años hasta desde está mi porque qué
sólo han yo hay vez puede todos así nos ni parte tiene él uno donde bien tiempo mismo ese ahora cada e vida otro
después te otros aunque esa eso hace otra gobierno tan durante siempre día tanto ella tres sí dijo sido gran país
según menos año antes estado contra sino forma caso nada hacer general estaba poco estos presente acuerdo esto
mientras dentro debe deberá deben cual cuales cualquier dicho dicha dichos tal toda todas sea sean será serán
ante bajo artículo art ley parágrafo literal numeral profesion es""".split())


def main():
    df = cargar()
    df["n_caracteres"] = df["texto"].str.len()

    # 1. Tamaño por documento y por profesión
    resumen = (df.groupby(["origen", "profesion", "doc_id"])
                 .agg(parrafos=("texto", "size"), palabras=("n_palabras", "sum"),
                      bloques=("grupo", "nunique"), mediana_palabras=("n_palabras", "median"))
                 .reset_index())
    resumen.to_csv("results/eda_por_documento.csv", index=False)

    tab = df.pivot_table(index="profesion", columns="origen", values="texto", aggfunc="size", fill_value=0)
    tab["Total"] = tab.sum(axis=1)
    tab = tab.sort_values("Total")
    tab.to_csv("results/eda_por_profesion.csv")
    ax = tab[["Colombia", "Internacional"]].plot.barh(stacked=True, figsize=(7, 5), color=["#2a6f97", "#a9c5d3"])
    ax.set_xlabel("Número de párrafos"); ax.set_ylabel("")
    ax.set_title("Párrafos por profesión y origen del código")
    plt.tight_layout(); plt.savefig("figures/fig1_parrafos_por_profesion.png", dpi=150); plt.close()

    # 2. Distribución de longitudes
    fig, ax = plt.subplots(figsize=(7, 4))
    for origen, sub in df.groupby("origen"):
        ax.hist(sub["n_palabras"].clip(upper=200), bins=40, alpha=0.6, label=origen)
    ax.set_xlabel("Palabras por párrafo (recortado a 200)"); ax.set_ylabel("Frecuencia"); ax.legend()
    ax.set_title("Longitud de los párrafos")
    plt.tight_layout(); plt.savefig("figures/fig2_longitud_parrafos.png", dpi=150); plt.close()

    # 3. Vocabulario: términos más frecuentes y peso de los nombres de profesión
    tokens = Counter()
    for t in df["texto"].str.lower():
        tokens.update(w for w in re.findall(r"[a-záéíóúñü]{3,}", t) if w not in STOP_ES)
    pd.DataFrame(tokens.most_common(40), columns=["termino", "frecuencia"]).to_csv(
        "results/eda_terminos_frecuentes.csv", index=False)
    df["menciona_profesion"] = df["texto"].apply(lambda t: bool(RE_PROFESION.search(t)))
    fuga = df.groupby(["origen", "profesion"])["menciona_profesion"].mean().round(3)
    fuga.to_csv("results/eda_menciona_profesion.csv")

    # 4. Exploración no supervisada: temas con NMF sobre los textos enmascarados
    vec = TfidfVectorizer(max_df=0.5, min_df=5, stop_words=list(STOP_ES), ngram_range=(1, 2))
    X = vec.fit_transform(df["texto"].map(enmascarar).str.lower())
    nmf = NMF(n_components=8, random_state=0, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(X)
    vocab = np.array(vec.get_feature_names_out())
    temas = [", ".join(vocab[np.argsort(c)[::-1][:8]]) for c in nmf.components_]
    pd.DataFrame({"tema": range(1, 9), "terminos": temas}).to_csv("results/eda_temas_nmf.csv", index=False)
    df["tema"] = W.argmax(axis=1) + 1
    mezcla = pd.crosstab(df["profesion"], df["tema"], normalize="index").round(2)
    mezcla.to_csv("results/eda_temas_por_profesion.csv")

    # Resumen en pantalla
    print(f"Párrafos: {len(df)} | palabras: {df.n_palabras.sum()} | documentos: {df.doc_id.nunique()}")
    print(f"Mediana de palabras por párrafo: {df.n_palabras.median():.0f} (IQR {df.n_palabras.quantile(.25):.0f}-{df.n_palabras.quantile(.75):.0f})")
    print(f"Vocabulario NMF: {X.shape[1]} términos")
    print(tab)
    print("Proporción de párrafos que nombran una profesión:\n", fuga)
    for i, t in enumerate(temas, 1):
        print(f"Tema {i}: {t}")


if __name__ == "__main__":
    main()
