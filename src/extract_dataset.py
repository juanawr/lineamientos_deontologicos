"""
Extracción del dataset de párrafos a partir del Anexo III (pestaña A III P1 CDTHS).

El Excel está maquetado para lectura humana: cada código ocupa un rango de filas
en una columna concreta. Este script recorre esos rangos (tomados de la tabla de
ubicación del propio Excel y del Anexo I), limpia cada celda y genera un CSV con
un párrafo por fila.

Uso:
    python src/extract_dataset.py --xlsx "data/raw/Anexo III.xlsx" --out data/processed/parrafos.csv
"""
import argparse
import re
import unicodedata

import openpyxl
import pandas as pd

SHEET = "A III P1 CDTHS"

# (doc_id, profesión, origen, columna (0=A), fila_inicio, fila_fin, descripción)
# Leyes colombianas: columna B (versión de la fuente 1: Juriscol / Senado).
# Códigos internacionales: columnas D, E y F.
DOCUMENTOS = [
    ("CO_MED", "Medicina", "Colombia", 1, 29, 1066, "Ley 23 de 1981"),
    ("CO_ODO", "Odontología", "Colombia", 1, 1067, 1693, "Ley 35 de 1989"),
    ("CO_FIS", "Fisioterapia", "Colombia", 1, 1694, 2120, "Ley 528 de 1999"),
    ("CO_OPT", "Optometría", "Colombia", 1, 2121, 2628, "Ley 650 de 2001"),
    ("CO_ENF", "Enfermería", "Colombia", 1, 2629, 3122, "Ley 911 de 2004"),
    ("CO_TO", "Terapia Ocupacional", "Colombia", 1, 3123, 3530, "Ley 949 de 2005"),
    ("CO_TR", "Terapia Respiratoria", "Colombia", 1, 3531, 3972, "Ley 1240 de 2008"),
    ("CO_BAC", "Bacteriología", "Colombia", 1, 3978, 4691, "Ley 841 de 2003"),
    ("CO_PSI", "Psicología", "Colombia", 1, 4692, 5414, "Ley 1090 de 2006"),
    ("INT_GINEBRA", "Medicina", "Internacional", 3, 235, 277, "Declaración de Ginebra, AMM"),
    ("INT_CIEM", "Medicina", "Internacional", 4, 5420, 5501, "Código Internacional de Ética Médica, AMM"),
    ("INT_OMC", "Medicina", "Internacional", 5, 5517, 6074, "Código de Deontología Médica, OMC España"),
    ("INT_DENT", "Odontología", "Internacional", 5, 6076, 7040, "Código Español de Ética y Deontología Dental"),
    ("INT_APTA", "Fisioterapia", "Internacional", 5, 7042, 7129, "Código de Ética APTA"),
    ("INT_CNOO", "Optometría", "Internacional", 5, 7130, 7497, "Código Deontológico CNOO"),
    ("INT_CIE", "Enfermería", "Internacional", 5, 7499, 7976, "Código de Ética del CIE"),
    ("INT_AOTA", "Terapia Ocupacional", "Internacional", 5, 7982, 8187, "Código de Ética AOTA"),
    ("INT_AARC", "Terapia Respiratoria", "Internacional", 5, 8189, 8218, "Declaración de Ética AARC"),
    ("INT_CMQC", "Bacteriología", "Internacional", 5, 8220, 8461, "Código de Ética CMQC Costa Rica"),
    ("INT_APA", "Psicología", "Internacional", 5, 8463, 8788, "Principios Éticos APA"),
    ("INT_ASHA", "Fonoaudiología", "Internacional", 5, 8790, 8895, "Código de Ética ASHA"),
    ("INT_ANIC", "Instrumentación Quirúrgica", "Internacional", 5, 8898, 8941, "Ética ANIC"),
    ("INT_CFN", "Nutrición y Dietética", "Internacional", 5, 8942, 9393, "Resolución CFN 599 de 2018"),
    ("INT_NAPG", "Gerontología", "Internacional", 5, 9394, 9414, "Código de Ética NAPG"),
    ("INT_PCNZ", "Química Farmacéutica", "Internacional", 5, 9417, 9567, "Código de Ética Pharmacy Council NZ"),
    ("INT_SASW", "Trabajo Social", "Internacional", 5, 9570, 9714, "Código de Ética SASW"),
]

# Líneas de metadatos que no forman parte del contenido normativo.
PATRONES_DESCARTE = [
    r"^disponible el", r"^diario oficial", r"^[íi]ndice", r"^arts?\.?:", r"^[úu]ltima actualizaci[óo]n",
    r"^consultado en", r"^juriscol", r"^supersalud", r"^senado de la rep[úu]blica", r"^<", r"^\[",
    r"disponible el \d", r"^[A-ZÁÉÍÓÚ][\wáéíóúñ ]{3,40}: [A-ZÁÉÍÓÚ\-–\. ]{8,}",
    r"^el (presidente|ministro|secretario)", r"^la (presidenta|ministra|secretaria)", r"^contacto:",
]
RE_DESCARTE = re.compile("|".join(PATRONES_DESCARTE), re.IGNORECASE)
RE_ARTICULO = re.compile(r"^(ART[ÍI]CULO|Art[íi]culo|ART\.|Art\.)\s*\d+", re.UNICODE)
MIN_PALABRAS = 5


def limpiar(texto: str) -> str:
    texto = unicodedata.normalize("NFC", str(texto))
    texto = texto.replace("\xa0", " ").replace("‐", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", texto).strip()


def es_encabezado(texto: str) -> bool:
    """Títulos, capítulos y firmas: pocas palabras o todo en mayúsculas."""
    letras = [c for c in texto if c.isalpha()]
    mayus = sum(c.isupper() for c in letras) / max(len(letras), 1)
    return len(texto.split()) < MIN_PALABRAS or mayus > 0.8


def extraer(xlsx: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb[SHEET]
    max_fila = max(d[5] for d in DOCUMENTOS)
    filas = list(ws.iter_rows(min_row=1, max_row=max_fila, max_col=6, values_only=True))

    registros = []
    for doc_id, profesion, origen, col, ini, fin, desc in DOCUMENTOS:
        bloque = 0  # identificador de artículo o sección, para la validación agrupada
        for n in range(ini, fin + 1):
            valor = filas[n - 1][col]
            if valor is None:
                continue
            texto = limpiar(valor)
            if not texto or RE_DESCARTE.search(texto):
                continue
            if RE_ARTICULO.match(texto) or es_encabezado(texto):
                bloque += 1  # un artículo o un título abre un bloque nuevo
            if es_encabezado(texto):
                continue
            registros.append({
                "doc_id": doc_id, "profesion": profesion, "origen": origen, "documento": desc,
                "fila_excel": n, "grupo": f"{doc_id}_{bloque:04d}", "texto": texto,
            })

    df = pd.DataFrame(registros)
    # Duplicados exactos dentro de un mismo documento (párrafos repetidos por la maquetación).
    df = df.drop_duplicates(subset=["doc_id", "texto"]).reset_index(drop=True)
    df["n_palabras"] = df["texto"].str.split().str.len()
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", default="data/processed/parrafos.csv")
    a = ap.parse_args()
    df = extraer(a.xlsx)
    df.to_csv(a.out, index=False, encoding="utf-8")
    print(f"{len(df)} párrafos, {df.n_palabras.sum()} palabras, {df.doc_id.nunique()} documentos -> {a.out}")
