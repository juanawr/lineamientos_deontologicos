"""Utilidades compartidas: carga del dataset y enmascarado de nombres de profesión."""
import re
import pandas as pd

PROFESIONES_CON_LEY = [
    "Medicina", "Odontología", "Fisioterapia", "Optometría", "Enfermería",
    "Terapia Ocupacional", "Terapia Respiratoria", "Bacteriología", "Psicología",
]

# Raíces que delatan la profesión (p. ej. "odontólogo", "enfermería", "fisioterapeuta").
# Se sustituyen por un token neutro para comprobar si el modelo aprende algo más que el nombre.
RAICES_PROFESION = [
    r"m[ée]dic\w*", r"medicina\w*", r"odont\w*", r"dental\w*", r"dentist\w*", r"fisioterap\w*",
    r"terapia f[íi]sica", r"opt[óo]metr\w*", r"optometr\w*", r"[óo]ptic\w*", r"enfermer\w*",
    r"terap\w* ocupacional\w*", r"terap\w* respiratori\w*", r"cuidados respiratorios",
    r"bacteri[óo]log\w*", r"microbi[óo]log\w*", r"qu[íi]mic\w* cl[íi]nic\w*", r"psic[óo]log\w*",
    r"fonoaudi[óo]log\w*", r"nutricionist\w*", r"farmac[ée]utic\w*", r"trabajador\w* social\w*",
    r"geront[óo]log\w*", r"instrumentador\w*", r"terapeut\w*",
]
RE_PROFESION = re.compile(r"\b(" + "|".join(RAICES_PROFESION) + r")\b", re.IGNORECASE)


def enmascarar(texto: str) -> str:
    return RE_PROFESION.sub(" PROFESION ", texto)


def cargar(ruta="data/processed/parrafos.csv") -> pd.DataFrame:
    return pd.read_csv(ruta)
