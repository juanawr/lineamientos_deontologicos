# Clasificación automática de códigos deontológicos de profesiones de la salud

Proyecto final de Aprendizaje Automático (Grado en Ingeniería Matemática, UFV, curso 2025-2026).

Se clasifican párrafos de códigos deontológicos según la profesión a la que pertenecen, a partir del
Anexo III de los *Lineamientos para la elaboración y actualización de códigos deontológicos de las
profesiones del Talento Humano en Salud* (Ministerio de Salud y Protección Social de Colombia, 2022).

## Estructura

```
src/extract_dataset.py   Extrae y limpia los párrafos del Excel (Anexo III, pestaña A III P1 CDTHS)
src/comun.py             Carga del dataset y enmascarado de nombres de profesión
src/eda.py               Análisis exploratorio: tamaños, longitudes, vocabulario, temas NMF
src/experimentos.py      Experimentos E1, E1m, E2, E3 (TF-IDF + NB / regresión logística / SVM)
data/processed/          Dataset limpio (parrafos.csv)
figures/                 Figuras generadas
results/                 Tablas y salidas de los experimentos
```

## Reproducción

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Copiar el Excel del Anexo III a data/raw/ (no se versiona por su tamaño, 13 MB)
python src/extract_dataset.py --xlsx "data/raw/Lineamientos Codigos Deontologicos Parte 4 de 4 Anexo III.xlsx"
python src/eda.py
python src/experimentos.py
```

Todos los procesos aleatorios usan la semilla 42. Los resultados de la Entrega 2 están en
`results/resultados_e1_e2.csv`.

## Dataset (`data/processed/parrafos.csv`)

| Columna | Descripción |
|---|---|
| doc_id | Identificador del documento (p. ej. `CO_MED`, `INT_APA`) |
| profesion | Etiqueta: profesión del código |
| origen | `Colombia` (ley nacional) o `Internacional` |
| documento | Nombre del código o ley |
| fila_excel | Fila de origen en el Excel, para trazabilidad |
| grupo | Artículo o sección al que pertenece el párrafo (se usa en la validación agrupada) |
| texto | Párrafo limpio |
| n_palabras | Número de palabras |

## Fuente

Ministerio de Salud y Protección Social de Colombia (2022). Lineamientos para la elaboración y
actualización de códigos deontológicos de las profesiones del Talento Humano en Salud. Anexo III.
Las leyes colombianas son normas públicas (Juriscol, Secretaría del Senado). Los códigos
internacionales se citan con su fuente original dentro del propio Excel.
