"""
Compone las figuras del informe (máx. 5 páginas) uniendo dos paneles por figura.
Requiere haber ejecutado antes eda.py y experimentos.py.

Uso: python src/figuras_informe.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PANELES = {
    "figures/informe_fig1_eda.png": ["figures/fig1_parrafos_por_profesion.png", "figures/fig2_longitud_parrafos.png"],
    "figures/informe_fig2_resultados.png": ["figures/fig3_confusion_e1m.png", "figures/fig4_f1_experimentos.png"],
}

for salida, rutas in PANELES.items():
    imgs = [mpimg.imread(r) for r in rutas]
    ratios = [im.shape[1] / im.shape[0] for im in imgs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 12 / sum(ratios)), gridspec_kw={"width_ratios": ratios})
    for ax, im, letra in zip(axes, imgs, "ab"):
        ax.imshow(im); ax.axis("off")
        ax.text(0, 0, f"({letra})", fontsize=12, fontweight="bold", va="bottom")
    plt.subplots_adjust(left=0, right=1, top=0.97, bottom=0, wspace=0.02)
    plt.savefig(salida, dpi=200); plt.close()
    print("->", salida)
