"""
Genera las 3 figuras por paciente que estudio_coherencia.py guarda como
figura1/2/3.png:
- generar_informe_visual: red sobre el scalp para 5 bandas + Pearson.
- generar_evolucion_pearson: evolución por ventana de 4 medidas Pearson.
- generar_evolucion_coherencia: evolución por ventana x banda de 6 medidas.

Equivalente MATLAB: Creacion_y_Medidas_Red.m (líneas 421-639).

DIFERENCIAS vs MATLAB:
- MATLAB las pinta dentro de Creacion_y_Medidas_Red.m, aquí se separan en
  tres funciones.
"""

import numpy as np
import matplotlib.pyplot as plt
from visualizar import representacion_red
from funciones_red import network_measures


# Lista declarativa de las 5 bandas.
BANDAS = [
    ('delta', (0, 0), (0.250, 0.250, 0.250), 'Delta (0 - 4 Hz)'),
    ('theta', (0, 1), (0.466, 0.674, 0.188), 'Theta (4 - 7 Hz)'),
    ('alpha', (0, 2), (0.000, 0.447, 0.741), 'Alpha (7 - 13 Hz)'),
    ('beta',  (1, 0), (0.850, 0.325, 0.098), 'Beta (13 - 30 Hz)'),
    ('gamma', (1, 1), (0.750, 0.000, 0.750), 'Gamma (>30 Hz)'),
]


# Dibuja la red de una banda. network_measures se vuelve a llamar aquí para obtener
# específicamente modules y bc de la matriz MEDIA.
def _pintar_banda(ax, matriz, color_arista, color_fondo, energia, titulo, color_titulo):
    res = network_measures(matriz, 1)
    modules, bc = res[5], res[7]
    representacion_red(matriz, modules, bc, color_arista, color_fondo, energia, ax=ax)
    # Pearson no muestra % de energía. Réplica fiel.
    ax.set_title(titulo, fontsize=14, color=color_titulo, fontweight='bold')
    if titulo != 'C. Pearson':
        ax.text(825, 682, f'{int(round(energia * 100))}%',
                ha='center', va='top', fontsize=12, color=color_titulo,
                fontweight='bold', zorder=5)


# Función que genera la figura 1: 6 subplots con la red dibujada sobre el scalp y la 
# barra lateral de energía por banda.
def generar_informe_visual(archivo, corr_media_l, coher_medias, band_energy_medias):
    band_energy_medias = np.ravel(np.asarray(band_energy_medias))

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.canvas.manager.set_window_title(f'Coherencia - {archivo}')

    for i, (nombre, (r, c), color, titulo) in enumerate(BANDAS):
        _pintar_banda(axs[r, c], coher_medias[nombre], color, (0, 0, 0),
                      band_energy_medias[i], f'Banda {titulo}', color)

    # Pearson: barra blanca sobre blanco. energía=0.0055 para que la barra ocupe espacio
    # pero quede invisible, igual que en MATLAB.
    _pintar_banda(axs[1, 2], corr_media_l, (1, 1, 1), (1, 1, 1),
                  0.0055, 'C. Pearson', 'black')

    plt.tight_layout()
    return fig


# Métricas Pearson a representar en la figura de evolución.
PEARSON_PLOTS = [('APL1', 0), ('DOL', 1), ('Global Eff', 2), ('ACC', 3)]


# Figura 2: 2×2 subplots con la evolución temporal (por ventana) de 4 medidas
# Pearson; cada uno anota su std en rojo en coordenadas relativas al eje.
def generar_evolucion_pearson(archivo, ev_pearson):
    ventanas = ev_pearson.shape[1]
    x = np.arange(1, ventanas + 1)

    fig, axs = plt.subplots(2, 2, figsize=(15, 7.5))
    fig.canvas.manager.set_window_title(f'C Pearson - Evolucion red - {archivo}')

    for ax, (titulo, idx) in zip(axs.flat, PEARSON_PLOTS):
        serie = ev_pearson[idx, :]
        ax.plot(x, serie)
        ax.set_title(titulo, fontweight='bold')
        ax.set_xlabel('ventanas')
        std_val = float(np.std(serie))
        # transAxes mantiene el texto dentro del eje (ylim estrecho podría tirar el texto 
        # fuera de pantalla).
        ax.text(0.7, 0.85, f'std = {std_val:.5f}', color='red', fontsize=14, transform=ax.transAxes)

    plt.tight_layout()
    return fig


# Métricas a representar (título, índice en ev_coher, mostrar leyenda). 
COHER_PLOTS = [('APL', 0, False), ('DoL', 1, False), ('Eglob', 2, True), ('ACC', 3, False),
               ('Modularity', 5, False), ('Sw', 6, False)]


# Figura 3: 3×2 subplots con la evolución temporal de 6 medidas, una curva
# por banda (5 curvas) en cada subplot. Solo Eglob lleva leyenda para no
# saturar el resto de subplots.
def generar_evolucion_coherencia(archivo, ev_coher):
    ventanas = ev_coher.shape[2]
    x = np.arange(1, ventanas + 1)

    fig, axs = plt.subplots(3, 2, figsize=(15, 7.5))
    fig.canvas.manager.set_window_title(f'CBandas - Evolucion red - {archivo}')

    etiquetas = [f'Banda {b[0]}' for b in BANDAS]

    for ax, (titulo, idx, with_legend) in zip(axs.flat, COHER_PLOTS):
        for bi, (_, _, color, _) in enumerate(BANDAS):
            ax.plot(x, ev_coher[idx, bi, :], linewidth=2, color=color)
        ax.set_title(titulo, fontsize=18, fontweight='bold')
        ax.set_xlabel('ventanas', fontsize=14)
        if with_legend:
            ax.legend(etiquetas, loc='center left',
                      bbox_to_anchor=(1.02, 0.5), fontsize=10)

    plt.tight_layout()
    return fig
