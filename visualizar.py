"""
Dibuja la red cerebral (nodos + aristas) sobre una imagen del scalp con barra
lateral de energía. Cada nodo se colorea por su módulo y se dimensiona por
su betweenness centrality.

Equivalente MATLAB:
- LinearSegmentedColormap       -> makeColorMap.m
- obtener_posiciones_electrodos -> DefPosElectrodos.ms
- representacion_red            -> representaacion_red.m
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


# Coordenadas cartesianas de los 18 electrodos del montaje diferencial,
# distribuidos en 3 radios (anillo externo, medio e interno) y orientados
# según ángulos polares predefinidos.
def obtener_posiciones_electrodos():
    r_ext = 1.0       
    r_int = 0.5
    r_ext_dif = (r_ext + r_int) / 2
    r_int_dif = r_int / 2

    # Orden: [Fp1F3, F3C3, C3P3, P3O1, Fp1F7, F7T3, T3T5, T5O1, Fp2F4, F4C4,
    #         C4P4, P4O2, Fp2F8, F8T4, T4T6, T6O2, FzCz, CzPz]
    angulos_grados = [
        118.0, 153.0, 207.5, 242.5,
        125.0, 160.0, 200.0, 235.0,
        62.5,  27.5,  335.0, 297.5,  
        55.0,  20.0,  340.0, 305.0,  
        90.0,  270.0
    ]
    radios = [
        r_ext_dif, r_int_dif, r_int_dif, r_ext_dif,
        r_ext,     r_ext,     r_ext,     r_ext,
        r_ext_dif, r_int_dif, r_int_dif, r_ext_dif,
        r_ext,     r_ext,     r_ext,     r_ext,
        r_int_dif, r_int_dif
    ]

    angulos_rad = np.deg2rad(angulos_grados)
    x_polar = np.array(radios) * np.cos(angulos_rad)
    y_polar = np.array(radios) * np.sin(angulos_rad)

    # Y se invierte porque las imágenes tienen el eje Y descendente.
    x_cart = np.round(x_polar * (800 / 3.2)) + 400
    y_cart = 400 - np.round(y_polar * (800 / 3.2))

    return x_cart, y_cart


# Dibuja la red sobre scalp.png con la barra lateral de energía.
def representacion_red(medias_l, modules, bc, color1, color2, energia, ruta_imagen_scalp='scalp.png', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    x_dif, y_dif = obtener_posiciones_electrodos()
    ncanales = medias_l.shape[0]

    # IMAGEN DE FONDO
    # PIL detecta el formato por los magic bytes, no por la extensión: útil si el archivo se 
    # renombra de .jpg a .png o viceversa.
    try:
        img = np.asarray(Image.open(ruta_imagen_scalp))
        ax.imshow(img)
    except FileNotFoundError:
        print(f"Advertencia: No se encontró la imagen '{ruta_imagen_scalp}'. Dibujando sin fondo.")
        ax.set_facecolor('white')
        ax.set_xlim(0, 1000)
        ax.set_ylim(800, 0)

    # BARRA DE ENERGÍA
    energia = float(np.ravel(np.asarray(energia))[0])

    # Mínimo del 1% para que la barra sea siempre visible en pantalla.
    if energia * 100 < 1:
        energia = 0.01

    l_barra = int(round(5.5 * round(energia * 100)))

    # Fondo blanco 
    ax.fill_betweenx([675 - l_barra, 675], 801, 850, color='white', zorder=1)

    # LinearSegmentedColormap sustituye la interpolación vectorial manual de makeColorMap.m sin perder
    # calidad de degradado.
    cmap_energia = LinearSegmentedColormap.from_list("custom_cmap", [color1, color2])
    gradiente = np.linspace(1, 0, 256).reshape(256, 1)
    ax.imshow(gradiente, aspect='auto', cmap=cmap_energia,
              extent=[801, 850, 675, 675 - l_barra], zorder=2)

    # ARISTAS
    max_corr = np.max(medias_l)
    umbral = max_corr / 2.0

    # Se pintan todas las aristas. Las débiles quedan casi invisibles (mejora respecto a MATLAB). 
    # Además, se ordenan las aristas de menor a mayor coherencia para que las más fuertes (oscuras) 
    # se pinten las últimas y queden en la capa superior (igual que en MATLAB).
    aristas = [(medias_l[i, j], i, j)
               for i in range(ncanales)
               for j in range(i + 1, ncanales)]
    aristas.sort(key=lambda t: t[0])

    for corr, i, j in aristas:
        linewidth = (corr**2) * 10
        gris = float(np.clip(1.0 - (corr**2), 0, 1))
        color_linea = (gris, gris, gris)

        ax.plot([x_dif[i], x_dif[j]], [y_dif[i], y_dif[j]],
                color=color_linea, linewidth=linewidth, zorder=3)

    # NODOS
    n_modules = int(np.max(modules))
    cmap_nodos = plt.get_cmap('hsv', n_modules)

    # Si una arista supera el umbral max/2, se pintan los 18 electrodos (igual que en MATLAB)
    if max_corr > umbral:
        for i in range(ncanales):
            # Tamaño nodo proporcional a betweenness centrality (con offset 6 para que sea visible).
            tamano_marcador = (bc[i] * 49) + 6
            # modules[i]-1 -> para pasar de la convención 1-based de BCT al 0-based que espera cmap_nodos.
            color_nodo = cmap_nodos(int(modules[i]) - 1)

            ax.plot(x_dif[i], y_dif[i], marker='o', markersize=tamano_marcador,
                    markerfacecolor=color_nodo, markeredgecolor='black',
                    linestyle='None', zorder=4)

    ax.axis('off')
