"""
Calcula 9 medidas de red sobre una matriz de conectividad (Pearson o
coherencia por banda): longitud media de camino, density of links, eficiencia
global, ACC, transitividad, módulos, modularidad, betweenness centrality y
smallworldness. Apoyado en BCT (Brain Connectivity Toolbox) vía bctpy.

Equivalente MATLAB: network_measures.m

DIFERENCIAS vs MATLAB:
- Density of links: MATLAB usa density_und que binariza primero, lo que con
  matrices densas (Pearson/coherencia) devolvía siempre 1. Aquí se usa la
  fórmula directa sum(W) / (N*(N-1)).
- Smallworldness: MATLAB usa la fórmula analítica Watts-Strogatz. Esto puede producir 
  Sw NEGATIVO o COMPLEJO cuando <k> < 2 (matrices de coherencia débil). Aquí se simulan
  N_RANDOM_GRAPHS=20 redes aleatorias reales y se promedian sus métricas.
"""

import numpy as np
import bct

# Número de redes aleatorias para estimar la referencia Watts-Strogatz de Sw.
N_RANDOM_GRAPHS = 20

# Convierte matriz de pesos en matriz de distancias (mayor peso -> menor distancia). 
# Pone inf donde el peso es 0 (convención BCT). Equivalente en MATLAB.
def _inv_safe(mat, alpha=1.0):
    mat = np.asarray(mat, dtype=float)
    inv = np.full_like(mat, np.inf)
    mask = mat > 0
    inv[mask] = 1.0 / (mat[mask] ** alpha)
    np.fill_diagonal(inv, 0)
    return inv


# Devuelve 9 medidas: (aplt, dol, eglob, acc, transitivity, modules, modularity,
# bc, smallworldness).
def network_measures(base_matrix, alpha):
    weights = np.asarray(base_matrix, dtype=float)
    ncanales = weights.shape[0]

    # INTEGRACIÓN
    lengths = _inv_safe(weights, alpha)
    # Fórmula directa: media de peso por par ordenado. Evita la binarización
    # implícita de bct.density_und cuando la matriz es densa.
    dol = np.sum(weights) / (ncanales * (ncanales - 1))

    distances_d, _ = bct.distance_wei(lengths)
    charpath_res = bct.charpath(distances_d, include_diagonal=False, include_infinite=False)
    aplt = charpath_res[0]
    eglob = charpath_res[1]

    # SEGREGACIÓN
    cc = bct.clustering_coef_wu(weights)
    acc = np.mean(cc)
    t = bct.transitivity_wu(weights)
    modules, modularity = bct.modularity_und(weights)
    modules = np.asarray(modules)

    # CENTRALIDAD
    ebc, bc = bct.edge_betweenness_wei(lengths)
    # Normaliza por el número de pares posibles excluyendo el propio nodo.
    bc = bc / ((ncanales - 1) * (ncanales - 2))

    # SMALLWORLDNESS
    apl_rand_acc = 0.0
    acc_rand_acc = 0.0
    n_validas = 0
    for i in range(N_RANDOM_GRAPHS):
        rng_i = np.random.RandomState(42 + i)
        weights_rand = rng_i.rand(ncanales, ncanales)
        lengths_rand = _inv_safe(weights_rand, alpha)
        dist_rand, _ = bct.distance_wei(lengths_rand)
        apl_i = bct.charpath(dist_rand)[0]
        cc_i = bct.clustering_coef_wu(weights_rand)
        acc_i = float(np.mean(cc_i))
        if apl_i > 0 and acc_i > 0:
            apl_rand_acc += apl_i
            acc_rand_acc += acc_i
            n_validas += 1
    if n_validas > 0:
        apl_rand = apl_rand_acc / n_validas
        acc_rand = acc_rand_acc / n_validas
    else:
        apl_rand = 0.0
        acc_rand = 0.0

    if apl_rand == 0 or acc_rand == 0 or aplt == 0:
        sw = np.nan
    else:
        sw = (acc / acc_rand) / (aplt / apl_rand)

    return aplt, dol, eglob, acc, t, modules, modularity, bc, sw
