"""
Ventanea la señal multicanal, aplica el montaje diferencial según equipo,
calcula matrices de Pearson y de coherencia por banda para cada ventana, y
agrega las medidas de red (network_measures) sobre la matriz media y sobre
la evolución por ventana.

Equivalente MATLAB: Creacion_y_Medidas_Red.m

DIFERENCIAS vs MATLAB:
- Los montajes están centralizados en la tabla MONTAJES. MATLAB tiene una
  cadena de if/elseif sobre (ncanales, caja).
"""

import numpy as np
from funciones_aux import corr_matrix, band_coherence
from funciones_red import network_measures

# DIFERENCIA vs MATLAB: se elimina montaje referencial. Se mantiene solo el diferencial,
# que es el que se usó en el estudio de Mario Refoyo y es el recomendado para scalp.
MONTAJES = {
    # UCI1 — 16 canales scalp + ECG
    (22, None): {
        'diferencial': [
            (8, 3), (3, 0), (0, 13), (13, 11),   # Fp1-F3, F3-C3, C3-P3, P3-O1
            (8, 5), (5, 16), (16, 18), (18, 11), # Fp1-F7, F7-T3, T3-T5, T5-O1
            (9, 4), (4, 1), (1, 14), (14, 12),   # Fp2-F4, F4-C4, C4-P4, P4-O2
            (9, 6), (6, 17), (17, 19), (19, 12), # Fp2-F8, F8-T4, T4-T6, T6-O2
            (7, 2), (2, 15),                     # Fz-Cz, Cz-Pz
        ],
    },
    # UCI2 (sin Fpz)
    (21, 1138): {
        'diferencial': [
            (8, 3), (3, 0), (0, 12), (12, 10),   # Fp1-F3, F3-C3, C3-P3, P3-O1
            (8, 5), (5, 15), (15, 17), (17, 10), # Fp1-F7, F7-T3, T3-T5, T5-O1
            (9, 4), (4, 1), (1, 13), (13, 11),   # Fp2-F4, F4-C4, C4-P4, P4-O2
            (9, 6), (6, 16), (16, 18), (18, 11), # Fp2-F8, F8-T4, T4-T6, T6-O2
            (7, 2), (2, 14),                     # Fz-Cz, Cz-Pz
        ],
    },
    # NRL
    (21, 65535): {
        'diferencial': [
            (0, 1), (1, 2), (2, 3), (3, 4),         # Fp1-F3, F3-C3, C3-P3, P3-O1
            (0, 5), (5, 6), (6, 7), (7, 4),         # Fp1-F7, F7-T3, T3-T5, T5-O1
            (11, 12), (12, 13), (13, 14), (14, 15), # Fp2-F4, F4-C4, C4-P4, P4-O2
            (11, 16), (16, 17), (17, 18), (18, 15), # Fp2-F8, F8-T4, T4-T6, T6-O2
            (8, 9), (9, 10),                         # Fz-Cz, Cz-Pz
        ],
    },
    # Maudsley
    (27, None): {
        'diferencial': [
            (0, 1), (1, 2), (2, 3), (3, 4),          # Fp1-F3, F3-C3, C3-P3, P3-O1
            (0, 5), (5, 8), (8, 9), (9, 4),          # Fp1-F7, F7-T3, T3-T5, T5-O1
            (0, 6), (6, 7), (7, 10), (10, 4),        
            (14, 15), (15, 16), (16, 17), (17, 18),  # Fp2-F4, F4-C4, C4-P4, P4-O2
            (14, 19), (19, 22), (22, 23), (23, 18),  # Fp2-F8, F8-T4, T4-T6, T6-O2
            (14, 20), (20, 21), (21, 24), (24, 18),  
            (11, 12), (12, 13),                       # Fz-Cz, Cz-Pz
        ],
    },
}


def _aplicar_montaje(datos, ncanales, caja):
    cfg = MONTAJES.get((ncanales, caja)) or MONTAJES.get((ncanales, None))

    if cfg is None:
        raise ValueError(f"Montaje no soportado: ncanales={ncanales}, caja={caja}")
    return [datos[i1] - datos[i2] for (i1, i2) in cfg['diferencial']]


BANDAS_COHER = ('delta', 'theta', 'alpha', 'beta', 'gamma')


# Segmenta la señal multicanal en ventanas de longitud M con la fracción de
# solapamiento indicada. Devuelve una lista por canal, cada elemento un array
# (M, ventanas).
def _ventanear(matriz_datos_filtrado, M, superposicion):
    matriz_datos_filtrado_traspuesta = matriz_datos_filtrado.T
    ncanales, length_data = matriz_datos_filtrado_traspuesta.shape

    if M > length_data:
        M = length_data

    fraccion_paso = 1 - superposicion
    ventanas = int(np.floor((length_data - M) / (fraccion_paso * M) + 1))

    datos = []
    for i in range(ncanales):
        ventanas_canal = np.zeros((M, ventanas))
        ventanas_canal[:, 0] = matriz_datos_filtrado_traspuesta[i, 0:M]
        ind = M
        for v in range(1, ventanas):
            lim1 = int(ind - np.ceil(superposicion * M))
            lim2 = lim1 + M
            ventanas_canal[:, v] = matriz_datos_filtrado_traspuesta[i, lim1:lim2]
            ind = lim2

        datos.append(ventanas_canal)
    return datos, ventanas


def _network_measures_compacto(matriz):
    apl1, dol, eglob, acc, t, _, modularity, _, sw = network_measures(matriz, 1)
    return [apl1, dol, eglob, acc, t, modularity, sw]


def _medidas_por_ventana(datos_l, fs):
    ncanales_l = len(datos_l)
    ventanas = datos_l[0].shape[1]

    corr_matrix_l = np.zeros((ncanales_l, ncanales_l, ventanas))
    lag_matrix_l = np.zeros((ncanales_l, ncanales_l, ventanas))
    coher = {b: np.zeros((ncanales_l, ncanales_l, ventanas)) for b in BANDAS_COHER}
    band_energy_tensor = np.zeros((5, ncanales_l, ventanas))

    ev_pearson = np.zeros((7, ventanas))
    ev_coher = np.zeros((7, 5, ventanas))

    for i in range(ventanas):
        corr_matrix_l[:, :, i], lag_matrix_l[:, :, i] = corr_matrix(datos_l, i)

        d, t, a, b, g, energy = band_coherence(datos_l, fs, i)
        coher['delta'][:, :, i] = d
        coher['theta'][:, :, i] = t
        coher['alpha'][:, :, i] = a
        coher['beta'][:, :, i] = b
        coher['gamma'][:, :, i] = g
        band_energy_tensor[:, :, i] = energy

        ev_pearson[:, i] = _network_measures_compacto(corr_matrix_l[:, :, i])
        for bi, banda in enumerate(BANDAS_COHER):
            ev_coher[:, bi, i] = _network_measures_compacto(coher[banda][:, :, i])

    return {
        'corr_matrix_l': corr_matrix_l,
        'lag_matrix_l': lag_matrix_l,
        'coher': coher,
        'band_energy_tensor': band_energy_tensor,
        'ev_pearson': ev_pearson,
        'ev_coher': ev_coher,
    }


# Devuelve un vector de 14 elementos con media y std de cada métrica.
def _stats_evolucion(serie_7xN):
    out = np.empty(14)
    for k in range(6):
        out[2 * k]     = np.mean(serie_7xN[k, :])
        out[2 * k + 1] = np.std(serie_7xN[k, :])
    out[12] = np.nanmean(serie_7xN[6, :])
    out[13] = np.nanstd(serie_7xN[6, :])
    return out


# Devuelve las matrices medias de Pearson y coherencia por banda para alimentar 
# las figuras del reporte visual.
def _agregar_resultados(med):
    corr_media_l = np.mean(med['corr_matrix_l'], axis=2)
    coher_medias = {b: np.mean(c, axis=2) for b, c in med['coher'].items()}

    band_energy_medias = np.mean(med['band_energy_tensor'], axis=(1, 2))
    total_energy = np.sum(band_energy_medias)

    if total_energy > 0:
        band_energy_medias = band_energy_medias / total_energy

    m_aux_all_data = np.array(_network_measures_compacto(corr_media_l)).reshape(-1, 1)
    m_aux_all_data_ev = _stats_evolucion(med['ev_pearson']).reshape(-1, 1)

    m_aux_all_data_coher = np.zeros((7, 5))
    for bi, banda in enumerate(BANDAS_COHER):
        m_aux_all_data_coher[:, bi] = _network_measures_compacto(coher_medias[banda])

    m_aux_all_data_ev_coher = np.zeros((14, 5))
    for bi in range(5):
        m_aux_all_data_ev_coher[:, bi] = _stats_evolucion(med['ev_coher'][:, bi, :])

    m_aux_all_data_energy = band_energy_medias.reshape(-1, 1)

    # ev_pearson y ev_coher se devuelven también para alimentar las figuras
    # de evolución por ventana (no aparecen en el .mat global de MATLAB).
    return (m_aux_all_data, m_aux_all_data_ev,
            m_aux_all_data_coher, m_aux_all_data_ev_coher,
            m_aux_all_data_energy,
            corr_media_l, coher_medias,
            med['ev_pearson'], med['ev_coher'])


# Función que realiza ventaneo, montaje, cálculo de medidas y agregación. Llamada 
# una vez por archivo desde estudio_coherencia.py.
def creacion_y_medidas_red(matriz_datos_filtrado, caja, params):
    fs = params['frecuencia_salida']
    M = params['longitud_ventana']
    superposicion = params['superposicion_ventana']

    datos, _ = _ventanear(matriz_datos_filtrado, M, superposicion)
    ncanales = len(datos)
    datos_l = _aplicar_montaje(datos, ncanales, caja)
    med = _medidas_por_ventana(datos_l, fs)
    return _agregar_resultados(med)
