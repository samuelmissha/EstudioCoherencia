"""
Cálculos numéricos sobre ventanas: coeficiente de Pearson, matrices
de correlación/retardos y coherencia espectral por bandas (más energía por
banda y canal).

Equivalente MATLAB:
- coef_pearson   -> coef_pearson.m
- corr_matrix    -> corr_matrix.m
- band_energy    -> band_energy.m
- band_coherence -> band_coherence.m

DIFERENCIAS vs MATLAB:
- coef_pearson: MATLAB no resta la media. Aquí aplicamos Pearsonclásico 
  restando la media. 
- band_energy: MATLAB selecciona los bins por índice entero, lo que cuenta 
  el bin frontera DOS VECES. Aquí asignamos a cada bin a una única banda. 
  También se excluye el bin DC (f >= 0.5).
- band_coherence: MATLAB selecciona bins por índice entero; aquí por valor
  con fftfreq (más preciso y permite excluir el bin DC).
"""

import numpy as np


# Coeficiente de Pearson.
def coef_pearson(canal1, canal2):
    c1 = canal1 - np.mean(canal1)
    c2 = canal2 - np.mean(canal2)

    den = np.sqrt(np.sum(c1**2) * np.sum(c2**2))
    if den == 0:
        return 0.0, 0

    return abs(float(np.sum(c1 * c2) / den)), 0


# Matrices simétricas de Pearson y retardos para todos los pares de canales
# en la ventana indicada. Solo se recorre la triangular superior y se duplica
# al triángulo inferior (como en MATLAB).
def corr_matrix(datos_celdas, ventana):
    n = len(datos_celdas)
    matriz_pearson = np.zeros((n, n))
    matriz_retardos = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pearson, lag = coef_pearson(datos_celdas[i][:, ventana], datos_celdas[j][:, ventana])
            matriz_pearson[i, j] = matriz_pearson[j, i] = pearson
            matriz_retardos[i, j] = matriz_retardos[j, i] = lag
    return matriz_pearson, matriz_retardos


# Reparte la potencia de un espectro entre las 5 bandas EEG estándar.
def band_energy(ps_arr, f_arr):
    bands = np.zeros(5)

    idx = np.where((f_arr >= 0.5) & (f_arr <= 4))[0]
    bands[0] = np.sum(ps_arr[idx])

    idx = np.where((f_arr > 4) & (f_arr <= 7))[0]
    bands[1] = np.sum(ps_arr[idx])

    idx = np.where((f_arr > 7) & (f_arr <= 13))[0]
    bands[2] = np.sum(ps_arr[idx])

    idx = np.where((f_arr > 13) & (f_arr <= 30))[0]
    bands[3] = np.sum(ps_arr[idx])

    idx = np.where(f_arr > 30)[0]
    bands[4] = np.sum(ps_arr[idx])

    return bands


# Coherencia espectral por bandas. Devuelve también la matriz de energía por 
# banda × canal que alimenta el reporte visual.
def band_coherence(datos_celdas, fs, ventana):
    ncanales = len(datos_celdas)
    n = len(datos_celdas[0][:, ventana])

    delta_coher = np.zeros((ncanales, ncanales))
    theta_coher = np.zeros((ncanales, ncanales))
    alpha_coher = np.zeros((ncanales, ncanales))
    beta_coher = np.zeros((ncanales, ncanales))
    gamma_coher = np.zeros((ncanales, ncanales))

    f_arr = np.fft.fftfreq(n, 1/fs)
    idx_pos = f_arr > 0
    f_arr = f_arr[idx_pos]

    matriz_energia_bandas = np.zeros((5, ncanales))

    for i in range(ncanales):
        fft_i = np.fft.fft(datos_celdas[i][:, ventana])
        p_ii = (np.abs(fft_i)**2) / n
        p_ii_pos = p_ii[idx_pos]

        matriz_energia_bandas[:, i] = band_energy(p_ii_pos, f_arr)

        for j in range(i + 1, ncanales):
            fft_j = np.fft.fft(datos_celdas[j][:, ventana])
            p_jj = (np.abs(fft_j)**2) / n
            p_jj_pos = p_jj[idx_pos]

            p_ij = (fft_i * np.conjugate(fft_j)) / n
            p_ij_pos = p_ij[idx_pos]

            # Función que calcula la coherencia de magnitud cuadrada (MSC) 
            # entre dos canales i y j para una banda de frecuencia [fmin, fmax].
            def calc_coherence(fmin, fmax):
                idx = np.where((f_arr >= fmin) & (f_arr <= fmax))[0]
                if len(idx) == 0:
                    return 0
                num = np.abs(np.mean(p_ij_pos[idx]))**2
                den = np.mean(p_ii_pos[idx]) * np.mean(p_jj_pos[idx])
                return num / den if den != 0 else 0

            c_delta = calc_coherence(0.5, 4)
            delta_coher[i, j] = delta_coher[j, i] = c_delta

            c_theta = calc_coherence(4, 7)
            theta_coher[i, j] = theta_coher[j, i] = c_theta

            c_alpha = calc_coherence(7, 13)
            alpha_coher[i, j] = alpha_coher[j, i] = c_alpha

            c_beta = calc_coherence(13, 30)
            beta_coher[i, j] = beta_coher[j, i] = c_beta

            c_gamma = calc_coherence(30, fs/2)
            gamma_coher[i, j] = gamma_coher[j, i] = c_gamma

    return delta_coher, theta_coher, alpha_coher, beta_coher, gamma_coher, matriz_energia_bandas
