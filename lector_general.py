"""
Lee un archivo .txt de XLTEK: extrae frecuencia de muestreo y headbox de la
cabecera, selecciona los canales scalp + ECG según el modelo de caja, filtra
(pasa-banda + notch de 50 Hz) y recorta el transitorio inicial por umbral
de amplitud. Devuelve la matriz filtrada lista para diezmar.

Equivalente MATLAB: lector_general.m

DIFERENCIAS vs MATLAB:
- MATLAB usa textscan con un formato explícito por tipo de caja; aquí cargamos
  toda la tabla con pd.read_csv y seleccionamos columnas por índice.
- MATLAB filtra con filter (causal, retraso de fase). Aquí usamos filtfilt
  (bidireccional, orden eficaz doble, sin retraso). Las señales filtradas
  DIFIEREN, especialmente cerca de las frecuencias de corte.

"""

import numpy as np
import pandas as pd
import re
from scipy.signal import butter, filtfilt


def lector_general(path, params):
    pasa_altos_scalp = params['pasa_altos_scalp']
    pasa_bajos_scalp = params['pasa_bajos_scalp']
    pasa_altos_ecg = params['pasa_altos_ecg']
    pasa_bajos_ecg = params['pasa_bajos_ecg']
    notch1 = params['notch1']
    notch2 = params['notch2']

    directorio = path

    # LECTURA DATOS
    frecuencia = None
    caja = None

    # latin1 porque XLTEK escribe caracteres no UTF-8 en la cabecera.
    # Igual que en MATLAB.
    with open(directorio, 'r', encoding='latin1') as f:
        for i in range(1, 16):
            linea = f.readline().strip()
            if i == 8:
                frecuencia = re.findall(r"\d+\.\d+|\d+", linea)
                frecuencia = float(frecuencia[0])
            elif i == 11:
                caja = int(''.join(filter(str.isdigit, linea)))

    periodo = 1.0 / frecuencia

    # El canal Fpz del equipo UCI1 (caja 1528) está cerrado: se reasigna a 1138
    # para reutilizar la misma configuración de columnas.
    if caja == 1528:
        caja = 1138

    # DIFERENCIA vs MATLAB: MATLAB descarta columnas con %*s/%*f/%*d. Aquí
    # cargamos toda la tabla y seleccionamos columnas por índice después.
    datos = pd.read_csv(directorio, skiprows=15, sep=r'\s+', header=None, on_bad_lines='skip', 
                     low_memory=False, na_values=['', ' '])

    if caja in [1527, 1528]:
        indices_scalp = list(range(3, 23))
        indices_ecg = [25, 26]
    elif caja == 1138:
        indices_scalp = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        indices_ecg = [25, 26]
    elif caja == 65535:
        # Headbox NRL/Neurocirugía-Neurología: se procesa NRL (opción 4) por defecto.
        indices_scalp = [3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27]
        indices_ecg = [30, 31]
    else:
        raise ValueError(f"Caja no reconocida: {caja}")

    matriz_scalp = datos.iloc[:, indices_scalp].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    matriz_ecg = datos.iloc[:, indices_ecg].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

    # Filtra filas con cualquier NaN para mantener ambas matrices alineadas.
    mask_validas = ~(np.isnan(matriz_scalp).any(axis=1) | np.isnan(matriz_ecg).any(axis=1))
    matriz_scalp = matriz_scalp[mask_validas]
    matriz_ecg = matriz_ecg[mask_validas]

    # FILTRADO SEÑALES
    nyq = 0.5 * frecuencia

    bn, an = butter(4, [notch1 / nyq, notch2 / nyq], btype='bandstop')
    matriz_scalp_notch = filtfilt(bn, an, matriz_scalp, axis=0)
    matriz_ecg_notch = filtfilt(bn, an, matriz_ecg, axis=0)

    # filtfilt aplica el filtro de doble pasada: cero retraso de fase y orden
    # eficaz doble. Equivalente offline de mejor calidad que el filter que se
    # usa MATLAB.
    def aplicar_filtro(datos, pasa_altos, pasa_bajos, nyq_freq):
        bajos = pasa_bajos / nyq_freq
        if pasa_altos == 0:
            b, a = butter(4, bajos, btype='low')
        else:
            altos = pasa_altos / nyq_freq
            b, a = butter(4, [altos, bajos], btype='bandpass')
        return filtfilt(b, a, datos, axis=0)

    matriz_scalp_filt = aplicar_filtro(matriz_scalp_notch, pasa_altos_scalp, pasa_bajos_scalp, nyq)
    matriz_ecg_filt = aplicar_filtro(matriz_ecg_notch, pasa_altos_ecg, pasa_bajos_ecg, nyq)

    matriz_datos_filtrado = np.hstack((matriz_scalp_filt, matriz_ecg_filt))

    # ELIMINACIÓN DE ARTEFACTOS INICIALES
    # Se descartan los primeros 5 s (transitorio del filtfilt) antes de estimar
    # el umbral, para no contaminar media y std con la cola del filtro.
    n5 = int(frecuencia * 5)
    matriz_transitoria = matriz_datos_filtrado[n5:, :]

    media_matriz_transitoria = np.mean(matriz_transitoria, axis=0)
    std_matriz_transitoria = np.std(matriz_transitoria, axis=0)
    umbral_maximo_matriz_transitoria = np.ceil(np.max(media_matriz_transitoria + 5 * std_matriz_transitoria))

    matriz_absoluta = np.abs(matriz_datos_filtrado)
    punto_corte = 1

    for i in range(20):
        puntos_fuera = np.sum(matriz_absoluta[:, i] > umbral_maximo_matriz_transitoria)  # Cuenta puntos fuera de umbral en cada canal
        if puntos_fuera > punto_corte:
            punto_corte = puntos_fuera  

    # Margen ×3 sobre el punto de corte para eliminación de artefactos (igual que en MATLAB).
    matriz_datos_filtrado = matriz_datos_filtrado[3 * punto_corte:, :]

    # VECTOR DE TIEMPO
    n_max = matriz_datos_filtrado.shape[0]
    tmax = n_max * periodo
    t = np.arange(0, tmax, periodo).reshape(-1, 1)

    return frecuencia, t, matriz_datos_filtrado, caja
