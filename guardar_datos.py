"""
Construye la fila de features (131 columnas con nombre) para el CSV, guarda 
las series temporales y matrices de conectividad en .npz, y mantiene actualizado 
features_ml.csv.

DIFERENCIA vs MATLAB: MATLAB guarda todo en un único StructAllData.mat (cell array). 
Aquí se reparte en CSV + .npz por paciente porque alimenta analisis_ml.py.
"""

import os
import unicodedata
import numpy as np
import pandas as pd

_METRICAS = ['APL', 'DOL', 'EGLOB', 'ACC', 'T', 'MOD', 'SW']
_BANDAS   = ['delta', 'theta', 'alpha', 'beta', 'gamma']


# Elimina acentos de un texto, para que no descarte registros si el nombre del archivo tiene acentos.
def _quitar_acentos(texto):
    descompuesto = unicodedata.normalize('NFD', texto)
    return ''.join(c for c in descompuesto if not unicodedata.combining(c))

# Extrae la etiqueta de clase a partir del nombre del archivo. 
def _extraer_etiqueta(archivo):
    nombre = _quitar_acentos(archivo).lower()
    if 'encefalopatia2' in nombre or 'encefalopatia_2' in nombre:
        return 'encefalopatia2'
    if 'encefalopatia1' in nombre or 'encefalopatia_1' in nombre or 'encefalopatia' in nombre:
        return 'encefalopatia1'
    if 'normal' in nombre:
        return 'normal'
    return 'desconocido'


# Aplana el diccionario de resultados en una fila con 131 columnas nombradas: 7 Pearson estático, 
# 14 Pearson temporal, 35 coherencia estática (7×5), 70 coherencia temporal (14×5), 5 energías por banda.
def construir_fila_features(resultado):
    fila = {
        'archivo': resultado['archivo'],
        'etiqueta': _extraer_etiqueta(resultado['archivo']),
    }

    for k, m in enumerate(_METRICAS):
        fila[f'pearson_{m}'] = float(resultado['m_aux_all_data'][k, 0])

    for k, m in enumerate(_METRICAS):
        fila[f'pearson_ev_{m}_mean'] = float(resultado['m_aux_all_data_ev'][2 * k,     0])
        fila[f'pearson_ev_{m}_std']  = float(resultado['m_aux_all_data_ev'][2 * k + 1, 0])

    for bi, b in enumerate(_BANDAS):
        for k, m in enumerate(_METRICAS):
            fila[f'coher_{b}_{m}'] = float(resultado['m_aux_all_data_coher'][k, bi])

    for bi, b in enumerate(_BANDAS):
        for k, m in enumerate(_METRICAS):
            fila[f'coher_ev_{b}_{m}_mean'] = float(resultado['m_aux_all_data_ev_coher'][2 * k,     bi])
            fila[f'coher_ev_{b}_{m}_std']  = float(resultado['m_aux_all_data_ev_coher'][2 * k + 1, bi])

    for k, b in enumerate(_BANDAS):
        fila[f'energy_{b}'] = float(resultado['m_aux_all_data_energy'][k, 0])

    return fila


# Guarda series por ventana y matrices medias en datos_ml.npz. Estas series se usan en analisis_ml.py 
# para construir features dinámicas.
def guardar_series_npz(archivo, ev_pearson, ev_coher, corr_media_l, coher_medias, patient_dir):
    ruta = os.path.join(patient_dir, 'datos_ml.npz')
    np.savez_compressed(
        ruta,
        ev_pearson=ev_pearson,
        ev_coher=ev_coher,
        corr_media_l=corr_media_l,
        coher_media_delta=coher_medias['delta'],
        coher_media_theta=coher_medias['theta'],
        coher_media_alpha=coher_medias['alpha'],
        coher_media_beta=coher_medias['beta'],
        coher_media_gamma=coher_medias['gamma'],
    )
    print(f"  - datos_ml.npz guardado en {ruta}")

# features_ml.csv es un CSV que sirve de entrada directa a analisis_ml.py (Machine Learning).
# Si se vuelve a procesar un registro, su fila previa se sustituye en lugar de duplicarse.
def guardar_features_csv(filas, base_dir):
    ruta_csv = os.path.join(base_dir, 'features_ml.csv')
    datos_nuevo = pd.DataFrame(filas)

    if os.path.exists(ruta_csv):
        datos_existente = pd.read_csv(ruta_csv)
        datos_existente = datos_existente[~datos_existente['archivo'].isin(datos_nuevo['archivo'])]
        # Número de registros finales tras agregar los nuevos y eliminar los duplicados. 
        datos_final = pd.concat([datos_existente, datos_nuevo], ignore_index=True)
    else:
        datos_final = datos_nuevo

    datos_final.to_csv(ruta_csv, index=False)
    print(f"  - features_ml.csv actualizado: {len(datos_final)} registros en {ruta_csv}")
