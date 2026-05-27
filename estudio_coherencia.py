"""
Selección de registros EEG, procesado por paciente lectura, diezmado, ventanas,
métricas de red sobre Pearson y coherencia por bandas, energía, figuras y preparación
de datos para el análisis ML posterior.

Equivalente MATLAB: EstudioCoherencia.m

DIFERENCIAS vs MATLAB:
- MATLAB guarda todo en StructAllData.mat (cell array). Python crea un
  subdirectorio por paciente con figuras y datos_ml.npz, y también un CSV
  agregado features_ml.csv (una fila por paciente) que alimenta analisis_ml.py.
"""

import os
import time
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from def_variables import def_variables
from lector_general import lector_general
from reduccion_frecuencia import reduccion_frecuencia
from creacion_medidas_red import creacion_y_medidas_red
from informe_visual import generar_informe_visual, generar_evolucion_pearson, generar_evolucion_coherencia
from guardar_datos import construir_fila_features, guardar_series_npz, guardar_features_csv


# Carpeta donde se crea un subdirectorio por paciente con sus figuras y datos_ml.npz.
DIR_RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Resultados')


# Abre un diálogo gráfico para que el usuario seleccione uno o varios archivos .txt con registros EEG.
def _seleccionar_archivos():
    root = tk.Tk()
    root.withdraw()         # Oculta la ventana raíz de Tk; solo se muestra el diálogo
    return filedialog.askopenfilenames(
        title='Selecciona varios archivos',
        filetypes=[('Text Files', '*.txt')]
    )


# Procesa un único registro EEG de principio a fin: lectura del .txt, diezmado a fs_salida y cálculo de
#  métricas de red por ventana (Pearson + coherencia por bandas + energía).
def _procesar_archivo(path_completo, params):
    archivo = os.path.splitext(os.path.basename(path_completo))[0]
    print(f"Procesando archivo: {archivo}")

    frecuencia, t, matriz_datos_filtrado, caja = lector_general(path_completo, params)

    frecuencia, matriz_datos_filtrado, t = reduccion_frecuencia(matriz_datos_filtrado, t, frecuencia, params)

    (m_aux_all_data, m_aux_all_data_ev,
     m_aux_all_data_coher, m_aux_all_data_ev_coher,
     m_aux_all_data_energy, corr_media_l, coher_medias,
     ev_pearson, ev_coher) = creacion_y_medidas_red(matriz_datos_filtrado, caja, params)

    # Datos numéricos que se convierten en una fila del CSV agregado.
    # Equivalente MATLAB: StructAllData{indGeneral, :} con 5 celdas.
    resultado = {
        'archivo': archivo,
        'm_aux_all_data': m_aux_all_data,
        'm_aux_all_data_ev': m_aux_all_data_ev,
        'm_aux_all_data_coher': m_aux_all_data_coher,
        'm_aux_all_data_ev_coher': m_aux_all_data_ev_coher,
        'm_aux_all_data_energy': m_aux_all_data_energy,
    }

    # Tupla con las series temporales por ventana, las matrices medias de conectividad y distribución de
    # energía por banda para las figuras del reporte visual.
    datos_viz = (archivo, corr_media_l, coher_medias, m_aux_all_data_energy, ev_pearson, ev_coher)

    return resultado, datos_viz


def main():
    t_inicio = time.perf_counter()

    archivos_paths = _seleccionar_archivos()
    if not archivos_paths:
        print("No se seleccionaron archivos. Terminando ejecución.")
        return

    params = def_variables()
    filas_features = []    # Filas del CSV agregado para análisis ML

    for path in archivos_paths:
        resultado, datos_viz = _procesar_archivo(path, params)
        archivo_v, corr_v, coher_v, energy_v, ev_p, ev_c = datos_viz

        # Subdirectorio dedicado por paciente: 3 figuras + datos_ml.npz con series temporales y matrices medias.
        patient_dir = os.path.join(DIR_RESULTADOS, archivo_v)
        os.makedirs(patient_dir, exist_ok=True)

        figuras = [
            generar_informe_visual(archivo_v, corr_v, coher_v, energy_v),
            generar_evolucion_pearson(archivo_v, ev_p),
            generar_evolucion_coherencia(archivo_v, ev_c),
        ]
        for i, fig in enumerate(figuras, 1):
            fig.savefig(os.path.join(patient_dir, f'figura{i}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)                            # Libera memoria; matplotlib retiene figs si no se cierran


        # DIFERENCIA vs MATLAB: MATLAB no guarda las series por ventana, solo las matrices medias en el .mat global.
        guardar_series_npz(archivo_v, ev_p, ev_c, corr_v, coher_v, patient_dir)
        filas_features.append(construir_fila_features(resultado))

    #CSV legible que sirve de entrada directa a analisis_ml.py.
    if filas_features:
        guardar_features_csv(filas_features, DIR_RESULTADOS)

    t_total = time.perf_counter() - t_inicio
    horas, resto = divmod(t_total, 3600)
    minutos, segundos = divmod(resto, 60)
    print(f'\nTiempo total de ejecución: {t_total:.2f} s ({int(horas):02d}:{int(minutos):02d}:{segundos:05.2f})')


if __name__ == '__main__':
    main()
