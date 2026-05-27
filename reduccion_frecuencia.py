"""
Diezma la señal multicanal a la frecuencia de salida configurada en
`params['frecuencia_salida']` y diezma el vector de tiempos en paralelo para
que siga alineado con los datos.

Equivalente MATLAB: reduccionFrecuencia.m

DIFERENCIAS vs MATLAB:
- Ninguna. MATLAB usa downsample(matriz, fr) (submuestreo directo seleccionando
  1 de cada fr muestras). Aquí se usa slicing de NumPy [::fr], que es
  exactamente lo mismo.
"""


# Si la frecuencia objetivo es mayor o igual a la de entrada se devuelven los
# datos tal cual.
def reduccion_frecuencia(matriz_datos, t, frecuencia_entrada, params):
    frecuencia_salida = params['frecuencia_salida']
    if frecuencia_salida > frecuencia_entrada:
        frecuencia_salida = frecuencia_entrada

    fr = round(frecuencia_entrada / frecuencia_salida)
    print(f"Reducción de la frecuencia de {round(frecuencia_entrada)} a {frecuencia_salida} Hz")

    if fr > 1:
        matriz_datos_diezmada = matriz_datos[::fr, :]
        t_diezmado = t[::fr, :]
    else:
        matriz_datos_diezmada = matriz_datos
        t_diezmado = t

    return frecuencia_salida, matriz_datos_diezmada, t_diezmado
