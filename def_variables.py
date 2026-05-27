"""
Diccionario único de configuración del pipeline: frecuencias de corte de los
filtros, frecuencia de diezmado, tamaño y solapamiento de las ventanas.

Equivalente MATLAB: DefVariables.m

DIFERENCIAS vs MATLAB:
- DefVariables.m define superposicion=0.2. En el estudio de Mario Refoyo se 
  consideró superposicion=0 como idóneo, así que se mantiene ese valor.
"""


# Devuelve el diccionario único de parámetros. 
def def_variables():
    return {
        'pasa_altos_scalp': 0.5,
        'pasa_bajos_scalp': 30,
        'pasa_altos_ecg': 3,
        'pasa_bajos_ecg': 25,
        'notch1': 45,
        'notch2': 55,
        'frecuencia_salida': 128,            # fs de salida idónea
        'longitud_ventana': 512,
        'superposicion_ventana': 0.0,
    }
