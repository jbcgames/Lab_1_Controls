import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Leer el archivo de texto
#    Si tu archivo tiene encabezados en la primera fila y está separado por comas:
df = pd.read_csv('Datos.txt')

# 2. Extraer las columnas relevantes
#    Ajusta los nombres de columnas de acuerdo con tu archivo:
tiempo = df['Tiempo (s)'].values
T1 = df[' T1 (C)'].values
pwm = df[' PWM_Heater1'].values

# (Opcional) Verificar rápidamente cuántas filas y columnas se leyeron
print("Número de muestras leídas:", len(df))
print(df.head())

indices_escalon = np.where((pwm[:-1] == 0) & (pwm[1:] > 0))[0]

if len(indices_escalon) == 0:
    print("No se encontró un escalón en PWM_Heater1")
else:
    idx_escalon = indices_escalon[0]  # Tomar el primer cambio
    t_escalon = tiempo[idx_escalon+1]
    print(f"Se detectó un escalón en PWM en el instante t = {t_escalon} s")
    
    # A partir de este índice, podemos tomar la porción de datos posterior al escalón
    tiempo_escalon = tiempo[idx_escalon:]
    T1_escalon = T1[idx_escalon:]
    pwm_escalon = pwm[idx_escalon:]