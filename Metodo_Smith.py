import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Leer los datos
df = pd.read_csv('https://raw.githubusercontent.com/jbcgames/Lab_1_Controls/main/Datos.txt')

# 2. Extraer columnas
tiempo = df['Tiempo (s)'].values
T1 = df[' T1 (C)'].values
pwm = df[' PWM_Heater1'].values

# 3. Detectar el instante donde se aplica el escalón
indices_escalon = np.where((pwm[:-1] == 0) & (pwm[1:] > 0))[0]
if len(indices_escalon) == 0:
    print("No se detectó un escalón.")
    exit()

idx_escalon = indices_escalon[0]
t_escalon = tiempo[idx_escalon + 1]
print(f"Escalón detectado en t = {t_escalon:.2f} s")

# 4. Recortar datos desde el escalón en adelante
tiempo = tiempo[idx_escalon:]
T1 = T1[idx_escalon:]

# 5. Determinar valores inicial y final
y0 = T1[0]
yf = T1[-1]
delta_y = yf - y0

# 6. Calcular valores objetivo al 28.3% y 63.2% del cambio
y_28 = y0 + 0.283 * delta_y
y_63 = y0 + 0.632 * delta_y

# 7. Encontrar tiempos donde se alcanza ese valor
t_28 = tiempo[np.argmin(np.abs(T1 - y_28))]
t_63 = tiempo[np.argmin(np.abs(T1 - y_63))]

# 8. Calcular constantes del modelo
tau = 1.5 * (t_63 - t_28)
L = t_63 - tau
K = delta_y / (pwm[idx_escalon + 1])  # Asumiendo escalón de PWM

print(f"Modelo aproximado:")
print(f"  Ganancia (K) = {K:.3f}")
print(f"  Constante de tiempo (tau) = {tau:.2f} s")
print(f"  Retardo (L) = {L:.2f} s")

# 9. Simular el modelo de primer orden con retardo
t_modelo = tiempo
respuesta_modelo = np.zeros_like(t_modelo)

for i, t in enumerate(t_modelo):
    if t < L:
        respuesta_modelo[i] = y0
    else:
        respuesta_modelo[i] = y0 + K * pwm[idx_escalon + 1] * (1 - np.exp(-(t - L) / tau))

# 10. Graficar comparación
plt.figure()
plt.plot(tiempo, T1, label='Datos reales')
plt.plot(t_modelo, respuesta_modelo, '--', label='Modelo de Smith')
plt.axvline(L, color='gray', linestyle='--', label='Tiempo muerto')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [°C]')
plt.title('Identificación con Método de Smith')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()