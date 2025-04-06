import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Leer los datos
df = pd.read_csv('https://raw.githubusercontent.com/jbcgames/Lab_1_Controls/main/Datos.txt')


# 2. Extraer columnas
tiempo = df['Tiempo (s)'].values
T1 = df[' T1 (C)'].values
pwm = df[' PWM_Heater1'].values

# 3. Detectar el instante donde se aplica el escalón en el PWM
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

# 5. Determinar valores inicial y final de la respuesta
y0 = T1[0]
yf = T1[-1]
delta_y = yf - y0

# 6. Calcular la derivada de la respuesta para encontrar el punto de inflexión
# Se usa np.gradient para obtener una aproximación de la derivada
dT1_dt = np.gradient(T1, tiempo)

# Encontrar el índice del máximo de la derivada (punto de máxima pendiente)
idx_inflection = np.argmax(dT1_dt)
t_inf = tiempo[idx_inflection]
y_inf = T1[idx_inflection]
m = dT1_dt[idx_inflection]  # pendiente máxima

# 7. Calcular el tiempo muerto aparente (tm) y la constante de tiempo (tau)
# La recta tangente en el punto de inflexión tiene la forma:
# y(t) = y_inf + m*(t - t_inf)
# - El tiempo muerto tm es el instante donde la tangente corta el valor inicial (y0)
tm = t_inf - (y_inf - y0) / m
# - El tiempo t_u es cuando la tangente alcanza el valor final (yf)
t_u = t_inf + (yf - y_inf) / m
tau = t_u - tm

# 8. Calcular la ganancia del sistema
# Se asume que la magnitud del escalón de entrada es el valor del PWM aplicado
K = delta_y / (pwm[idx_escalon + 1])

print("Modelo aproximado (Ziegler-Nichols):")
print(f"  Ganancia (K) = {K:.3f}")
print(f"  Constante de tiempo (tau) = {tau:.2f} s")
print(f"  Tiempo muerto aparente (tm) = {tm:.2f} s")

# 9. Simular la respuesta del modelo FOPDT (primer orden más tiempo muerto)
t_modelo = tiempo
respuesta_modelo = np.zeros_like(t_modelo)

# Se simula la respuesta: para t < tm, la salida permanece en y0; para t >= tm se aplica la dinámica exponencial.
for i, t in enumerate(t_modelo):
    if t < tm:
        respuesta_modelo[i] = y0
    else:
        respuesta_modelo[i] = y0 + K * pwm[idx_escalon + 1] * (1 - np.exp(-(t - tm) / tau))

# 10. Calcular la línea tangente para visualizar el método
t_tangent = np.linspace(t_inf - 2, t_inf + 2, 100)
y_tangent = y_inf + m * (t_tangent - t_inf)

# 11. Graficar comparación
plt.figure(figsize=(10, 6))
plt.plot(tiempo, T1, label='Datos reales')
plt.plot(t_modelo, respuesta_modelo, '--', label='Modelo FOPDT (Z-N)')
plt.plot(t_tangent, y_tangent, ':', label='Recta tangente en t_inf')
plt.axvline(tm, color='gray', linestyle='--', label='Tiempo muerto (tm)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [°C]')
plt.title('Identificación con el método de la tangente de Ziegler y Nichols')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
