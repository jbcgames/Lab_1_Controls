import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Leer los datos
df = pd.read_csv('https://raw.githubusercontent.com/jbcgames/Lab_1_Controls/main/Datos.txt')

# Asegurarse de usar los nombres exactos de las columnas:
tiempo = df['Tiempo (s)'].values
T1 = df[' T1 (C)'].values   # Ojo con el espacio inicial en la columna
pwm = df[' PWM_Heater1'].values

# 2. Detectar el instante donde se aplica el escalón (PWM: 0 -> valor positivo)
indices_escalon = np.where((pwm[:-1] == 0) & (pwm[1:] > 0))[0]
if len(indices_escalon) == 0:
    print("No se detectó un escalón en la señal PWM.")
    exit()

# Tomamos el primer instante donde ocurre
idx_escalon = indices_escalon[0]
t_escalon = tiempo[idx_escalon + 1]
print(f"Escalón detectado en t = {t_escalon:.2f} s")

# 3. Recortar datos desde el escalón en adelante (para aislar la respuesta)
tiempo = tiempo[idx_escalon:]
T1 = T1[idx_escalon:]
pwm = pwm[idx_escalon:]  # Por si se requiere posteriormente

# 4. Determinar valores inicial (y0) y final (yf) de la respuesta
y0 = T1[0]
yf = T1[-1]
delta_y = yf - y0

# 5. Calcular la ganancia del proceso (k_p)
#    Asumiendo que el escalón del PWM pasa de 0 a un valor constante (pwm_step)
pwm_step = pwm[0] if pwm[0] > 0 else pwm[np.where(pwm>0)[0][0]]
k_p = delta_y / pwm_step

# 6. Hallar el punto de máxima pendiente de la respuesta
#    (punto de inflexión aproximado)
dT1_dt = np.gradient(T1, tiempo)  # Aproxima la derivada dT1/dt
idx_max_slope = np.argmax(dT1_dt) # Índice de la pendiente máxima

t_inflex = tiempo[idx_max_slope]
y_inflex = T1[idx_max_slope]
max_slope = dT1_dt[idx_max_slope]

print(f"Punto de inflexión en t = {t_inflex:.2f} s, con pendiente máxima = {max_slope:.4f}")

# 7. Ecuación de la recta tangente en el punto de inflexión
#    y(t) = y_inflex + max_slope * (t - t_inflex)

# 7.1 Calcular el tiempo donde la tangente cruza el valor inicial (y0)
#     y0 = y_inflex + max_slope * (t_dead - t_inflex)
# =>  t_dead = t_inflex + (y0 - y_inflex)/max_slope
t_dead = t_inflex + (y0 - y_inflex)/max_slope

# 7.2 Calcular el tiempo donde la tangente llega al valor final (yf)
#     yf = y_inflex + max_slope * (t_end - t_inflex)
# =>  t_end = t_inflex + (yf - y_inflex)/max_slope
t_end = t_inflex + (yf - y_inflex)/max_slope

# 8. Tiempo muerto (tm) y constante de tiempo (tau) según Ziegler-Nichols
tm = t_dead - t_escalon  # Retardo (aparente) desde que se aplica el escalón
tau = t_end - t_dead     # Constante de tiempo

# Asegurarnos de que tm y tau sean positivos
if tm < 0:
    print("Advertencia: el tiempo muerto calculado es negativo; revisar la detección del escalón o datos.")
if tau < 0:
    print("Advertencia: la constante de tiempo resultó negativa; revisar datos.")

print(f"\nModelo FOPDT (Ziegler-Nichols):")
print(f" Ganancia de proceso, Kp  = {k_p:.4f}")
print(f" Tiempo muerto, tm       = {tm:.4f} s")
print(f" Constante de tiempo, τ  = {tau:.4f} s")

# 9. Simular el modelo FOPDT con retardo (forma canónica)
#    y(t) = y0,                  si t < (t_escalon + tm)
#    y(t) = y0 + Kp * pwm_step * (1 - exp(-(t - (t_escalon + tm))/tau)), si t >= (t_escalon + tm)

t_modelo = tiempo.copy()
respuesta_modelo = np.zeros_like(t_modelo)

for i, t in enumerate(t_modelo):
    if t < (t_escalon + tm):
        respuesta_modelo[i] = y0
    else:
        respuesta_modelo[i] = y0 + k_p * pwm_step * (1 - np.exp(-(t - (t_escalon + tm))/tau))

# 10. Graficar comparación
plt.figure()
plt.plot(tiempo, T1, label='Datos reales')
plt.plot(t_modelo, respuesta_modelo, '--', label='Modelo Ziegler-Nichols')

# Líneas verticales de referencia
plt.axvline(t_escalon, color='orange', linestyle='--', label='Inicio escalón')
plt.axvline(t_escalon + tm, color='gray', linestyle='--', label='Tiempo muerto')
plt.axhline(y0, color='red', linestyle=':', label='Valor inicial')
plt.axhline(yf, color='green', linestyle=':', label='Valor final')

plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [°C]')
plt.title('Identificación con Método de la Tangente de Ziegler y Nichols')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
