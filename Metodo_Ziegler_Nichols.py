import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 1. Leer los datos
df = pd.read_csv('https://raw.githubusercontent.com/jbcgames/Lab_1_Controls/main/Datos.txt')

# Asegurarse de usar los nombres exactos de las columnas:
tiempo = df['Tiempo (s)'].values
T1 = df[' T1 (C)'].values   # Revisar si el espacio inicial es correcto
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

# 3. Recortar datos desde el escalón en adelante
mask = (tiempo <= 450)

tiempo = tiempo[mask]
T1 = T1[mask]
pwm = pwm[mask]


# 4. Aplicar un filtro (Savitzky-Golay) a la señal de T1
#    - window_length: tamaño de la ventana (debe ser impar y >= polyorder)
#    - polyorder: grado del polinomio
# Ajustar estos valores según la cantidad de ruido y la frecuencia de muestreo
window_length = 51  # prueba con 21, 51, etc. según el tamaño de tus datos
polyorder = 3
T1_smooth = savgol_filter(T1, window_length, polyorder)

# 5. Determinar valores inicial y final (usando la señal NO filtrada o la filtrada; a criterio)
y0 = T1[0]
yf = T1[-1]
delta_y = yf - y0

# 6. Calcular la ganancia del proceso (Kp)
pwm_step = pwm[0] if pwm[0] > 0 else pwm[np.where(pwm > 0)[0][0]]
Kp = delta_y / pwm_step

# 7. Hallar el punto de máxima pendiente usando T1_smooth
#    Calculamos la derivada de la señal suavizada
dT1_dt_smooth = np.gradient(T1_smooth, tiempo)
idx_max_slope = np.argmax(dT1_dt_smooth)

t_inflex = tiempo[idx_max_slope]
y_inflex = T1_smooth[idx_max_slope]
max_slope = dT1_dt_smooth[idx_max_slope]

print(f"Punto de inflexión en t = {t_inflex:.2f} s, con pendiente máxima (suavizada) = {max_slope:.4f}")

# 8. Ecuación de la tangente
#    t_dead: intersección de la tangente con y0
t_dead = t_inflex + (y0 - y_inflex)/max_slope
#    t_end:  intersección de la tangente con yf
t_end  = t_inflex + (yf - y_inflex)/max_slope

tm = t_dead - t_escalon
tau = t_end - t_dead

print("\nModelo FOPDT (Ziegler-Nichols) con filtro aplicado:")
print(f" Ganancia de proceso, Kp  = {Kp:.4f}")
print(f" Tiempo muerto, tm       = {tm:.4f} s")
print(f" Constante de tiempo, τ  = {tau:.4f} s")

# 9. Simular el modelo FOPDT
t_modelo = tiempo.copy()
respuesta_modelo = np.zeros_like(t_modelo)
for i, t in enumerate(t_modelo):
    if t < (t_escalon + tm):
        respuesta_modelo[i] = y0
    else:
        respuesta_modelo[i] = y0 + Kp * pwm_step * (1 - np.exp(-(t - (t_escalon + tm)) / tau))

# 10. Graficar comparación
plt.figure()
plt.plot(tiempo, T1, label='Datos originales')
plt.plot(tiempo, T1_smooth, label='Datos suavizados (Savitzky-Golay)')
plt.plot(t_modelo, respuesta_modelo, '--', label='Modelo Z-N FOPDT (filtrado)')

plt.axvline(t_escalon, color='orange', linestyle='--', label='Inicio escalón')
plt.axvline(t_escalon + tm, color='gray', linestyle='--', label='Tiempo muerto')

plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [°C]')
plt.title('Método Tangente Ziegler-Nichols con Datos Suavizados')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()