import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 1. Leer los datos desde el repositorio
url_datos = 'https://raw.githubusercontent.com/jbcgames/Lab_1_Controls/main/Datos.txt'
df = pd.read_csv(url_datos)

# 2. Extraer columnas (verifica que los nombres coincidan)
tiempo = df['Tiempo (s)'].values
T1 = df[' T1 (C)'].values
pwm = df[' PWM_Heater1'].values

# 2.1. Descartar datos a partir de t = 450 s
mask = (tiempo <= 450)
tiempo = tiempo[mask]
T1 = T1[mask]
pwm = pwm[mask]

# 3. Detectar el instante donde se aplica el escalón (PWM: 0 -> valor positivo)
indices_escalon = np.where((pwm[:-1] == 0) & (pwm[1:] > 0))[0]
if len(indices_escalon) == 0:
    print("No se detectó un escalón (0 -> valor>0) en PWM.")
    exit()
idx_escalon = indices_escalon[0]
t_escalon = tiempo[idx_escalon + 1]
print(f"Escalón detectado en t = {t_escalon:.2f} s")

# 4. Recortar datos desde el escalón en adelante
tiempo = tiempo[idx_escalon:]
T1 = T1[idx_escalon:]
pwm = pwm[idx_escalon:]

# --- APLICAMOS FILTRO Savitzky–Golay A LA SEÑAL T1 ---
window_length = 31  
polyorder = 3 
T1_smooth = savgol_filter(T1, window_length, polyorder)

# 5. Determinar valores inicial y final (usaremos la señal suavizada para la identificación)
y0 = T1_smooth[0]
yf = T1_smooth[-1]
delta_y = yf - y0

# 6. Calcular valores objetivo al 28.3% y 63.2% del cambio
y_28 = y0 + 0.283 * delta_y
y_63 = y0 + 0.632 * delta_y

# 7. Encontrar los instantes donde se alcanza cada valor en la señal suavizada
t_28 = tiempo[np.argmin(np.abs(T1_smooth - y_28))]
t_63 = tiempo[np.argmin(np.abs(T1_smooth - y_63))]

# 8. Calcular las constantes del modelo (método de Smith)
tau = 1.5 * (t_63 - t_28)   # Constante de tiempo
L = t_63 - tau              # Tiempo muerto (retardo)
# Ganancia del proceso (K), usando la amplitud del escalón PWM detectado
#   Asumimos que pwm[idx_escalon+1] es la magnitud del escalón
K = delta_y / pwm[idx_escalon + 1]

print(f"\nModelo aproximado (método de Smith, con filtrado):")
print(f"  Ganancia (K)           = {K:.3f}")
print(f"  Constante de tiempo τ  = {tau:.2f} s")
print(f"  Retardo (L)            = {L:.2f} s")

# 9. Simular el modelo de primer orden con retardo
t_modelo = tiempo
respuesta_modelo = np.zeros_like(t_modelo)

for i, t in enumerate(t_modelo):
    if t < L:
        respuesta_modelo[i] = y0
    else:
        # y(t) = y0 + K*(Escalon)*(1 - e^(-(t - L)/tau))
        respuesta_modelo[i] = y0 + K * pwm[idx_escalon + 1] * (1 - np.exp(-(t - L)/tau))

# 10. Graficar comparación
plt.figure(figsize=(8,5))
plt.plot(tiempo, T1, label='Datos originales', alpha=0.7)
plt.plot(tiempo, T1_smooth, label='Datos suavizados (Savitzky-Golay)')
plt.plot(t_modelo, respuesta_modelo, '--', label='Modelo FOPDT (Smith)')

# Referencia del tiempo muerto
plt.axvline(L, color='gray', linestyle='--', label='Tiempo muerto L')
plt.xlabel('Tiempo [s]')
plt.ylabel('Temperatura [°C]')
plt.title('Identificación con Método de Smith')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
