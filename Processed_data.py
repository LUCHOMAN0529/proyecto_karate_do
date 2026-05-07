import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('datos_fuerza_segmentados.csv')
m, g, fps = 70.0, 9.81, 60
dt = 1/fps
df['Tiempo'] = df['Frame'] * dt

# 2. Cálculos: Fuerza Neta -> Aceleración -> Velocidad -> Desplazamiento
df['F_neta_Z'] = df['Fuerza_Z'] - (m * g)

# 3. Rate of Force Development (Derivada)
df['RFD_Z'] = df['Fuerza_Z'].diff() / dt

# 4. Generación de Gráficas
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
#axs[0].plot(df['Tiempo'], df['Aceleracion_Z'], color='red')
#axs[0].set_title('Aceleración Vertical (Z)'); axs[0].set_ylabel('m/s²')
#axs[1].plot(df['Tiempo'], df['Velocidad_Z'], color='blue')
#axs[1].set_title('Velocidad Vertical (Z)'); axs[1].set_ylabel('m/s')
#axs[2].plot(df['Tiempo'], df['Desplazamiento_Z'], color='green')
#axs[2].set_title('Desplazamiento Vertical (Z)'); axs[2].set_xlabel('Tiempo (s)')
plt.tight_layout()
plt.show()