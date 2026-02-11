import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\adria\OneDrive - UAB\4 ENGINY\TFG\data\David Harris_Acura ARX06 GTP_Adelaide Street Circuit.csv')

# Filtrar solo una vuelta 
# Aquí asumimos que el archivo es de una sola vuelta o usas LapDistPct para cortar
plt.figure(figsize=(10, 6))

# Pintar el mapa del circuito
# Invertimos X e Y si sale rotado
plt.plot(df['Lon'], df['Lat'])
plt.title("Trazada del Circuito - Datos Garage 61")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.axis('equal') # Para que no se deforme el circuito
plt.grid(True)
plt.show()

# Pintar la telemetría
plt.figure(figsize=(10, 4))
plt.plot(df['LapDistPct'], df['Speed'], label='Velocidad')
plt.plot(df['LapDistPct'], df['Throttle'] * 100, label='Acelerador %', alpha=0.5)
plt.plot(df['LapDistPct'], df['Brake'] * 100, label='Freno %', alpha=0.5)
plt.legend()
plt.show()