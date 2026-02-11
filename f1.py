# Instalar la librería clave
# pip install fastf1 matplotlib pandas

import fastf1
from fastf1 import plotting
import matplotlib.pyplot as plt

# Habilitar caché (para que no descargue todo cada vez)
fastf1.Cache.enable_cache('cache')

# Cargar la sesión de carrera del GP de España 2024
race = fastf1.get_session(2024, 'Spain', 'R')
race.load()

# Ver datos de un piloto (ej. Sainz)
laps_sainz = race.laps.pick_driver('LEC')

# Mostrar las primeras 5 vueltas con sus tiempos y neumáticos
print(laps_sainz[['LapNumber', 'LapTime', 'Compound', 'TyreLife']].head())
