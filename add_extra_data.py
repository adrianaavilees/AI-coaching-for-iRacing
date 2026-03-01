import requests
import pandas as pd
import os 
import time
from pathlib import Path

cookies = {
    'auth': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMUtFUTZQSlRaWFFSMlRQNjBNNlpHVDRKRSIsImV4cCI6MTk5MTU3ODEwMiwibmJmIjoxNzcwODI2MTAyLCJpYXQiOjE3NzA4MjYxMDIsInR5cCI6ImFwcCJ9.gVCD8Cz9eDI5pJWeYNKbpC3a-7c5s0ZrCMmrzjrzTM8',
}

headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'es-ES,es;q=0.9,ca;q=0.8',
    'Connection': 'keep-alive',
    'Referer': 'https://garage61.net/app/laps/53/155;a=-1;g=2',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    # 'Cookie': 'auth=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMUtFUTZQSlRaWFFSMlRQNjBNNlpHVDRKRSIsImV4cCI6MTk5MTU3ODEwMiwibmJmIjoxNzcwODI2MTAyLCJpYXQiOjE3NzA4MjYxMDIsInR5cCI6ImFwcCJ9.gVCD8Cz9eDI5pJWeYNKbpC3a-7c5s0ZrCMmrzjrzTM8',
}

csv_folder = r"C:\Users\AvAd738\OneDrive - HP Inc\Documents\Adriana\AI-coaching-for-iRacing\data\Ferrari 296 GT3\Imola"
folder_path = Path(csv_folder)

files = list(folder_path.glob('*.csv'))
print(f"Found {len(files)} CSV files in the folder.")

data = []
# Iterate through each file we have download
for file in files:
    file_name = file.name

    # Extract the lap_id, garage61 always put the ID at the end of the file name, before the .csv extension
    # Ex: Garage 61 - Piloto - Coche - Pista - Tiempo - 01KDG8B3QB3MHC5B708S58FFD5.csv
    try:
        lap_id = file_name.split(' - ')[-1].replace('.csv', '')
    except IndexError:
        print(f"We couldn't extract the lap_id from the file name '{file_name}'. Skipping this file.")
        continue

    print(f"Obtaining data for lap '{lap_id}'...")

    # Do the request to the API endpoint to get the laps data, including weather conditions
    URL = f'https://garage61.net/api/internal/laps/{lap_id}'
    response = requests.get(URL, cookies=cookies, headers=headers)

    if response.status_code == 200:
        lap = response.json()
        
        data.append({
            'lap_id': lap_id, 
            'driver': lap.get('driver_name'),
            'lap_time_str': lap.get('lap_time_str', ''), 
            'air_temp': lap.get('weather_air_temp'),
            'track_temp': lap.get('track_temp', lap.get('weather_air_temp')), 
            'track_usage': lap.get('track_usage'),
            'humidity': lap.get('weather_relative_humidity'),
            'wind_vel': lap.get('weather_wind_vel'),
            'wind_dir': lap.get('weather_wind_dir'),
            'precipitation': lap.get('weather_precipitation'),
            'cloud_cover': lap.get('weather_cloud_cover'),
            'fog_level': lap.get('weather_fog_level'),
            'fuel_level': lap.get('fuel_level'),
            'fuel_used': lap.get('fuel_used'),
            'tire_compound': lap.get('tire_compound'),
        })
        print(f"Data for lap '{lap_id}' obtained successfully.")
    else:
        print(f"Error {response.status_code}")
        
    #! IMPORTANT: Wait 1.5 seconds between each request to avoid overloading the server
    #! and prevent our account or IP from being banned for making "spam" requests
    time.sleep(1.5)

# Convert the list of dicts to a DataFrame and save to CSV
if data:
    df_clima = pd.DataFrame(data)
    output_file = 'garage61_extra_data.csv'
    df_clima.to_csv(output_file, index=False)
    
    print(f"Data saved to '{output_file}'")
    
    # Display the first 5 rows to check everything is fine
    print(df_clima.head())
    
else:
    print(f"Request failed.")
