from functools import reduce
import os
import zipfile
import requests
import io
import pandas as pd
import numpy as np


IMGW_STATION_ID = 566
GIOS_STATION_CODES = ['MpKrakBulwar','MpKrakowWIOSBulw6118']
YEARS_FROM = 2010
YEARS_TO = 2024

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
}

IMGW_URL = 'https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop'

def fetch_imgw_data(years_from,years_to):
    all_data = []
    
    for year in range(years_from, years_to + 1):
        filename = f'{year}_{IMGW_STATION_ID}_s.zip'
        url = f'{IMGW_URL}/{year}/{filename}'
        
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            
            if response.status_code != 200:
                print(f"Failed to download {filename}: Status code {response.status_code}")
                continue
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                file_names = [name for name in z.namelist() if '.csv' in name and 's_t' in name]
                if not file_names:
                    continue
                with z.open(file_names[0]) as f:
                    df = pd.read_csv(f,header=None,encoding='cp1250',low_memory=False)
                    df = df.iloc[:,[2,3,4,5,29,37,43]].copy()
                    df.columns = ['year','month','day','hour','temp_c','humidity_percent','pressure_hpa']
                    
                    for col in ['temp_c','humidity_percent','pressure_hpa']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                    all_data.append(df)
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    if not all_data:
        print("No data fetched.")
        return pd.DataFrame()
    
    final_df = pd.concat(all_data)
    
    final_df['timestamp'] = pd.to_datetime(final_df[['year','month','day','hour']].astype(str).agg('.'.join, axis=1) + ':00')
    exclude = (final_df['day'] == 1) & (final_df['month'] == 1) & (final_df['hour'] == 0)
    final_df = final_df[~exclude]
    
    return final_df[['timestamp','temp_c','humidity_percent','pressure_hpa']].sort_values('timestamp')

def load_gios_data(pollutant_name,station_codes,years_from,years_to):
    all_data = []
    for year in range(years_from, years_to + 1):
        filename = f'{year}_{pollutant_name}_1g.xlsx'
        
        if not os.path.exists(f'../pm_data/{filename}'):
            print(f"File {filename} does not exist.")
            continue
        try:
            print(f"Loading data from {filename}...")
                   
            df = pd.read_excel(f'../pm_data/{filename}', header = None)
            
            indexes = df.iloc[:10].isin(station_codes).any()
            
            if not indexes.any():
                print(f"Station code not found in {filename}.")
                continue
            
            data_index = np.where(indexes)[0][0]
            potential_dates = pd.to_datetime(df.iloc[:20,0], errors='coerce',format='%d.%m.%Y %H:%M:%S')
            first_date_row = potential_dates.notna().idxmax()
            extracted_data = df.iloc[first_date_row:, [0, data_index]].copy().reset_index(drop=True)
            extracted_data.columns = ['timestamp', pollutant_name]
            extracted_data['timestamp'] = pd.to_datetime(extracted_data['timestamp'], format='%Y-%m-%d %H:%M:00')
            
            exclude = (extracted_data['timestamp'].dt.day == 1) & (extracted_data['timestamp'].dt.month == 1) & (extracted_data['timestamp'].dt.year == 2025)
            extracted_data = extracted_data[~exclude].copy()
            all_data.append(extracted_data)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
        
    if not all_data:
        print("No GIOS data loaded.")
        return pd.DataFrame()
    
    final_df = pd.concat(all_data, axis=0, ignore_index=True).sort_values('timestamp')
    print(final_df)
    return final_df
        
def merge_data(pm2_5_df,pm_10_df,imgw_df):
    df_list = [pm2_5_df, pm_10_df, imgw_df]
    
    df_merged = reduce(lambda left,right: pd.merge(left,right, on='timestamp', how='outer'), df_list)
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
    return df_merged

if __name__ == "__main__":
    imgw_df = fetch_imgw_data(YEARS_FROM, YEARS_TO)
    pm2_5_df = load_gios_data('PM25', GIOS_STATION_CODES, YEARS_FROM, YEARS_TO)
    pm10_df = load_gios_data('PM10', GIOS_STATION_CODES, YEARS_FROM, YEARS_TO)
    merged_df = merge_data(pm2_5_df, pm10_df, imgw_df)
    
    merged_df.to_csv('../data/merged_data.csv', index=False)