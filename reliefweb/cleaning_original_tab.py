import pandas as pd
import os

def process_cyclone_data(input_file, output_file):
    """
    Lit le fichier IBTrACS/ERA5, nettoie les données, filtre pour 2022-2023
    et renomme les colonnes pour les rendre compréhensibles.
    """
    df = pd.read_csv(input_file)
    
    # Nettoyage des noms et identifiants (enlever b' et ')
    print("Nettoyage des noms et identifiants...")
    cols_to_clean = ['Storm_ID', 'Storm_Name']
    for col in cols_to_clean:
        if col in df.columns:
            # Nettoyer b'...' et aussi gérer les espaces
            df[col] = df[col].astype(str).str.replace(r"^b'|'$", "", regex=True).str.strip()

    # Filtrage des années 2022 et 2023
    print("Filtrage des années 2022 et 2023...")
    if 'Year' in df.columns:
        df_reduced = df[df['Year'].isin([2022, 2023])].copy()
        print(f"Nombre de lignes après filtrage : {len(df_reduced)}")
    else:
        print("Erreur : Colonne 'Year' manquante.")
        return

    # Dictionnaire de renommage
    column_mapping = {
        'sid': 'Storm_ID',
        'name': 'Storm_Name',
        'basin': 'Ocean_Basin',
        'season': 'Year',
        'time_stamp': 'Timestamp',
        'lat': 'Latitude',
        'lon': 'Longitude',
        'wind': 'Observed_Wind_Max_Knots',      
        'pressure': 'Observed_Pressure_Min_mb', 
        'storm_speed': 'Storm_Speed_Knots',
        'storm_dir': 'Storm_Direction_Deg',
        '2m_temperature': 'ERA5_Temp_2m_Kelvin',
        'mean_sea_level_pressure_hpa': 'ERA5_Pressure_MSL_hPa',
        '10m_u_component_of_wind': 'ERA5_Wind_U_Component',
        '10m_v_component_of_wind': 'ERA5_Wind_V_Component',
        'era5_spatial_error_km': 'ERA5_Position_Error_km'
    }

    # Application du renommage
    print("Renommage des colonnes...")
    df_reduced = df_reduced.rename(columns=column_mapping)

    # Conversion de la date en format datetime
    if 'Timestamp' in df_reduced.columns:
        df_reduced['Timestamp'] = pd.to_datetime(df_reduced['Timestamp'])

    # Afficher quelques exemples de noms nettoyés
    if 'Storm_Name' in df_reduced.columns:
        print("\nExemples de noms de tempêtes après nettoyage :")
        print(df_reduced['Storm_Name'].unique()[:10])

    # Sauvegarde
    print(f"\nSauvegarde des données ({len(df_reduced)} lignes)...")
    df_reduced.to_csv(output_file, index=False)
    print(f"Succès ! Fichier créé : {output_file}")

if __name__ == "__main__":
    INPUT_FILENAME = "C:\\Users\\user\\OneDrive\\Documents\\IASD\\data extraction\\Projet\\multimodal_natural_disaster_dataset\\data\\processed\\ibtracs_era5_20251218_1520.csv"
    OUTPUT_FILENAME = "C:\\Users\\user\\OneDrive\\Documents\\IASD\\data extraction\\Projet\\multimodal_natural_disaster_dataset\\data\\processed\\ibtracs_era5_20251218_1520.csv"
    
    process_cyclone_data(INPUT_FILENAME, OUTPUT_FILENAME)