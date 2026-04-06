import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def analyze_weather_descriptions():
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'historical-hourly-weather-data', 'weather_description.csv')
    
    # Read CSV
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop datetime column
    df_desc = df.drop(columns=['datetime'])
    
    # Flatten all city columns into a single series
    # This collects all descriptions from all cities across all times
    all_descriptions = df_desc.values.flatten()
    
    # Convert to string and handle NaNs
    all_descriptions = [str(d).lower().strip() for d in all_descriptions if pd.notna(d)]
    
    # Get unique descriptions
    unique_descriptions = sorted(list(set(all_descriptions)))
    
    print("\n" + "="*50)
    print("           WEATHER DESCRIPTION ANALYSIS           ")
    print("="*50)
    print(f"Total entries processed: {len(all_descriptions)}")
    print(f"Number of unique descriptions: {len(unique_descriptions)}")
    print("-" * 50)
    print("Unique Descriptions List:")
    for i, desc in enumerate(unique_descriptions):
        print(f"{i+1:02d}. {desc}")
    print("="*50)

if __name__ == "__main__":
    analyze_weather_descriptions()
