import pandas as pd
import numpy as np

def load_data():
    # Read the CSV file
    df = pd.read_csv('data/player_stats.csv')
    
    # Select the specified columns
    selected_columns = [
        'player',
        'games',
        'win_rate',
        'kda',
        'avg_kills',
        'avg_deaths',
        'avg_assists',
        'csm',  # This is CSPM in the original data
        'kp%'   # This is KP% in the original data
    ]
        
    # Create a new dataframe with only the selected columns
    model_df = df[selected_columns]
    
    return model_df

if __name__ == "__main__":
    # Load the data
    model_df = load_data()
    
    # Display basic information about the dataset
    print("Dataset Shape:", model_df.shape)
    print("\nFirst few rows:")
    print(model_df.head())
    print("\nDataset Description:")
    print(model_df.describe())
    
    # Display any missing values
    print("\nMissing Values:")
    print(model_df.isnull().sum())
