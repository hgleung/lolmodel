import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import StringIO

def get_match_history(leaguepedia_id):
    """
    Scrape match history for a player from lol.fandom.com using BeautifulSoup
    Returns only specific columns: Date, Tournament, P, W/L, Side, Team, Vs, Len, C, Vs, K, D, A, KDA, CS
    """
    url = f'https://lol.fandom.com/wiki/{leaguepedia_id}/Match_History'
    
    # Set up headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the match history table
        table = soup.find('table', {'class': 'wikitable'})
        if not table:
            raise ValueError("Match history table not found")
        
        # Convert table to string for pandas
        table_html = str(table)
        
        # Read the table using pandas
        dfs = pd.read_html(StringIO(table_html))
        df = dfs[0]
        
        # Handle multi-index columns by taking the last level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        
        # Drop the first row which contains descriptions
        df = df.iloc[1:]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Print current columns for debugging
        print("\nCurrent columns:", df.columns.tolist())
        
        # Map the columns we want
        column_mapping = {
            'Date': 'date',
            'Tournament': 'tournament',
            'P': 'patch',
            'W/L': 'result',
            'Side': 'side',
            'Team': 'team',
            'Vs': 'opponent',
            'Len': 'game_length',
            'C': 'champion',
            'K': 'kills',
            'D': 'deaths',
            'A': 'assists',
            'KDA': 'kda',
            'CS': 'cs'
        }
        
        # Rename columns that exist
        new_columns = {}
        for col in df.columns:
            if col in column_mapping:
                new_columns[col] = column_mapping[col]
        
        # Rename only the columns that we mapped
        df = df.rename(columns=new_columns)
        
        # Keep only the columns we want
        keep_columns = list(new_columns.values())
        df = df[keep_columns]
        
        # Clean up the data
        # Convert W/L to Win/Loss
        if 'result' in df.columns:
            df['result'] = df['result'].apply(lambda x: 'Win' if str(x).startswith('W') else 'Loss' if str(x).startswith('L') else 'NA')
        
        # Convert K/D/A to numeric
        numeric_columns = ['kills', 'deaths', 'assists']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch match history: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing match history: {str(e)}")

def calculate_averages(match_history, result_type='Win', last_n=[3, 5, 7, 9]):
    """
    Calculate averages for the last N games with specified result
    Returns a dictionary with averages for each N
    """
    # Filter for the specified result
    filtered_df = match_history[match_history['result'] == result_type]
    
    averages = {}
    for n in last_n:
        if len(filtered_df) >= n:
            last_n_games = filtered_df.head(n)
            avg_kills = last_n_games['kills'].astype(float).mean()
            averages[f'last_{n}_{result_type.lower()}s_avg'] = avg_kills
        else:
            averages[f'last_{n}_{result_type.lower()}s_avg'] = 'NA'
    
    return averages
