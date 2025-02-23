import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(script_dir, 'data')

def get_match_history(leaguepedia_id):
    """
    Scrape match history for a player from lol.fandom.com using BeautifulSoup
    Returns only specific columns: Date, Tournament, P, W/L, Side, Team, Vs, Len, C, Vs, K, D, A, KDA, CS
    """
    url = f'https://lol.fandom.com/wiki/{leaguepedia_id}/Match_History'
    
    # Set up headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    try:
        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the match history table
        table = soup.find('table', {'class': 'wikitable'})
        if not table:
            print(f"No match history found for player: {leaguepedia_id}")
            return None
        
        # Convert table to string for pandas
        table_html = str(table)
        
        # Read the table using pandas
        dfs = pd.read_html(StringIO(table_html))
        df = dfs[0]
        
        # Handle multi-index columns by taking the last level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        
        # Remove any rows that don't contain actual match data (e.g., headers, descriptions)
        df = df[df['Date'].str.match(r'\d{4}-\d{2}-\d{2}', na=False)]
        
        # Drop the first row which contains descriptions
        df = df.iloc[1:]
        
        # Reset index
        df = df.reset_index(drop=True)
        
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
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    # Keep as string if conversion fails
                    pass
        
        return df
        
    except requests.RequestException as e:
        print(f"Error fetching match history for {leaguepedia_id}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing match history for {leaguepedia_id}: {str(e)}")
        return None

def calculate_averages(match_history, result_type='Win', last_n=[3, 5, 7, 9]):
    """
    Calculate averages for the last N games with specified result.
    If there aren't enough games to fill the window, uses all available games.
    Returns a dictionary with averages for each N.
    """
    if match_history is None or match_history.empty:
        return {}
    
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    match_history = match_history.copy()
    
    # Convert date column to datetime if it exists
    if 'date' in match_history.columns:
        match_history['date'] = pd.to_datetime(match_history['date'])
        # Filter for 2025 matches only
        match_history = match_history[match_history['date'].dt.year == 2025]
        
        # If no matches in 2025, return empty dict
        if match_history.empty:
            return {}
    
    # Filter by result type if specified
    if result_type:
        match_history = match_history[match_history['result'] == result_type]
    
    # If no matches found after filtering
    if match_history.empty:
        return {}
    
    # Convert numeric columns if they aren't already
    numeric_columns = ['kills', 'deaths', 'assists', 'cs']
    for col in numeric_columns:
        if col in match_history.columns:
            try:
                match_history.loc[:, col] = pd.to_numeric(match_history[col], errors='coerce').fillna(0)
            except ValueError:
                # If conversion fails, set to 0
                match_history.loc[:, col] = 0
    
    averages = {}
    for n in last_n:
        # Take last n games
        last_games = match_history.head(n)
        
        # Calculate averages for kills, assists, deaths, and cs
        avg_kills = last_games['kills'].mean() if 'kills' in last_games else 0
        avg_assists = last_games['assists'].mean() if 'assists' in last_games else 0
        avg_deaths = last_games['deaths'].mean() if 'deaths' in last_games else 0
        avg_cs = last_games['cs'].mean() if 'cs' in last_games else 0
        
        # Store with the correct key format
        base_key = f'last_{n}_{"win" if result_type == "Win" else "loss"}_avg'
        averages[base_key] = avg_kills  # Keep original kills average
        averages[f'{base_key}_assists'] = avg_assists  # Add assists average
        averages[f'{base_key}_deaths'] = avg_deaths  # Add deaths average
        averages[f'{base_key}_cs'] = avg_cs  # Add CS average
    
    return averages
