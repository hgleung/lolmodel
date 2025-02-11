import pandas as pd
import os
from datetime import datetime
import requests
from io import StringIO

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# URL of the team statistics
url = 'https://gol.gg/teams/list/season-S15/split-Winter/tournament-ALL/'

# Set up headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

try:
    # Get the webpage content
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    # Parse the HTML tables using pandas
    html_io = StringIO(response.text)
    tables = pd.read_html(html_io)
    
    # Find the team stats table - it should have specific columns
    team_table = None
    required_columns = {'Team', 'Games', 'Win rate', 'Avg. game duration', 'Kills per game', 'Deaths per game'}
    
    for table in tables:
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = table.columns.get_level_values(-1)
        table_cols = set(str(col) for col in table.columns)
        if any(required_columns.intersection(table_cols)):
            team_table = table
            break
    
    if team_table is None:
        raise ValueError("Could not find team stats table with required columns")
    
    df = team_table
    
    # Clean up column names - convert to strings first
    df.columns = [str(col).strip().replace(' ', '_').lower() for col in df.columns]
    
    # Print column names for debugging
    # print("Columns in new data:", df.columns.tolist())
    
    # Convert percentage strings to floats
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert column to string first to handle any numeric values
            df[col] = df[col].astype(str)
            
            # Try to convert percentage values
            if df[col].str.contains('%').any():
                df[col] = df[col].replace('-', '0%')  # Replace '-' with '0%'
                df[col] = df[col].str.rstrip('%').astype('float') / 100.0
            
            # Try to convert other numeric values
            else:
                try:
                    # Only convert non-NA values
                    mask = (df[col] != 'NA')
                    df.loc[mask, col] = df.loc[mask, col].replace('-', '0')
                    try:
                        df.loc[mask, col] = pd.to_numeric(df.loc[mask, col])
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
                except:
                    pass
    
    # Load existing data if it exists
    output_file = os.path.join(data_dir, 'team_stats.csv')
    old_df = pd.DataFrame()
    if os.path.exists(output_file):
        old_df = pd.read_csv(output_file)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Compare old and new data
    if not old_df.empty:
        # Get common columns
        common_cols = list(set(old_df.columns) & set(df.columns))
        
        # Count new and removed teams
        old_teams = set(old_df['name'])
        new_teams = set(df['name'])
        num_added = len(new_teams - old_teams)
        num_removed = len(old_teams - new_teams)
        
        # Find changed rows by comparing values in common teams
        merged_df = pd.merge(old_df[common_cols], df[common_cols], on='name', how='inner', suffixes=('_old', '_new'))
        num_changed = 0
        
        # Compare values with appropriate handling for different types
        for col in common_cols:
            if col != 'name':  # Skip the team name column since it's our merge key
                old_col = f"{col}_old"
                new_col = f"{col}_new"
                
                # Convert to numeric if possible and round to handle floating point differences
                try:
                    old_values = pd.to_numeric(merged_df[old_col]).round(3)
                    new_values = pd.to_numeric(merged_df[new_col]).round(3)
                except:
                    # For non-numeric columns, convert to string and strip whitespace
                    old_values = merged_df[old_col].astype(str).str.strip()
                    new_values = merged_df[new_col].astype(str).str.strip()
                
                num_changed += (old_values != new_values).sum()
        
        if num_changed > 0:
            print(f"Updated team stats: {num_changed} values changed across {len(merged_df)} teams")
            print(f"Teams added: {num_added}, removed: {num_removed}")
        else:
            print("No changes to team stats")
    else:
        print(f"Created new team stats with {len(df)} rows")
    
    # Update README timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    readme_path = os.path.join(script_dir, 'README.md')
    
    try:
        with open(readme_path, 'r') as f:
            content = f.readlines()
    except FileNotFoundError:
        content = [
            '# League Model Data\n',
            '\n',
            '## Data Files\n',
            '- team_stats.csv\n',
            f'- Last scraped: {current_time}\n'
        ]
    
    # Find and update the team statistics timestamp
    timestamp_found = False
    for i, line in enumerate(content):
        if '- Last scraped:' in line and 'team_stats.csv' in content[i-1]:
            content[i] = f'- Last scraped: {current_time}\n'
            timestamp_found = True
            break
    
    if not timestamp_found:
        content.extend([
            '- team_stats.csv\n',
            f'- Last scraped: {current_time}\n'
        ])
    
    # Write the updated content back to README
    with open(readme_path, 'w') as f:
        f.writelines(content)

except Exception as e:
    print(f"Error scraping team stats: {str(e)}")
    raise
