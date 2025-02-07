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
    
    # The team stats table is usually the first table
    df = tables[0]
    
    # Clean up column names - convert to strings first
    df.columns = [str(col).strip().replace(' ', '_').lower() for col in df.columns]
    
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
                    df[col] = df[col].replace('-', '0')  # Replace '-' with '0'
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except ValueError:
                        # Keep as string if conversion fails
                        pass
                except:
                    pass
    
    # Save to CSV
    output_file = os.path.join(data_dir, 'team_stats.csv')
    df.to_csv(output_file, index=False)
    
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
    for i, line in enumerate(content):
        if '- Last scraped:' in line and 'team_stats.csv' in content[i-1]:
            content[i] = f'- Last scraped: {current_time}\n'
            break
    
    # Write the updated content back to README
    with open(readme_path, 'w') as f:
        f.writelines(content)

except Exception as e:
    print(f"Error scraping team stats: {str(e)}")
    raise
