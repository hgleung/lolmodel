import pandas as pd
import numpy as np
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from io import StringIO
import re

# URL of the player statistics
url = 'https://gol.gg/players/list/season-S15/split-Winter/tournament-ALL/'

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run in headless mode
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# Initialize the Chrome WebDriver
driver = webdriver.Chrome(options=chrome_options)

try:
    # Navigate to the URL
    driver.get(url)
    
    # Wait for the table to load
    wait = WebDriverWait(driver, 10)
    table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'table_list')))
    
    # Get the table HTML
    table_html = table.get_attribute('outerHTML')
    
    # Read the table using pandas
    df = pd.read_html(StringIO(table_html))[0]
    
    # Clean the column names - remove any special characters and spaces
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    # Convert percentage strings to floats
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert percentage values
            if df[col].str.contains('%').any():
                df[col] = df[col].replace('-', '0%')  # Replace '-' with '0%'
                df[col] = df[col].str.rstrip('%').astype('float') / 100.0
            
            # Try to convert KDA and other numeric values
            else:
                try:
                    df[col] = df[col].replace('-', '0')  # Replace '-' with '0'
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
    
    # Save to CSV
    df.to_csv('data/player_stats.csv', index=False)
    
    # Update README with timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')
    readme_path = 'README.md'
    
    with open(readme_path, 'r') as f:
        content = f.readlines()
    
    # Find and update the player statistics timestamp
    for i, line in enumerate(content):
        if '- Last scraped:' in line and 'player_stats.csv' in content[i-1]:
            content[i] = f'- Last scraped: {current_time}\n'
            break
    
    # Write the updated content back to README
    with open(readme_path, 'w') as f:
        f.writelines(content)
    
    print("Data has been scraped and saved to data/player_stats.csv")
    print(f"README.md has been updated with timestamp: {current_time}")
    print(f"\nShape of the dataset: {df.shape}")
    print("\nFirst few rows of the data:")
    print(df.head())
    print("\nColumns in the dataset:")
    print(df.columns.tolist())

finally:
    # Close the browser
    driver.quit()
