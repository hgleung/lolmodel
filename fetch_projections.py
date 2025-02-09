import requests
import json
import os
from datetime import datetime
import cloudscraper
import time
import random

def fetch_projections(league_id=121):
    url = f'https://api.prizepicks.com/projections?league_id={league_id}'
    
    # Create a cloudscraper instance
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        }
    )
    
    # Add headers that look like a real browser
    headers = {
        'Connection': 'keep-alive',
        'Accept': 'application/json; charset=UTF-8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Access-Control-Allow-Credentials': 'true',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Referer': 'https://app.prizepicks.com/',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    # Add random delay between requests to avoid rate limiting
    time.sleep(random.uniform(1, 3))

    try:
        response = scraper.get(url, headers=headers)
        response.raise_for_status()
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save the response to projections.json
        with open('data/projections.json', 'w') as f:
            json.dump(response.json(), f, indent=2)
            
        print(f"Successfully updated projections.json at {datetime.now()}")
        return True
        
    except (cloudscraper.exceptions.CloudflareChallengeError, requests.exceptions.RequestException) as e:
        print(f"Error fetching data: {e}")
        return False

if __name__ == "__main__":
    fetch_projections()
