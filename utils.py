import os
import subprocess
import time

def update_stats():
    """Update both player and team statistics"""
    start_time = time.time()
    
    print("\nUpdating player and team statistics...")
    
    # Run both scraping scripts
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\nScraping player stats...")
    subprocess.run(['python3', os.path.join(script_dir, 'scrape_players.py')], check=True)
    
    print("\nScraping team stats...")
    subprocess.run(['python3', os.path.join(script_dir, 'scrape_teams.py')], check=True)
    
    elapsed_time = time.time() - start_time
    print(f"\nStats update complete! Time taken: {elapsed_time:.2f} seconds")
