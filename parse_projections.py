# https://api.prizepicks.com/projections?league_id=121

import json
import os

# Get the absolute path of the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

def split_display_name(display_name):
    """Split display name into individual players, handling '+' separator and whitespace"""
    if not display_name:
        return None, None, None
    
    players = [p.strip() for p in display_name.split('+')]
    # Pad with None to ensure we have exactly 3 elements
    players.extend([None] * (3 - len(players)))
    return players[0], players[1], players[2]

def parse_stat_type(stat_type_str):
    """Parse stat_type string to extract stat type and map count"""
    if not stat_type_str:
        return None, None
    
    # Initialize defaults
    stat_type = None
    map_count = 1  # Default to 1 if not specified
    
    # Check for stat type
    if "Kills" in stat_type_str:
        stat_type = "Kills"
    elif "Assists" in stat_type_str:
        stat_type = "Assists"
    elif "Deaths" in stat_type_str:
        stat_type = "Deaths"
    
    # Check for map count
    if "MAPS 1-3" in stat_type_str:
        map_count = 3
    elif "MAPS 1-2" in stat_type_str:
        map_count = 2
    elif "MAP 1" in stat_type_str:
        map_count = 1
    
    return stat_type, map_count

def parse_projections(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Create a dictionary to store player information from included section
    included_data = {}
    for item in data.get('included', []):
        if item['type'] == 'new_player':
            player_id = item['id']
            attributes = item.get('attributes', {})
            included_data[player_id] = {
                'display_name': attributes.get('display_name'),
                'team': attributes.get('team'),
                'combo': attributes.get('combo', False)  # Get combo from included section
            }
    
    # Create a dictionary to map game IDs to teams and opponents
    game_teams = {}
    
    # First pass: collect all teams for each game ID
    for entry in data.get('data', []):
        attributes = entry.get('attributes', {})
        player_id = entry.get('relationships', {}).get('new_player', {}).get('data', {}).get('id')
        game_id = attributes.get('game_id')
        if game_id and player_id:
            team = included_data.get(player_id, {}).get('team')
            if game_id not in game_teams:
                game_teams[game_id] = set()
            if team:
                game_teams[game_id].add(team)
    
    # Process the main data section and match with included data
    parsed_entries = []
    for entry in data.get('data', []):
        entry_id = entry['id']
        attributes = entry.get('attributes', {})
        
        # Get the player ID from relationships
        player_id = entry.get('relationships', {}).get('new_player', {}).get('data', {}).get('id')
        
        # Get player info from included data
        player_info = included_data.get(player_id, {})
        
        # Get team and game_id
        team = player_info.get('team')
        game_id = attributes.get('game_id')
        
        # Find opponent
        opponent = None
        if game_id and team and game_id in game_teams:
            teams = game_teams[game_id]
            opponents = [t for t in teams if t != team]
            if opponents:
                opponent = opponents[0]  # Take the first team that isn't the player's team
        
        # Handle player names based on whether it's a combo
        is_combo = player_info.get('combo', False)
        if is_combo:
            player_1, player_2, player_3 = split_display_name(player_info.get('display_name'))
        else:
            player_1 = player_info.get('display_name')
            player_2 = None
            player_3 = None
        
        # Parse stat type and map count
        stat_type, map_count = parse_stat_type(attributes.get('stat_type'))
        
        parsed_entry = {
            'player_1': player_1,
            'player_2': player_2,
            'player_3': player_3,
            'team': team,
            'combo': is_combo,
            'opponent': opponent,
            'line': float(attributes.get('line_score')),
            'stat_type': stat_type,
            'map_count': map_count
        }
        parsed_entries.append(parsed_entry)
    
    return parsed_entries

if __name__ == "__main__":
    # Example usage
    file_path = os.path.join(DATA_DIR, 'projections.json')
    try:
        results = parse_projections(file_path)
        print(f"Successfully parsed {len(results)} entries")
        
        # Filter and display combo entries
        # combo_entries = [entry for entry in results if entry['combo']]
        # print(f"\nFound {len(combo_entries)} combo entries")
        
        # print("\nFirst 5 combo entries:")
        print(f"\nEntry #1:")
        for key, value in results[0].items():
            print(f"{key}: {value}")
        print("-" * 50)

        # print(f"\nCombo Entry #1:")
        # for key, value in combo_entries[0].items():
        #     print(f"{key}: {value}")
        # print("-" * 50)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON file")
    except Exception as e:
        print(f"An error occurred: {e}")
