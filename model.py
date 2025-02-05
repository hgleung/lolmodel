import pandas as pd
from scrape_match_history import get_match_history, calculate_averages

class Model:
    def __init__(self):
        """Initialize the model by loading data and training the random forest"""
        self.players = self.load_player_data()
        self.teams = self.load_team_data()
        self.indexmatch = pd.read_csv('data/index_match.csv', keep_default_na=False)
    
    def load_player_data(self):
        """Load and preprocess player statistics"""
        selected_columns = [
            'player',
            'games',
            'win_rate',
            'kda',
            'avg_kills',
            'avg_deaths',
            'avg_assists',
            'csm',
            'kp%'   # This is KP% in the original data
        ]
        return pd.read_csv('data/player_stats.csv')[selected_columns]

    def load_team_data(self):
        """Load and preprocess team statistics"""
        return pd.read_csv('data/team_stats.csv')
    
    def _get_team_stats(self, team_name, default_stats=None):
        """Helper to get team stats with defaults"""
        if default_stats is None:
            default_stats = {
                'kills_/_game': 0,
                'deaths_/_game': 0,
                'game_duration': '30:00'
            }
            
        team_stats = self.teams[self.teams['name'] == team_name]
        if len(team_stats) == 0:
            print(f"Warning: Team {team_name} not found in team statistics, using defaults")
            return default_stats
        return team_stats.iloc[0]
    
    def _parse_game_time(self, time_str):
        """Convert MM:SS format to minutes as float"""
        minutes, seconds = map(float, time_str.split(':'))
        return minutes + seconds/60
    
    def get_team_name_from_code(self, team_code):
        """Get the full team name from team code"""
        # Case-insensitive team code lookup
        team_data = self.indexmatch[self.indexmatch['Team'].str.upper() == team_code.upper()]
        if team_data.empty:
            raise ValueError(f"Team code {team_code} not found")
        return team_data['Team GOL ID'].iloc[0]

    def _get_default_performance(self):
        """Get default performance stats when data is missing"""
        wins = {f'last_{n}_win_avg': 0 for n in [3, 5, 7, 9]}
        losses = {f'last_{n}_loss_avg': 0 for n in [3, 5, 7, 9]}
        return {
            'wins': wins,
            'losses': losses,
            'split_avg': 0
        }
    
    def analyze_player(self, player_name):
        """
        Analyze a player's statistics including team data and recent performance
        
        Args:
            player_name (str): Name of the player to analyze
            
        Returns:
            dict: Dictionary containing player and team statistics
        """
        result = {}
        
        # Find player in the index (case-insensitive)
        player_info = self.indexmatch[self.indexmatch['PP ID'].str.upper() == player_name.upper()]
        if len(player_info) == 0:
            raise ValueError(f"Player {player_name} not found in index")
        
        # Get player's team and leaguepedia ID
        result['team_name'] = player_info['Team GOL ID'].iloc[0]
        leaguepedia_id = player_info['Leaguepedia ID'].iloc[0]
        
        # Try to get player stats using Leaguepedia ID first (case-insensitive)
        player_stats = self.players[self.players['player'].str.upper() == leaguepedia_id.upper()]
        
        # If not found, try using PP ID as fallback (case-insensitive)
        if len(player_stats) == 0:
            player_stats = self.players[self.players['player'].str.upper() == player_name.upper()]
        
        if len(player_stats) == 0:
            print(f"Warning: Player {player_name} not found in player statistics using either ID, using defaults")
            result['player_stats'] = {
                'games': 0,
                'win_rate': 0.5,
                'kda': 0,
                'avg_kills': 0,
                'avg_deaths': 0,
                'avg_assists': 0,
                'csm': 0,
                'kp%': 0
            }
        else:
            result['player_stats'] = player_stats.iloc[0].to_dict()
        
        # Get match history and calculate averages
        try:
            match_history = get_match_history(leaguepedia_id)
            if match_history is not None and len(match_history) > 0:
                # Calculate averages for wins and losses
                win_averages = calculate_averages(match_history, 'Win')
                loss_averages = calculate_averages(match_history, 'Loss')
                
                result['recent_performance'] = {
                    'wins': win_averages,
                    'losses': loss_averages,
                    'split_avg': result['player_stats']['avg_kills']  # Use avg_kills from player_stats
                }
            else:
                print(f"Warning: No match history found for {player_name}, using defaults")
                result['recent_performance'] = self._get_default_performance()
                result['recent_performance']['split_avg'] = result['player_stats']['avg_kills']  # Use avg_kills from player_stats
        except Exception as e:
            print(f"Warning: Error calculating match history averages: {str(e)}, using defaults")
            result['recent_performance'] = self._get_default_performance()
            result['recent_performance']['split_avg'] = result['player_stats']['avg_kills']  # Use avg_kills from player_stats
        
        return result

    def _calculate_features(self, player_name, player_team, opponent_team):
        """Calculate features for prediction"""
        features = {}
        
        # Get player analysis
        player_analysis = self.analyze_player(player_name)
        
        # Get win/loss averages
        recent_perf = player_analysis['recent_performance']
        for n in [3, 5, 7, 9]:
            features[f'player_{n}_game_win_avg'] = recent_perf['wins'][f'last_{n}_win_avg']
            features[f'player_{n}_game_loss_avg'] = recent_perf['losses'][f'last_{n}_loss_avg']
        features['player_split_avg'] = recent_perf['split_avg']
        
        # Get team stats
        team_stats = self._get_team_stats(player_team)
        features['team_kills_avg'] = team_stats['kills_/_game']
        features['team_avg_game_time'] = self._parse_game_time(team_stats['game_duration'])
        
        # Get opponent stats
        opp_stats = self._get_team_stats(opponent_team)
        features['opp_deaths_avg'] = opp_stats['deaths_/_game']
        features['opp_avg_game_time'] = self._parse_game_time(opp_stats['game_duration'])
        
        # Calculate derived features
        features['team_kills_per_min'] = features['team_kills_avg'] / features['team_avg_game_time']
        features['opp_deaths_per_min'] = features['opp_deaths_avg'] / features['opp_avg_game_time']
        features['k_multi'] = features['opp_deaths_per_min'] / features['team_kills_per_min'] if features['team_kills_per_min'] > 0 else 1
        
        return features

    def calculate_prediction_features(self, player_name, opponent_team_code):
        """Calculate all features needed for prediction"""
        # Get team names
        opponent_team = self.get_team_name_from_code(opponent_team_code)
        
        # Get player's current team
        player_data = self.indexmatch[self.indexmatch['PP ID'].str.upper() == player_name.upper()]
        if player_data.empty:
            raise ValueError(f"Player {player_name} not found")
        
        player_team = player_data['Team GOL ID'].iloc[0]
        
        # Calculate features
        features = self._calculate_features(player_name, player_team, opponent_team)
        return features

    def prediction(self, features, win_chance):
        """
        Calculate preliminary prediction using weighted averages of player performance
        
        Args:
            features (dict): Dictionary of calculated features
            win_chance (float): Probability of winning (0-1)
            
        Returns:
            float: Predicted kills for the player
        """
        # Basic prediction using split average
        basic_prediction = (
            win_chance * 1.25 * features['player_split_avg'] +
            (1 - win_chance) * 0.75 * features['player_split_avg']
        )
        
        # Adjust k_multi for predictions
        k_multi_adjustment = (features['k_multi'] - 1) / 5 + 1
        
        # Calculate weighted win prediction
        win_prediction = (
            features['player_3_game_win_avg'] * 1 +
            features['player_5_game_win_avg'] * 2 +
            features['player_7_game_win_avg'] * 3 +
            features['player_9_game_win_avg'] * 4
        ) / 10
        win_prediction *= k_multi_adjustment
        
        # Calculate weighted loss prediction
        loss_prediction = (
            features['player_3_game_loss_avg'] * 1 +
            features['player_5_game_loss_avg'] * 2 +
            features['player_7_game_loss_avg'] * 3 +
            features['player_9_game_loss_avg'] * 4
        ) / 10
        loss_prediction *= k_multi_adjustment
        
        # Combine predictions based on win chance
        advanced_prediction = win_chance * win_prediction + (1 - win_chance) * loss_prediction
        
        # Average basic and advanced predictions
        final_prediction = (basic_prediction + advanced_prediction) / 2
        
        return max(0, final_prediction)


if __name__ == "__main__":
    # Load the data
    model = Model()    
    print("\nLeague Prediction Model")
    print("Enter input as: <player_name> <opponent_team_code> <win_chance>")
    print("Example: Ruler T1 0.7")
    print("Type 'q' to exit")
    
    while True:
        try:
            # Get input
            user_input = input("\nEnter prediction parameters: ").strip()
            
            # Check for quit command
            if user_input.lower() == 'q':
                break
            
            # Parse input
            try:
                player_name, opponent_team_code, win_chance = user_input.split()
                win_chance = float(win_chance)
                
                # Validate win chance
                if not 0 <= win_chance <= 1:
                    print("Error: Win chance must be between 0 and 1")
                    continue
                
            except ValueError:
                print("Error: Please provide input in the format: player_name opponent_team_code win_chance")
                continue
            
            # Calculate prediction
            try:
                features = model.calculate_prediction_features(player_name, opponent_team_code)
                baseline_pred = model.prediction(features, win_chance)
                
                # Get team names for display
                player_team_code = model.indexmatch[model.indexmatch['PP ID'].str.upper() == player_name.upper()]['Team'].iloc[0]
                player_team_name = model.get_team_name_from_code(player_team_code)
                opponent_team_name = model.get_team_name_from_code(opponent_team_code)
                
                print(f"\nPrediction for {player_name} ({player_team_name}) vs {opponent_team_name}")
                print(f"Win chance: {win_chance:.1%}")
                print(f"Baseline prediction: {baseline_pred:.2f} kills")
                
            except Exception as e:
                print(f"Error calculating prediction: {str(e)}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            continue
