import pandas as pd
from scrape_match_history import get_match_history, calculate_averages
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import os
from utils import update_stats, calc_odds

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

class Model:
    def __init__(self):
        """Initialize the model by loading necessary data"""
        self.players = self.load_player_data()
        self.teams = self.load_team_data()
        self.indexmatch = pd.read_csv(os.path.join(data_dir, 'index_match.csv'), keep_default_na=False)
        self.rf_model = None
        self.model_path = os.path.join(data_dir, 'rf_model.joblib')
        self.feature_columns = [
            'avg_kills', 'win_rate', 'kda', 'team_kills_avg', 
            'team_kills_per_min', 'player_3_game_win_avg', 
            'player_5_game_win_avg', 'player_3_game_loss_avg', 
            'player_5_game_loss_avg'
        ]
        
        # Try to load pre-trained model
        if os.path.exists(self.model_path):
            try:
                print("Loading pre-trained model...")
                self.rf_model = joblib.load(self.model_path)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                self.rf_model = None

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
        return pd.read_csv(os.path.join(data_dir, 'player_stats.csv'))[selected_columns]

    def load_team_data(self):
        """Load and preprocess team statistics"""
        return pd.read_csv(os.path.join(data_dir, 'team_stats.csv'))

    def _get_team_stats(self, team_name):
        """Helper to get team stats"""

        team_stats = self.teams[self.teams['name'] == team_name]
        if len(team_stats) == 0:
            # print(f"Warning: Team {team_name} not found in team statistics, using defaults")
            return
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
            # print(f"Warning: Player {player_name} not found in player statistics using either ID")
            return
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
                # print(f"Warning: No match history found for {player_name}, using defaults")
                return
        except Exception as e:
            # print(f"Warning: Error calculating match history averages: {str(e)}, using defaults")
            return
        
        return result

    def _calculate_features(self, player_name, player_team, opponent_team):
        """Calculate features for prediction"""
        features = {}

        # Get player analysis
        player_analysis = self.analyze_player(player_name)
        if not player_analysis:
            raise ValueError(f"Could not analyze player {player_name}")

        # Get win/loss averages with default values if missing
        recent_perf = player_analysis.get('recent_performance', {})
        wins = recent_perf.get('wins', {})
        losses = recent_perf.get('losses', {})

        # Set win/loss averages with defaults
        for n in [3, 5, 7, 9]:
            win_key = f'last_{n}_win_avg'
            loss_key = f'last_{n}_loss_avg'
            features[f'player_{n}_game_win_avg'] = wins.get(win_key, 0)
            features[f'player_{n}_game_loss_avg'] = losses.get(loss_key, 0)
            features[f'player_{n}_game_win_assists_avg'] = wins.get(f'{win_key}_assists', 0)
            features[f'player_{n}_game_loss_assists_avg'] = losses.get(f'{loss_key}_assists', 0)
            features[f'player_{n}_game_win_deaths_avg'] = wins.get(f'{win_key}_deaths', 0)
            features[f'player_{n}_game_loss_deaths_avg'] = losses.get(f'{loss_key}_deaths', 0)
            features[f'player_{n}_game_win_cs_avg'] = wins.get(f'{win_key}_cs', 0)
            features[f'player_{n}_game_loss_cs_avg'] = losses.get(f'{loss_key}_cs', 0)

        # Get split average with default
        features['player_split_avg'] = recent_perf.get('split_avg', 0)
        features['player_split_assists_avg'] = player_analysis['player_stats'].get('avg_assists', 0)
        features['player_split_deaths_avg'] = player_analysis['player_stats'].get('avg_deaths', 0)
        features['player_csm'] = float(player_analysis['player_stats'].get('csm', 0))

        # Get team stats
        team_stats = self._get_team_stats(player_team)
        features['team_kills_avg'] = team_stats.get('kills_/_game', 0)
        features['team_deaths_avg'] = team_stats.get('deaths_/_game', 0)
        features['team_avg_game_time'] = self._parse_game_time(team_stats.get('game_duration', '00:00'))

        # Get opponent stats
        if opponent_team is not None:
            opp_stats = self._get_team_stats(opponent_team)
            features['opp_deaths_avg'] = opp_stats.get('deaths_/_game', 0)
            features['opp_kills_avg'] = opp_stats.get('kills_/_game', 0)
            features['opp_avg_game_time'] = self._parse_game_time(opp_stats.get('game_duration', '00:00'))

            # Calculate average game time between team and opponent
            features['avg_game_time'] = (features['team_avg_game_time'] + features['opp_avg_game_time']) / 2

        # Calculate derived features
        features['team_kills_per_min'] = features['team_kills_avg'] / features['team_avg_game_time'] if features['team_avg_game_time'] > 0 else 0
        features['team_deaths_per_min'] = features['team_deaths_avg'] / features['team_avg_game_time'] if features['team_avg_game_time'] > 0 else 0
        if opponent_team is not None:
            features['opp_deaths_per_min'] = features['opp_deaths_avg'] / features['opp_avg_game_time'] if features['opp_avg_game_time'] > 0 else 0
            features['opp_kills_per_min'] = features['opp_kills_avg'] / features['opp_avg_game_time'] if features['opp_avg_game_time'] > 0 else 0
            features['k_multi'] = features['opp_deaths_per_min'] / features['team_kills_per_min'] if features['team_kills_per_min'] > 0 else 1
            features['d_multi'] = features['team_deaths_per_min'] / features['opp_kills_per_min'] if features['opp_kills_per_min'] > 0 else 1

        return features

    def prediction(self, features, win_chance, num_games=1):
        """
        Calculate preliminary prediction using weighted averages of player performance

        Args:
            features (dict): Dictionary of calculated features
            win_chance (float): Probability of winning (0-1)
            num_games (int): Number of games to predict for

        Returns:
            dict: Dictionary containing predicted kills, assists, deaths, cs, and fantasy score
        """
        # Basic kill prediction using split average
        basic_kill_prediction = (
            win_chance * 1.25 * features['player_split_avg'] +
            (1 - win_chance) * 0.75 * features['player_split_avg']
        )

        # Basic assist prediction using split average
        basic_assist_prediction = (
            win_chance * 1.25 * features['player_split_assists_avg'] +
            (1 - win_chance) * 0.75 * features['player_split_assists_avg']
        )

        # Basic death prediction using split average
        basic_death_prediction = (
            win_chance * 0.75 * features['player_split_deaths_avg'] +
            (1 - win_chance) * 1.25 * features['player_split_deaths_avg']
        )

        # Adjust k_multi and d_multi for predictions
        k_multi_adjustment = (features['k_multi'] - 1) / 5 + 1
        d_multi_adjustment = (features['d_multi'] - 1) / 5 + 1

        # Calculate weighted win predictions
        win_kill_prediction = (
            features['player_3_game_win_avg'] * 1 +
            features['player_5_game_win_avg'] * 2 +
            features['player_7_game_win_avg'] * 3 +
            features['player_9_game_win_avg'] * 4
        ) / 10
        win_kill_prediction *= k_multi_adjustment

        win_assist_prediction = (
            features['player_3_game_win_assists_avg'] * 1 +
            features['player_5_game_win_assists_avg'] * 2 +
            features['player_7_game_win_assists_avg'] * 3 +
            features['player_9_game_win_assists_avg'] * 4
        ) / 10
        win_assist_prediction *= k_multi_adjustment

        win_death_prediction = (
            features['player_3_game_win_deaths_avg'] * 1 +
            features['player_5_game_win_deaths_avg'] * 2 +
            features['player_7_game_win_deaths_avg'] * 3 +
            features['player_9_game_win_deaths_avg'] * 4
        ) / 10
        win_death_prediction *= d_multi_adjustment

        # Calculate weighted loss predictions
        loss_kill_prediction = (
            features['player_3_game_loss_avg'] * 1 +
            features['player_5_game_loss_avg'] * 2 +
            features['player_7_game_loss_avg'] * 3 +
            features['player_9_game_loss_avg'] * 4
        ) / 10
        loss_kill_prediction *= k_multi_adjustment

        loss_assist_prediction = (
            features['player_3_game_loss_assists_avg'] * 1 +
            features['player_5_game_loss_assists_avg'] * 2 +
            features['player_7_game_loss_assists_avg'] * 3 +
            features['player_9_game_loss_assists_avg'] * 4
        ) / 10
        loss_assist_prediction *= k_multi_adjustment

        loss_death_prediction = (
            features['player_3_game_loss_deaths_avg'] * 1 +
            features['player_5_game_loss_deaths_avg'] * 2 +
            features['player_7_game_loss_deaths_avg'] * 3 +
            features['player_9_game_loss_deaths_avg'] * 4
        ) / 10
        loss_death_prediction *= d_multi_adjustment

        # Combine predictions based on win chance
        advanced_kill_prediction = win_chance * win_kill_prediction + (1 - win_chance) * loss_kill_prediction
        advanced_assist_prediction = win_chance * win_assist_prediction + (1 - win_chance) * loss_assist_prediction
        advanced_death_prediction = win_chance * win_death_prediction + (1 - win_chance) * loss_death_prediction

        # Average basic and advanced predictions
        final_kill_prediction = (basic_kill_prediction + advanced_kill_prediction) / 2
        final_assist_prediction = (basic_assist_prediction + advanced_assist_prediction) / 2
        final_death_prediction = (basic_death_prediction + advanced_death_prediction) / 2

        # Calculate CS prediction
        predicted_cs = features['player_csm'] * features['avg_game_time']

        # Multiply by number of games
        final_kill_prediction *= num_games
        final_assist_prediction *= num_games
        final_death_prediction *= num_games
        predicted_cs *= num_games

        # Calculate fantasy score
        fantasy_score = (
            3 * final_kill_prediction +
            2 * final_assist_prediction -
            final_death_prediction +
            0.02 * predicted_cs
        )

        return {
            'kills': max(0, final_kill_prediction),
            'assists': max(0, final_assist_prediction),
            'deaths': max(0, final_death_prediction),
            'cs': max(0, predicted_cs),
            'fantasy_score': max(0, fantasy_score)
        }

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

    def train_random_forest(self, force_retrain=False):
        """
        Train a random forest model using player data and match history
        
        Args:
            force_retrain (bool): If True, retrain model even if a saved model exists
        """
        # Check if we already have a trained model
        if not force_retrain and self.rf_model is not None:
            print("Using existing trained model")
            return self.rf_model
        else:
            print("Training random forest model...")

        training_data = []
        total_players = len(self.indexmatch)
        processed = 0
        
        # Collect training data from all players
        for _, row in self.indexmatch.iterrows():
            try:
                player_name = row['PP ID']
                team_name = row['Team GOL ID']
                
                # Update progress
                processed += 1
                print(f"\rProcessing {player_name:<20} ({processed}/{total_players} players)", end='', flush=True)
                
                # Get player analysis using existing method
                player_analysis = self.analyze_player(player_name)
                if not player_analysis:
                    continue
                    
                # Calculate features using existing method
                features = self._calculate_features(player_name, team_name, None)
                if not features:
                    continue
                
                # Extract relevant features
                sample = {
                    'avg_kills': float(player_analysis['player_stats'].get('avg_kills')),
                    'win_rate': float(player_analysis['player_stats'].get('win_rate')),
                    'kda': float(player_analysis['player_stats'].get('kda')),
                    'team_kills_avg': float(features.get('team_kills_avg')),
                    'team_kills_per_min': float(features.get('team_kills_per_min')),
                    'player_3_game_win_avg': float(features.get('player_3_game_win_avg')),
                    'player_5_game_win_avg': float(features.get('player_5_game_win_avg')),
                    'player_3_game_loss_avg': float(features.get('player_3_game_loss_avg')),
                    'player_5_game_loss_avg': float(features.get('player_5_game_loss_avg')),
                    'target_kills': float(player_analysis['player_stats'].get('avg_kills'))
                }
                
                # Ensure all required features are present
                if all(col in sample for col in self.feature_columns):
                    training_data.append(sample)
                
            except Exception as e:
                # print(f"\nError processing player {row['PP ID']}: {str(e)}")
                continue
        
        print()  # New line after progress
        
        if not training_data:
            print("Error: No valid training data collected")
            return None
            
        # Convert to DataFrame and ensure feature order
        df = pd.DataFrame(training_data)
        X = df[self.feature_columns].values  # Convert to numpy array without column names
        y = df['target_kills'].values
        
        # Train model
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)
        print(f"Successfully trained random forest model on {len(training_data)} samples")
        
        # Save model to disk
        try:
            print("Saving model to disk...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.rf_model, self.model_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
        
        return self.rf_model

    def predict_rf(self, player_name, opponent_team_code, win_chance=0.5):
        """Make a prediction using the random forest model"""
        if self.rf_model is None:
            print("Training random forest model first...")
            self.train_random_forest()
            if self.rf_model is None:
                return None
        
        try:
            # Get player analysis
            player_analysis = self.analyze_player(player_name)
            if not player_analysis:
                return None
                
            # Get player's team
            player_data = self.indexmatch[self.indexmatch['PP ID'].str.upper() == player_name.upper()]
            if player_data.empty:
                return None
                
            # Get team names
            try:
                player_team = self.get_team_name_from_code(player_data['Team'].iloc[0])
                opponent_team = self.get_team_name_from_code(opponent_team_code)
            except ValueError as e:
                print(f"Error getting team names: {str(e)}")
                return None
            
            # Calculate features
            features = self._calculate_features(player_name, player_team, opponent_team)
            if not features:
                return None
            
            # Prepare features for prediction using same columns as training
            feature_dict = {
                'avg_kills': float(player_analysis['player_stats'].get('avg_kills')),
                'win_rate': float(player_analysis['player_stats'].get('win_rate')),
                'kda': float(player_analysis['player_stats'].get('kda')),
                'team_kills_avg': float(features.get('team_kills_avg')),
                'team_kills_per_min': float(features.get('team_kills_per_min')),
                'player_3_game_win_avg': float(features.get('player_3_game_win_avg')),
                'player_5_game_win_avg': float(features.get('player_5_game_win_avg')),
                'player_3_game_loss_avg': float(features.get('player_3_game_loss_avg')),
                'player_5_game_loss_avg': float(features.get('player_5_game_loss_avg'))
            }
            
            # Create numpy array with features in correct order
            X = pd.DataFrame([feature_dict])[self.feature_columns].values
            
            # Get predictions from all trees
            predictions = []
            for estimator in self.rf_model.estimators_:
                predictions.append(estimator.predict(X)[0])
            
            # Calculate prediction and confidence
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            confidence = 1 - (std_pred / mean_pred) if mean_pred > 0 else 0
            
            return {
                'prediction': max(0, mean_pred),  # Ensure non-negative prediction
                'confidence': confidence,
                'range': (max(0, mean_pred - 1.96 * std_pred), mean_pred + 1.96 * std_pred)
            }
            
        except Exception as e:
            print(f"Error making RF prediction: {str(e)}")
            return None

    def predict_match(self, team1_code, team2_code, win_chance, num_maps=1):
        """
        Run predictions for all players in a team vs team match
        
        Args:
            team1_code (str): Code for team 1
            team2_code (str): Code for team 2
            win_chance (float): Win probability for team 1 (0-1)
            num_maps (int): Number of maps to predict for
            
        Returns:
            dict: Dictionary containing predictions for all players
        """
        results = {'team1': [], 'team2': []}
        
        try:            
            # Get players for team 1
            team1_players = self.indexmatch[self.indexmatch['Team'].str.upper() == team1_code.upper()]
            for _, player in team1_players.iterrows():
                try:
                    player_name = player['PP ID']
                    baseline_pred = self.prediction(
                        self.calculate_prediction_features(player_name, team2_code),
                        win_chance,
                        num_maps
                    )
                    # rf_pred = self.predict_rf(player_name, team2_code, win_chance)
                    
                    results['team1'].append({
                        'player': player_name,
                        'baseline': baseline_pred
                        # ,'rf': rf_pred
                    })
                except Exception as e:
                    print(f"Error predicting for {player_name}: {str(e)}")
                    continue
            
            # Get players for team 2
            team2_players = self.indexmatch[self.indexmatch['Team'].str.upper() == team2_code.upper()]
            for _, player in team2_players.iterrows():
                try:
                    player_name = player['PP ID']
                    baseline_pred = self.prediction(
                        self.calculate_prediction_features(player_name, team1_code),
                        1 - win_chance,  # Use inverse win chance for team 2
                        num_maps
                    )
                    # rf_pred = self.predict_rf(player_name, team1_code, 1 - win_chance)
                    
                    results['team2'].append({
                        'player': player_name,
                        'baseline': baseline_pred
                        # ,'rf': rf_pred
                    })
                except Exception as e:
                    print(f"Error predicting for {player_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in match prediction: {str(e)}")
            
        return results

if __name__ == "__main__":
    # Load the data
    model = Model()
    print("\nLeague Prediction Model")

    while True:
        try:
            # Get user input
            user_input = input("\nEnter prediction parameters: ").strip()
            if user_input.lower() == 'help':
                print("Enter input as one of:")
                print("1. Player prediction: <player_name> <opponent_team_code> <win_chance> <optional: num maps>")
                print("   Example: Ruler HLE 0.7")
                print("2. Match prediction: match <team1_code> <team2_code> <team1_win_chance> <optional: num maps>")
                print("   Example: match T1 HLE 0.7")
                print("3. Odds calculation: odds <team_odds> <opponent_odds>")
                print("Type 'q' to exit, 'update' to update stats")
                continue
            if user_input.lower() == 'q':
                break
            if user_input.lower() == 'update':
                update_stats()
                continue
            if user_input.lower().startswith('odds'):
                print(f"{calc_odds(*map(float, user_input[5:].split())):.2f}")
                continue
                
            inputs = user_input.split()
            if inputs[0].lower() == 'match':
                # Handle team vs team match prediction
                team1_code, team2_code, win_chance = inputs[1:4]
                win_chance = float(win_chance)
                num_maps = int(inputs[4]) if len(inputs) > 4 else 1
                
                print(f"\nPredicting match: {team1_code.upper()} vs {team2_code.upper()}")
                print(f"Win chance for {team1_code.upper()}: {win_chance:.1%}")
                print(f"Number of maps: {num_maps}")
                
                results = model.predict_match(team1_code, team2_code, win_chance, num_maps)
                
                # Print team 1 results
                print(f"\n{team1_code.upper()} Predictions:")
                for player in results['team1']:
                    print(f"\n{player['player']}:")
                    baseline = player['baseline']
                    print(f"Baseline: {baseline['kills']:.2f} kills, {baseline['assists']:.2f} assists, "
                          f"{baseline['deaths']:.2f} deaths, {baseline['fantasy_score']:.2f} fantasy score")
                    # if player['rf']:
                    #     rf = player['rf']
                    #     print(f"RF: {rf['prediction']:.2f} kills (confidence: {rf['confidence']:.2f})")
                    #     print(f"RF range: ({rf['range'][0]:.2f}, {rf['range'][1]:.2f})")
                
                # Print team 2 results
                print(f"\n{team2_code.upper()} Predictions:")
                for player in results['team2']:
                    print(f"\n{player['player']}:")
                    baseline = player['baseline']
                    print(f"Baseline: {baseline['kills']:.2f} kills, {baseline['assists']:.2f} assists, "
                          f"{baseline['deaths']:.2f} deaths, {baseline['fantasy_score']:.2f} fantasy score")
                    # if player['rf']:
                    #     rf = player['rf']
                    #     print(f"RF: {rf['prediction']:.2f} kills (confidence: {rf['confidence']:.2f})")
                    #     print(f"RF range: ({rf['range'][0]:.2f}, {rf['range'][1]:.2f})")
            else:
                # Handle single player prediction
                player_name, opponent_team_code, win_chance = inputs[:3]
                win_chance = float(win_chance)
                num_maps = int(inputs[3]) if len(inputs) > 3 else 1

                print("\nMaking predictions...")

                baseline_pred = model.prediction(
                    model.calculate_prediction_features(player_name, opponent_team_code),
                    win_chance,
                    num_maps
                )

                print(f"Player: {model.indexmatch[model.indexmatch['PP ID'].str.upper() == player_name.upper()]['PP ID'].iloc[0]}")
                print(f"Opponent: {model.get_team_name_from_code(opponent_team_code)}")
                print(f"Win chance: {win_chance:.0%}")
                print(f"Maps: {num_maps}")

                print(f"Baseline prediction: {baseline_pred['kills']:.2f} kills, "
                      f"{baseline_pred['assists']:.2f} assists, {baseline_pred['deaths']:.2f} deaths, "
                      f"{baseline_pred['fantasy_score']:.2f} fantasy score")

                rf_pred = model.predict_rf(player_name, opponent_team_code, win_chance)
                if rf_pred:
                    print(f"RF prediction: {rf_pred['prediction']:.2f} kills (confidence: {rf_pred['confidence']:.2f})")
                    print(f"RF prediction range: ({rf_pred['range'][0]:.2f}, {rf_pred['range'][1]:.2f})")

        except ValueError as e:
            print(f"Invalid input: {str(e)}")
        except KeyboardInterrupt:
            break
