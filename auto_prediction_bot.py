import schedule
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from match_predictor import MatchPredictor
from xg_calculator import xGCalculator

class FootballAnalyticsBot:
    """
    An automated bot for football analytics predictions and reports.
    Features:
    - Scheduled match predictions
    - Automated xG calculations
    - Injury risk alerts
    - Performance reports
    """
    
    def __init__(self, data_path="./data"):
        """
        Initialize the auto prediction bot.
        
        Args:
            data_path (str): Path to store data and models.
        """
        self.data_path = data_path
        self.match_predictor = MatchPredictor()
        self.xg_calculator = xGCalculator()
        self.is_running = False
        
        # Create necessary directories
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(os.path.join(data_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "reports"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "predictions"), exist_ok=True)
        
        # Try to load pre-trained models
        self._load_models()
        
        print("Football Analytics Bot initialized")
    
    def _load_models(self):
        """Load pre-trained models if available."""
        model_dir = os.path.join(self.data_path, "models")
        try:
            self.match_predictor.load_model(model_dir)
            self.xg_calculator.load_model(model_dir)
            print("Pre-trained models loaded successfully")
        except Exception as e:
            print(f"No pre-trained models found: {e}")
            print("Models will be trained on first use")
    
    def _ensure_models_trained(self):
        """Ensure models are trained before making predictions."""
        # Check if match predictor needs training
        # Check if model exists and is fitted (has feature_importances_ or similar)
        match_predictor_needs_training = (
            self.match_predictor.model is None or 
            not hasattr(self.match_predictor.model, 'feature_importances_')
        )
        if match_predictor_needs_training:
            print("Training match predictor model...")
            self._train_match_predictor()
        
        # Check if xG calculator needs training
        xg_calculator_needs_training = (
            self.xg_calculator.model is None or 
            not hasattr(self.xg_calculator.model, 'feature_importances_')
        )
        if xg_calculator_needs_training:
            print("Training xG calculator model...")
            self._train_xg_calculator()
    
    def _train_match_predictor(self):
        """Train the match predictor with synthetic data."""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: [team_form_home, team_form_away, xG_home, xG_away, 
        #           xGA_home, xGA_away, possession_home, possession_away,
        #           pressing_home, pressing_away, injuries_home, injuries_away,
        #           shots_home, shots_away, shots_on_target_home, shots_on_target_away]
        
        X = np.random.rand(n_samples, 16)
        
        # Create realistic relationships for target variable
        home_advantage = (
            X[:, 2] * 0.3 - X[:, 3] * 0.2 +  # xG_home - xG_away weight
            X[:, 6] * 0.2 - X[:, 7] * 0.15 + # possession_home - possession_away
            X[:, 8] * 0.15 - X[:, 9] * 0.1   # pressing_home - pressing_away
        )
        
        # Convert to probabilities and sample outcomes
        prob_win = 1 / (1 + np.exp(-(home_advantage + 0.5)))  # Sigmoid with home bias
        prob_draw = np.full(n_samples, 0.25)  # Base draw probability
        prob_loss = 1 - prob_win - prob_draw
        
        # Normalize probabilities
        total = prob_win + prob_draw + prob_loss
        prob_win /= total
        prob_draw /= total
        prob_loss /= total
        
        # Sample outcomes
        y = np.array([np.random.choice([0, 1, 2], p=[l, d, w]) 
                      for l, d, w in zip(prob_loss, prob_draw, prob_win)])
        
        # Split into train/validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.match_predictor.train(X_train, y_train, X_val, y_val)
        print("Match predictor model trained")
    
    def _train_xg_calculator(self):
        """Train the xG calculator with synthetic data."""
        # Generate synthetic training data for xG
        np.random.seed(42)
        n_samples = 2000
        
        # Features: [x, y, distance_to_goal, angle_to_goal, body_part, assist_type, 
        #           game_situation, defender_distance, goalkeeper_position, shot_type, 
        #           is_first_time, is_header]
        
        X = np.zeros((n_samples, 12))
        
        # x, y coordinates (normalized 0-1 pitch)
        X[:, 0] = np.random.beta(2, 5, n_samples)  # x: more shots near opponent goal
        X[:, 1] = np.random.beta(2, 2, n_samples) * 0.8 + 0.1  # y: across width
        
        # Derived features
        X[:, 2] = np.sqrt((X[:, 0] - 1)**2 + (X[:, 1] - 0.5)**2) * 105  # distance_to_goal (m)
        X[:, 3] = np.degrees(np.arctan2(7.32, X[:, 2]))  # angle_to_goal (degrees)
        
        # Other features
        X[:, 4] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.4, 0.15, 0.05])  # body_part
        X[:, 5] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.15, 0.1, 0.05])  # assist_type
        X[:, 6] = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1])  # game_situation
        X[:, 7] = np.random.exponential(5, n_samples)  # defender_distance
        X[:, 8] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.2, 0.2, 0.1])  # goalkeeper_position
        X[:, 9] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.1, 0.15, 0.05])  # shot_type
        X[:, 10] = np.random.binomial(1, 0.3, n_samples)  # is_first_time
        X[:, 11] = (X[:, 4] == 2).astype(int)  # is_header
        
        # Create realistic xG values based on features
        xg_true = (
            0.1 * (1 - X[:, 0]) +  # Better position (higher x)
            0.1 * (1 - abs(X[:, 1] - 0.5) * 2) +  # Central position
            0.3 * np.exp(-X[:, 2] / 20) +  # Distance decay
            0.2 * (X[:, 3] / 30) +  # Angle factor
            0.1 * (X[:, 4] == 2).astype(float) +  # Header bonus
            0.1 * (X[:, 5] == 2).astype(float) +  # Cross assist bonus
            0.1 * (X[:, 9] == 1).astype(float) +  # Volley bonus
            np.random.normal(0, 0.05, n_samples)  # Noise
        )
        
        # Ensure xG is between 0 and 1
        xg_true = np.clip(xg_true, 0.01, 0.95)
        
        # Generate actual goals (binary) based on xG probability
        y = np.random.binomial(1, xg_true)
        
        # Split into train/validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.xg_calculator.train(X_train, y_train, X_val, y_val)
        print("xG calculator model trained")
    
    def _save_models(self):
        """Save trained models."""
        model_dir = os.path.join(self.data_path, "models")
        self.match_predictor.save_model(model_dir)
        self.xg_calculator.save_model(model_dir)
    
    def fetch_fixture_data(self, league="Champions League", date=None):
        """
        Fetch fixture data for upcoming matches.
        In a real implementation, this would connect to APIs like:
        - Football-Data.org
        - API-Football
        - Sportradar
        - Or direct club/league feeds
        
        For demonstration, we'll generate synthetic fixture data.
        
        Args:
            league (str): League name to fetch fixtures for.
            date (str): Date in YYYY-MM-DD format (defaults to today).
            
        Returns:
            list: List of fixture dictionaries.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Fetching {league} fixtures for {date}...")
        
        # In reality, this would be an API call
        # For demo, we'll create synthetic data based on known fixtures
        
        fixtures = []
        
        if league == "Champions League" and date == "2026-05-30":
            # The specific final we're interested in
            fixtures.append({
                'match_id': 'UCL_2026_FINAL',
                'date': date,
                'home_team': 'Paris Saint-Germain',
                'away_team': 'Arsenal',
                'venue': 'Puskás Aréna, Budapest',
                'competition': 'UEFA Champions League',
                'stage': 'Final',
                # Features needed for prediction
                'team_form_home': 0.82,
                'team_form_away': 0.75,
                'xG_home': 1.9,
                'xG_away': 1.6,
                'xGA_home': 0.85,
                'xGA_away': 0.95,
                'possession_home': 56,
                'possession_away': 44,
                'pressing_home': 0.72,
                'pressing_away': 0.58,
                'injuries_home': 1,
                'injuries_away': 2,
                'shots_home': 17,
                'shots_away': 14,
                'shots_on_target_home': 7,
                'shots_on_target_away': 5
            })
        
        elif league == "Categoria Primera A":
            # Colombian league fixtures (synthetic)
            colombian_teams = [
                'Atlético Nacional', 'Millonarios FC', 'América de Cali',
                'Independiente Medellín', 'Junior FC', 'Deportivo Cali',
                'Santa Fe', 'Once Caldas', 'Envigado FC', 'Águilas Doradas'
            ]
            
            # Generate 5 synthetic matches
            for i in range(5):
                home_team = np.random.choice(colombian_teams)
                away_team = np.random.choice([t for t in colombian_teams if t != home_team])
                
                fixtures.append({
                    'match_id': f'COL_{date.replace("-", "")}_{i+1:02d}',
                    'date': date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'venue': f'Estadio {home_team.split()[-1]}',  # Simplified
                    'competition': 'Categoria Primera A',
                    'stage': 'Regular Season',
                    # Features for Colombian league (slightly different ranges)
                    'team_form_home': np.random.beta(5, 3),
                    'team_form_away': np.random.beta(5, 3),
                    'xG_home': np.random.gamma(1.5, 0.8),
                    'xG_away': np.random.gamma(1.5, 0.8),
                    'xGA_home': np.random.gamma(1.2, 0.6),
                    'xGA_away': np.random.gamma(1.2, 0.6),
                    'possession_home': np.random.normal(50, 10),
                    'possession_away': np.random.normal(50, 10),
                    'pressing_home': np.random.beta(4, 4),
                    'pressing_away': np.random.beta(4, 4),
                    'injuries_home': np.random.poisson(1.5),
                    'injuries_away': np.random.poisson(1.5),
                    'shots_home': np.random.poisson(12),
                    'shots_away': np.random.poisson(12),
                    'shots_on_target_home': np.random.poisson(5),
                    'shots_on_target_away': np.random.poisson(5)
                })
        
        # Ensure realistic ranges
        for fixture in fixtures:
            fixture['team_form_home'] = np.clip(fixture['team_form_home'], 0.3, 0.9)
            fixture['team_form_away'] = np.clip(fixture['team_form_away'], 0.3, 0.9)
            fixture['xG_home'] = np.clip(fixture['xG_home'], 0.2, 3.0)
            fixture['xG_away'] = np.clip(fixture['xG_away'], 0.2, 3.0)
            fixture['xGA_home'] = np.clip(fixture['xGA_home'], 0.2, 2.5)
            fixture['xGA_away'] = np.clip(fixture['xGA_away'], 0.2, 2.5)
            fixture['possession_home'] = np.clip(fixture['possession_home'], 25, 75)
            fixture['possession_away'] = np.clip(fixture['possession_away'], 25, 75)
            fixture['pressing_home'] = np.clip(fixture['pressing_home'], 0.1, 0.9)
            fixture['pressing_away'] = np.clip(fixture['pressing_away'], 0.1, 0.9)
            fixture['injuries_home'] = np.clip(fixture['injuries_home'], 0, 5)
            fixture['injuries_away'] = np.clip(fixture['injuries_away'], 0, 5)
            fixture['shots_home'] = np.clip(fixture['shots_home'], 3, 25)
            fixture['shots_away'] = np.clip(fixture['shots_away'], 3, 25)
            fixture['shots_on_target_home'] = np.clip(fixture['shots_on_target_home'], 0, 12)
            fixture['shots_on_target_away'] = np.clip(fixture['shots_on_target_away'], 0, 12)
        
        print(f"Fetched {len(fixtures)} fixtures")
        return fixtures
    
    def predict_match(self, fixture_data):
        """
        Generate prediction for a single match.
        
        Args:
            fixture_data (dict): Match fixture data.
            
        Returns:
            dict: Prediction results.
        """
        # Ensure models are trained before prediction
        self._ensure_models_trained()
        
        # Extract features needed for match predictor
        match_features = {
            'team_form_home': fixture_data['team_form_home'],
            'team_form_away': fixture_data['team_form_away'],
            'xG_home': fixture_data['xG_home'],
            'xG_away': fixture_data['xG_away'],
            'xGA_home': fixture_data['xGA_home'],
            'xGA_away': fixture_data['xGA_away'],
            'possession_home': fixture_data['possession_home'],
            'possession_away': fixture_data['possession_away'],
            'pressing_home': fixture_data['pressing_home'],
            'pressing_away': fixture_data['pressing_away'],
            'injuries_home': fixture_data['injuries_home'],
            'injuries_away': fixture_data['injuries_away'],
            'shots_home': fixture_data['shots_home'],
            'shots_away': fixture_data['shots_away'],
            'shots_on_target_home': fixture_data['shots_on_target_home'],
            'shots_on_target_away': fixture_data['shots_on_target_away']
        }
        
        # Get prediction from match predictor
        probabilities = self.match_predictor.predict(match_features)
        predicted_outcome = self.match_predictor.predict_outcome(match_features)
        
        # Calculate expected score using xG
        home_xg = fixture_data['xG_home']
        away_xg = fixture_data['xG_away']
        
        # Simple Poisson-based expected score
        expected_home_goals = round(home_xg, 1)
        expected_away_goals = round(away_xg, 1)
        
        # Format result
        result = {
            'match_id': fixture_data['match_id'],
            'date': fixture_data['date'],
            'home_team': fixture_data['home_team'],
            'away_team': fixture_data['away_team'],
            'venue': fixture_data['venue'],
            'competition': fixture_data['competition'],
            'stage': fixture_data['stage'],
            'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'probabilities': {
                'home_win': probabilities['prob_win'],
                'draw': probabilities['prob_draw'],
                'away_win': probabilities['prob_loss']
            },
            'predicted_outcome': predicted_outcome,
            'expected_score': f"{expected_home_goals} - {expected_away_goals}",
            'confidence': max(probabilities['prob_win'], probabilities['prob_draw'], probabilities['prob_loss'])
        }
        
        # Add readable outcome
        if predicted_outcome == 'win':
            result['readable_prediction'] = f"{fixture_data['home_team']} win"
        elif predicted_outcome == 'draw':
            result['readable_prediction'] = "Draw"
        else:
            result['readable_prediction'] = f"{fixture_data['away_team']} win"
        
        return result
    
    def calculate_match_xg(self, fixture_data):
        """
        Calculate expected goals for a match using shot data.
        In reality, this would use actual shot data from the match.
        For demonstration, we'll estimate based on team xG values.
        
        Args:
            fixture_data (dict): Match fixture data.
            
        Returns:
            dict: xG analysis results.
        """
        # In a real implementation, we would:
        # 1. Get actual shot data (x, y, body_part, etc.) for the match
        # 2. Feed each shot to the xG calculator
        # 3. Sum up for total team xG
        
        # For demo, we'll use the pre-match xG estimates as our prediction
        # and show how actual xG would be calculated post-match
        
        home_xg_prediction = fixture_data['xG_home']
        away_xg_prediction = fixture_data['xG_away']
        
        # Simulate some variance for "actual" xG (in reality would come from shots)
        home_xg_actual = max(0, home_xg_prediction + np.random.normal(0, 0.3))
        away_xg_actual = max(0, away_xg_prediction + np.random.normal(0, 0.3))
        
        return {
            'match_id': fixture_data['match_id'],
            'home_team': fixture_data['home_team'],
            'away_team': fixture_data['away_team'],
            'predicted_xg': {
                'home': round(home_xg_prediction, 2),
                'away': round(away_xg_prediction, 2)
            },
            'estimated_actual_xg': {
                'home': round(home_xg_actual, 2),
                'away': round(away_xg_actual, 2)
            },
            'xg_difference': {
                'home': round(home_xg_actual - home_xg_prediction, 2),
                'away': round(away_xg_actual - away_xg_prediction, 2)
            }
        }
    
    def generate_report(self, predictions, report_type="daily"):
        """
        Generate a formatted report from predictions.
        
        Args:
            predictions (list): List of prediction dictionaries.
            report_type (str): Type of report (daily, weekly, etc.).
            
        Returns:
            str: Formatted report.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_header = f"""
{'='*80}
FOOTBALL ANALYTICS BOT - {report_type.upper()} REPORT
Generated: {timestamp}
{'='*80}
"""
        
        if not predictions:
            return report_header + "\nNo matches to report.\n"
        
        report_body = ""
        for pred in predictions:
            report_body += f"""
Match: {pred['home_team']} vs {pred['away_team']}
Competition: {pred['competition']} - {pred['stage']}
Date: {pred['date']}
Venue: {pred['venue']}

Prediction:
  Most Likely: {pred['readable_prediction']}
  Confidence: {pred['confidence']:.1%}
  Probabilities:
    - {pred['home_team']} Win: {pred['probabilities']['home_win']:.1%}
    - Draw: {pred['probabilities']['draw']:.1%}
    - {pred['away_team']} Win: {pred['probabilities']['away_win']:.1%}
  Expected Score: {pred['expected_score']}

---
"""
        
        report_footer = f"""
{'='*80}
End of Report
{'='*80}
"""
        
        return report_header + report_body + report_footer
    
    def save_predictions(self, predictions, filename=None):
        """
        Save predictions to a JSON file.
        
        Args:
            predictions (list): List of prediction dictionaries.
            filename (str): Optional filename.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.json"
        
        filepath = os.path.join(self.data_path, "predictions", filename)
        
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        print(f"Predictions saved to {filepath}")
    
    def save_report(self, report, filename=None):
        """
        Save report to a text file.
        
        Args:
            report (str): Report text.
            filename (str): Optional filename.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.txt"
        
        filepath = os.path.join(self.data_path, "reports", filename)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {filepath}")
    
    def run_daily_predictions(self):
        """
        Run the daily prediction routine.
        This is the main function that would be scheduled.
        """
        print("\n" + "="*60)
        print("RUNNING DAILY PREDICTIONS")
        print("="*60)
        
        all_predictions = []
        
        # Predict Champions League fixtures
        ucl_fixtures = self.fetch_fixture_data("Champions League", "2026-05-30")
        for fixture in ucl_fixtures:
            try:
                prediction = self.predict_match(fixture)
                xg_analysis = self.calculate_match_xg(fixture)
                
                # Combine prediction and xG analysis
                prediction['xg_analysis'] = xg_analysis
                all_predictions.append(prediction)
                
                print(f"[+] Predicted: {prediction['home_team']} vs {prediction['away_team']}")
            except Exception as e:
                print(f"[-] Error predicting {fixture.get('home_team', 'Unknown')} vs {fixture.get('away_team', 'Unknown')}: {e}")
        
        # Predict Colombian league fixtures
        col_fixtures = self.fetch_fixture_data("Categoria Primera A")
        for fixture in col_fixtures:
            try:
                prediction = self.predict_match(fixture)
                xg_analysis = self.calculate_match_xg(fixture)
                
                # Combine prediction and xG analysis
                prediction['xg_analysis'] = xg_analysis
                all_predictions.append(prediction)
                
                print(f"[+] Predicted: {prediction['home_team']} vs {prediction['away_team']}")
            except Exception as e:
                print(f"[-] Error predicting {fixture.get('home_team', 'Unknown')} vs {fixture.get('away_team', 'Unknown')}: {e}")
        
        if all_predictions:
            # Generate and save report
            report = self.generate_report(all_predictions, "Daily")
            print("\n" + report)
            
            # Save outputs
            self.save_predictions(all_predictions)
            self.save_report(report)
            
            # Save models after training (if they were trained during this run)
            self._save_models()
            
            return all_predictions
        else:
            print("No predictions generated")
            return []
    
    def start_scheduled_mode(self):
        """
        Start the bot in scheduled mode.
        This would run predictions at regular intervals.
        """
        print("Starting Football Analytics Bot in scheduled mode...")
        print("Scheduled jobs:")
        print("  - Daily predictions at 08:00 AM")
        print("  - Pre-match predictions 2 hours before kickoff")
        print("  - Post-match analysis 2 hours after final whistle")
        
        # Schedule daily predictions
        schedule.every().day.at("08:00").do(self.run_daily_predictions)
        
        # For demonstration, we'll run once immediately and then exit
        # In a real deployment, this would run continuously
        print("\nRunning initial prediction cycle...")
        self.run_daily_predictions()
        
        print("\nBot would now wait for scheduled times...")
        print("To run continuously, remove the 'break' and uncomment the loop below:")
        print("# while self.is_running:")
        print("#     schedule.run_pending()")
        print("#     time.sleep(60)")
        
        # For demo purposes, we'll just run once
        # Uncomment the following lines for continuous operation:
        # self.is_running = True
        # while self.is_running:
        #     schedule.run_pending()
        #     time.sleep(60)
    
    def run_once(self):
        """
        Run the bot once for testing/demonstration.
        """
        return self.run_daily_predictions()

# Example usage and demonstration
if __name__ == "__main__":
    print("="*60)
    print("FOOTBALL ANALYTICS AUTO PREDICTION BOT")
    print("="*60)
    
    # Initialize the bot
    bot = FootballAnalyticsBot()
    
    # Run once for demonstration
    predictions = bot.run_once()
    
    print("\n" + "="*60)
    print("BOT OPERATION COMPLETE")
    print("="*60)
    print(f"Generated {len(predictions)} predictions")
    print("Check the 'data/predictions' and 'data/reports' directories for outputs")
    print("\nTo run in scheduled mode, call: bot.start_scheduled_mode()")
    print("(This would require keeping the script running)")
