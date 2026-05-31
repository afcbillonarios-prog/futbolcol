import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from match_predictor import MatchPredictor
import numpy as np

def predict_champions_league_final():
    """
    Predict the outcome of the 2026 UEFA Champions League final:
    Paris Saint-Germain vs Arsenal
    """
    print("="*60)
    print("PREDICTING 2026 UEFA CHAMPIONS LEAGUE FINAL")
    print("Paris Saint-Germain vs Arsenal")
    print("Puskás Aréna, Budapest - 30 May 2026")
    print("="*60)
    
    # Initialize the match predictor
    # In a real scenario, we would load a pre-trained model
    # For this example, we'll create and train a model with synthetic data
    # that reflects realistic patterns for this type of match
    
    predictor = MatchPredictor()
    
    # Since we don't have a pre-trained model, we'll create one with data
    # that reflects the characteristics of PSG and Arsenal
    print("\nTraining predictive model with relevant data...")
    
    # Generate training data that reflects patterns for elite European matches
    np.random.seed(42)  # For reproducibility
    n_samples = 1500
    
    # Features: [team_form_home, team_form_away, xG_home, xG_away, 
    #           xGA_home, xGA_away, possession_home, possession_away,
    #           pressing_home, pressing_away, injuries_home, injuries_away,
    #           shots_home, shots_away, shots_on_target_home, shots_on_target_away]
    
    X = np.random.rand(n_samples, 16)
    
    # Adjust features to reflect realistic patterns for PSG vs Arsenal type matches
    # PSG typically: strong attack, good possession, high pressing
    # Arsenal typically: good attack, improving defense, moderate possession
    
    # Team form (0-1 scale, higher is better)
    # PSG as defending champions: strong form
    # Arsenal: good form but slightly less consistent
    X[:, 0] = np.random.beta(8, 2, n_samples) * 0.3 + 0.6  # PSG form: 0.6-0.9
    X[:, 1] = np.random.beta(7, 3, n_samples) * 0.25 + 0.6  # Arsenal form: 0.6-0.85
    
    # xG (expected goals) - PSG typically creates more high-quality chances
    X[:, 2] = np.random.gamma(2, 0.8, n_samples)  # PSG xG: typically 1.0-2.5
    X[:, 3] = np.random.gamma(1.8, 0.7, n_samples)  # Arsenal xG: typically 0.9-2.2
    
    # xGA (expected goals against) - defense
    X[:, 4] = np.random.gamma(1.5, 0.5, n_samples)  # PSG xGA: typically 0.6-1.5
    X[:, 5] = np.random.gamma(1.6, 0.5, n_samples)  # Arsenal xGA: typically 0.7-1.6
    
    # Possession (%)
    X[:, 6] = np.random.normal(58, 8, n_samples)    # PSG possession: typically 50-66%
    X[:, 7] = np.random.normal(48, 8, n_samples)    # Arsenal possession: typically 40-56%
    
    # Pressing intensity (0-1 scale)
    X[:, 8] = np.random.beta(7, 3, n_samples)       # PSG pressing: typically 0.6-0.9
    X[:, 9] = np.random.beta(6, 4, n_samples)       # Arsenal pressing: typically 0.4-0.8
    
    # Injuries (number of key players unavailable)
    X[:, 10] = np.random.poisson(1.5, n_samples)    # PSG injuries: typically 0-4
    X[:, 11] = np.random.poisson(2.0, n_samples)    # Arsenal injuries: typically 0-5
    
    # Shots (total)
    X[:, 12] = np.random.poisson(18, n_samples)     # PSG shots: typically 10-28
    X[:, 13] = np.random.poisson(15, n_samples)     # Arsenal shots: typically 8-24
    
    # Shots on target
    X[:, 14] = np.random.poisson(7, n_samples)      # PSG SoT: typically 3-12
    X[:, 15] = np.random.poisson(6, n_samples)      # Arsenal SoT: typically 2-10
    
    # Ensure realistic ranges
    X[:, 0] = np.clip(X[:, 0], 0.4, 0.95)   # Team form
    X[:, 1] = np.clip(X[:, 1], 0.4, 0.9)
    X[:, 2] = np.clip(X[:, 2], 0.2, 3.5)    # xG
    X[:, 3] = np.clip(X[:, 3], 0.2, 3.0)
    X[:, 4] = np.clip(X[:, 4], 0.2, 2.5)    # xGA
    X[:, 5] = np.clip(X[:, 5], 0.2, 2.8)
    X[:, 6] = np.clip(X[:, 6], 30, 80)      # Possession
    X[:, 7] = np.clip(X[:, 7], 20, 70)
    X[:, 8] = np.clip(X[:, 8], 0.2, 0.95)   # Pressing
    X[:, 9] = np.clip(X[:, 9], 0.1, 0.9)
    X[:, 10] = np.clip(X[:, 10], 0, 6)      # Injuries
    X[:, 11] = np.clip(X[:, 11], 0, 8)
    X[:, 12] = np.clip(X[:, 12], 4, 40)     # Shots
    X[:, 13] = np.clip(X[:, 13], 3, 35)
    X[:, 14] = np.clip(X[:, 14], 0, 18)     # Shots on target
    X[:, 15] = np.clip(X[:, 15], 0, 15)
    
    # Create target variable (match outcome from home team perspective)
    # 0: PSG loss, 1: Draw, 2: PSG win
    
    # Calculate a "match score" based on features
    # This simulates how the model would learn patterns
    home_advantage = (
        # Form difference
        (X[:, 0] - X[:, 1]) * 0.3 +
        # xG difference (attack)
        (X[:, 2] - X[:, 3]) * 0.25 +
        # xGA difference (defense, negative because lower is better)
        (X[:, 4] - X[:, 5]) * (-0.2) +
        # Possession advantage
        (X[:, 6] - X[:, 7]) * 0.15 +
        # Pressing advantage
        (X[:, 8] - X[:, 9]) * 0.1 +
        # Injury disadvantage (negative)
        (X[:, 11] - X[:, 10]) * (-0.08) +
        # Shot creation
        (X[:, 12] - X[:, 13]) * 0.02 +
        # Shot quality
        (X[:, 14] - X[:, 15]) * 0.03
    )
    
    # Add home advantage (playing at neutral venue, but PSG might have slight edge)
    home_advantage += 0.1
    
    # Add some randomness
    home_advantage += np.random.normal(0, 0.15, n_samples)
    
    # Convert to probabilities using softmax-like approach
    # Higher home_advantage -> more likely PSG win
    # Lower home_advantage -> more likely Arsenal win
    # Middle values -> more likely draw
    
    # Convert to win/draw/loss probabilities
    exp_win = np.exp(np.maximum(home_advantage, 0))
    exp_draw = np.exp(np.abs(home_advantage) * 0.5)  # Draw likelihood highest when teams are close
    exp_loss = np.exp(np.maximum(-home_advantage, 0))
    
    # Normalize to get probabilities
    sum_exp = exp_win + exp_draw + exp_loss
    psg_win_prob = exp_win / sum_exp
    draw_prob = exp_draw / sum_exp
    arsenal_win_prob = exp_loss / sum_exp
    
    # Sample outcomes based on these probabilities
    y = np.array([
        np.random.choice([0, 1, 2], p=[l, d, w])
        for l, d, w in zip(arsenal_win_prob, draw_prob, psg_win_prob)
    ])
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training XGBoost model...")
    predictor.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    from sklearn.metrics import accuracy_score
    y_pred = predictor.model.predict(predictor.scaler.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test data: {accuracy:.3f}")
    
    # Now predict the actual Champions League final
    # Based on recent form, squad quality, and other factors
    
    final_match_data = {
        # Team form (0-1 scale)
        'team_form_home': 0.82,   # PSG: Defending champions, strong squad
        'team_form_away': 0.75,   # Arsenal: Improved under Arteta, consistent
        
        # Expected Goals (xG) per match
        'xG_home': 1.9,           # PSG: Strong attacking options
        'xG_away': 1.6,           # Arsenal: Good but slightly less clinical
        
        # Expected Goals Against (xGA) per match
        'xGA_home': 0.85,         # PSG: Solid defense when organized
        'xGA_away': 0.95,         # Arsenal: Improved defense but can be vulnerable
        
        # Possession percentage
        'possession_home': 56,    # PSG: Typically controls possession
        'possession_away': 44,    # Arsenal: Effective on counter
        
        # Pressing intensity (0-1 scale)
        'pressing_home': 0.72,    # PSG: High press under Enrique
        'pressing_away': 0.58,    # Arsenal: Moderate press, focuses on shape
        
        # Injuries (key players unavailable)
        'injuries_home': 1,       # PSG: Minor concerns
        'injuries_away': 2,       # Arsenal: Some injury worries
        
        # Shots per match
        'shots_home': 17,         # PSG: Creates plenty of chances
        'shots_away': 14,         # Arsenal: Fewer but quality chances
        
        # Shots on target per match
        'shots_on_target_home': 7, # PSG: Good accuracy
        'shots_on_target_away': 5  # Arsenal: Decent accuracy
    }
    
    # Make prediction
    print("\nAnalyzing match data...")
    probabilities = predictor.predict(final_match_data)
    predicted_outcome = predictor.predict_outcome(final_match_data)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Paris Saint-Germain (Home) Win Probability: {probabilities['prob_win']:.1%}")
    print(f"Draw Probability: {probabilities['prob_draw']:.1%}")
    print(f"Arsenal (Away) Win Probability: {probabilities['prob_loss']:.1%}")
    print("-"*60)
    print(f"Most Likely Outcome: {predicted_outcome.upper()} (from PSG perspective)")
    
    # Convert to actual match result
    if predicted_outcome == 'win':
        result_text = "PARIS SAINT-GERMAIN WIN"
    elif predicted_outcome == 'draw':
        result_text = "DRAW"
    else:
        result_text = "ARSENAL WIN"
    
    print(f"PREDICTED RESULT: {result_text}")
    print("="*60)
    
    # Additional insights
    print("\nKEY FACTORS INFLUENCING PREDICTION:")
    print("- PSG's experience as defending champions")
    print("- Arsenal's improvement under Mikel Arteta")
    print("- PSG's slight edge in attacking quality (xG)")
    print("- Arsenal's solid defensive organization")
    print("- Neutral venue in Budapest (Puskás Aréna)")
    
    # Monte Carlo simulation for score prediction
    print("\n" + "="*60)
    print("MONTE CARLO SCORE SIMULATION (10,000 matches)")
    print("="*60)
    
    # Simulate goals using Poisson distribution based on xG
    n_simulations = 10000
    psg_goals = np.random.poisson(final_match_data['xG_home'], n_simulations)
    arsenal_goals = np.random.poisson(final_match_data['xG_away'], n_simulations)
    
    # Calculate most common scores
    from collections import Counter
    score_combinations = list(zip(psg_goals, arsenal_goals))
    score_counts = Counter(score_combinations)
    top_scores = score_counts.most_common(5)
    
    print("Most Likely Scores:")
    for i, ((psg, arsenal), count) in enumerate(top_scores, 1):
        probability = count / n_simulations * 100
        print(f"{i}. PSG {psg} - {arsenal} Arsenal ({probability:.1f}%)")
    
    # Calculate win/draw/loss from simulation
    psg_wins = sum(1 for p, a in score_combinations if p > a)
    draws = sum(1 for p, a in score_combinations if p == a)
    arsenal_wins = sum(1 for p, a in score_combinations if p < a)
    
    print(f"\nSimulated Outcomes:")
    print(f"PSG Wins: {psg_wins/n_simulations:.1%}")
    print(f"Draws: {draws/n_simulations:.1%}")
    print(f"Arsenal Wins: {arsenal_wins/n_simulations:.1%}")
    
    return {
        'probabilities': probabilities,
        'predicted_outcome': predicted_outcome,
        'most_likely_score': top_scores[0][0] if top_scores else (1, 1)
    }

if __name__ == "__main__":
    try:
        result = predict_champions_league_final()
        print("\nPrediction complete!")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install xgboost scikit-learn numpy pandas")