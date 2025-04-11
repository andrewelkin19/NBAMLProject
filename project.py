import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to pull team season data from the api
def get_nba_data(season='2022-23', season_type='Regular Season'):
    team_stats = LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed='PerGame'
    )

    # Get data frame of team stats
    df = team_stats.get_data_frames()[0]
    return df

# Function to clean the data
def clean_data(df):
    float_cols = ["FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "TOV", "PLUS_MINUS"]
    # Ensure data is numerical
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove missing values
    df.dropna(subset=float_cols, inplace=True)
    return df

# Function to split the data into training and test sets
def split_data(df, target_col="W"):
    feature_cols = [
        "FG_PCT", 
        "FG3_PCT", 
        "FT_PCT", 
        "REB", 
        "AST", 
        "TOV", 
        "PLUS_MINUS"
    ]

    X = df[feature_cols]
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# Function to fit a linear regression model to the training data
def train_lr_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to valuate the fit of the Linear Regression model
def evaluate_model_lr(model, X_test, y_test):
    score = model.score(X_test, y_test)
    print(f"R^2 Stat on test set: {score: .4f}")

# Generate final team record predictions from Linear Regression Model
def predict_final_record_lr(model, df):
    features_cols = [
        "FG_PCT", 
        "FG3_PCT", 
        "FT_PCT", 
        "REB", 
        "AST", 
        "TOV", 
        "PLUS_MINUS"
    ]
    X_current = df[features_cols]
    predicted_wins = model.predict(X_current)

    results_df = pd.DataFrame({
        "TEAM_NAME": df["TEAM_NAME"],
        "Predicted_Wins": predicted_wins
    })

    # Round prediction to nearest integer
    results_df["Predicted_Wins"] = results_df["Predicted_Wins"].round().astype(int)
    results_df["Predicted_Losses"] = 82 - results_df["Predicted_Wins"]
    
    return results_df

# Function to plot the Linear Regression model
def plot_lr_fit(y_actual, y_hat):
    plt.figure(figsize=(8,6))

    #Display actual vs predicted datapoints
    plt.scatter(y_actual, y_hat, alpha=0.7, label="Predicted Wins", color="blue")

    # Plot the LR
    min_val = min(y_actual.min(), y_hat.min())
    max_val = max(y_actual.max(), y_hat.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", lw=2, label="Perfect Fit (y = x)")

    plt.xlabel("Actual Wins")
    plt.ylabel("Predicted Wins")
    plt.title("Actual Wins vs. Predicted Wins")
    plt.legend()
    plt.show()

# Get the stats for a given year
nba_stats_2023 = get_nba_data(season="2022-23", season_type="Regular Season")
nba_stats_2023 = clean_data(nba_stats_2023)

# Split into training and test sets
X_train, X_test, y_train, y_test = split_data(nba_stats_2023, target_col="W")

# Train the model on historical season
model_lr = train_lr_model(X_train, y_train)
evaluate_model_lr(model_lr, X_test, y_test)

# Get different season to predict on
nba_stats_2025 = get_nba_data(season="2024-25", season_type="Regular Season")
nba_stats_2025 = clean_data(nba_stats_2025)

# Generate predictions
predictions_lr = predict_final_record_lr(model_lr, nba_stats_2025)
print(predictions_lr)

plot_lr_fit(y_test, predictions_lr)