import pandas as pd
import csv
import numpy as np
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguedashteamstats

#generate a dictionary of all nba teams
teams = teams.get_teams()
#print(teams[:3])

teamStats = leaguedashteamstats.LeagueDashTeamStats(season="2024-25")
teamStatsDF = teamStats.get_data_frames()[0]
print(teamStatsDF.columns)

def statsToCSV(filename, df):
    dfCSV = df.to_csv(filename, index=False)
    
    
statsToCSV('teamStats.csv', teamStatsDF)