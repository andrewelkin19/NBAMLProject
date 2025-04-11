import pandas as pd
import numpy as np
import statsmodels.api as sum
from matplotlib.pyplot import subplots
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
import nbaDataManager

teamStatsDF = nbaDataManager.teamStatsDF
#print(teamStatsDF.columns)



