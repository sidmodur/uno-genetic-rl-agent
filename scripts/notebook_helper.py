import q_learning_agent as qa
import genetic_search as genetic
import strategy_agent as sa
import environment as uno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

def getResult(player, winners, turns):
    result = pd.concat([pd.Series(winners), pd.Series(turns)], axis = 1)
    result = result.rename(columns={0:"Winner",1:"Turns"})

    result["Win_rate"] = result["Winner"].apply(lambda x: 1 if x == player else 0)
    result["Win_rate"] = result["Win_rate"].cumsum()/(result.index+1)

    fig = plt.figure(1, figsize=(15,7))

    plt.plot(result.index, result["Win_rate"])
    plt.hlines(0.5, 0, len(winners), colors="grey", linestyles="dashed")

    # Formatting
    plt.title("Win-Rate with Starting Advantage")
    plt.xlabel("Simulations")
    plt.ylabel("Win Rate")
    plt.ylim((0.30,0.80))

    return result, plt

def getWinRate(player, winners):
    return winners.count(player)/len(winners)
