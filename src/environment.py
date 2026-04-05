import pandas as pd
import numpy as np
import torch

## this is the gym used for simulating the actual trading environment 
class TransactionEnvironment: 

    def __init__(self, stockData, featureColumns, transactionCost=0.0005, reward_fn="base", sharpe_lambda=0.1,regime_weights=None):
        self.stockData = stockData
        self.featureColumns = featureColumns
        self.transactionCost = transactionCost
        ## for quick accessing of a random stock
        self.stockList = self.stockData['ticker'].unique()
        self.currentStock = None
        self.currentStockData = None
        self.currentDay = None
        self.currentPosition = None
        ## 0 for cash 1 for invested
        self.reward_fn = reward_fn
        ## different versions of the model 
        self.sharpe_lambda = sharpe_lambda
        self.regime_weights = regime_weights if regime_weights is not None else torch.tensor([1.0, 1.0, 1.0])


    def reset(self):
        rng = np.random.default_rng()
        newStock = rng.choice(self.stockList)
        self.currentStock = newStock
        self.currentStockData = self.stockData[self.stockData['ticker'] == self.currentStock]
        self.currentStockData = self.currentStockData.sort_values('date').reset_index(drop=True)
        self.currentDay = 0
        self.currentPosition = 0
        ## must be cash because you cant hold stock when starting to analyze it 
        return self.getState()


    def getState(self):
        features = self.currentStockData[self.featureColumns].iloc[self.currentDay].values
        regime = self.currentStockData['regime'].iloc[self.currentDay]
        state = np.append(features, [self.currentPosition, regime])
        return state
    

    def rolling_sharpe(self, window=20):
        start = max(0, self.currentDay - window)
        rets = self.currentStockData['ret'].iloc[start:self.currentDay]
        if len(rets) < 2 or rets.std() == 0:
            return 0.0
        return rets.mean() / rets.std()


    def step(self, action):
        next_row = self.currentStockData.iloc[self.currentDay + 1]

        stock_ret = next_row['ret']
        cross_sect_mean = next_row['cross_sect_mean_ret']

        if action == 1:
            if self.reward_fn == "sharpe":
                sharpe = self.rolling_sharpe(window=20)
                reward = stock_ret + self.sharpe_lambda * sharpe - (1 - self.currentPosition) * self.transactionCost
            elif self.reward_fn == "sharpe+regime":
                regime_probs = self.currentStockData[['regime_prob_0', 'regime_prob_1', 'regime_prob_2']].iloc[self.currentDay].values
                regime_probs = torch.FloatTensor(regime_probs.copy())
                regime_scale = (regime_probs * self.regime_weights).sum().item()
                sharpe = self.rolling_sharpe(window=20)
                reward = stock_ret + self.sharpe_lambda * regime_scale * sharpe - (1 - self.currentPosition) * self.transactionCost
            else:  # base 
                reward = stock_ret - (1 - self.currentPosition) * self.transactionCost

        else:
            reward = cross_sect_mean

        self.currentDay += 1
        self.currentPosition = action
        done = self.currentDay >= len(self.currentStockData) - 1
        next_state = None if done else self.getState()

        return next_state, reward, done