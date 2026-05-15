import numpy as np
import pandas as pd

class ExponentialGradientPortfolio:
    def __init__(self, n_assets, learning_rate=0.1, adaptive=True, scale=10.0):
        self.n = n_assets
        self.eta = learning_rate
        self.adaptive = adaptive
        self.scale = scale          # multiplier for returns to amplify differences
        self.weights = np.ones(n_assets) / n_assets
        self.history = []

    def update(self, returns):
        returns = np.clip(returns, -1.0, 1.0)
        if np.any(np.isnan(returns)):
            return 0.0
        # Scale returns to increase sensitivity
        scaled_returns = self.scale * returns
        exp_terms = self.weights * np.exp(self.eta * scaled_returns)
        Z = np.sum(exp_terms)
        if Z == 0 or np.isnan(Z):
            new_weights = np.ones(self.n) / self.n
        else:
            new_weights = exp_terms / Z
        if np.any(np.isnan(new_weights)):
            new_weights = np.ones(self.n) / self.n
        if self.adaptive:
            t = len(self.history) + 1
            self.eta = 0.1 / np.sqrt(t)   # initial eta = 0.1
        port_ret = np.dot(self.weights, returns)
        self.weights = new_weights
        return port_ret

    def run_online(self, returns_df):
        dates = returns_df.index
        n = len(dates)
        self.history = []
        portfolio_returns = []
        self.weights = np.ones(self.n) / self.n
        if self.adaptive:
            self.eta = 0.1
        for i in range(n):
            ret_vec = returns_df.iloc[i].values
            if np.all(np.isnan(ret_vec)):
                port_ret = 0.0
            else:
                ret_vec = np.nan_to_num(ret_vec, nan=0.0)
                port_ret = self.update(ret_vec)
            portfolio_returns.append(port_ret)
            self.history.append({
                'date': dates[i],
                'weights': self.weights.copy(),
                'portfolio_return': port_ret
            })
        cum_wealth = np.cumprod(1 + np.array(portfolio_returns))
        return self.history, cum_wealth

    def current_weights(self):
        return self.weights.copy()

    def top_assets(self, asset_names, top_n=3):
        idx = np.argsort(self.weights)[::-1][:top_n]
        return [(asset_names[i], self.weights[i]) for i in idx]
