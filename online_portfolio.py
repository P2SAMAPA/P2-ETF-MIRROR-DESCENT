import numpy as np
import pandas as pd

class ExponentialGradientPortfolio:
    def __init__(self, n_assets, learning_rate=0.05, adaptive=True):
        self.n = n_assets
        self.eta = learning_rate
        self.adaptive = adaptive
        self.weights = np.ones(n_assets) / n_assets   # initial equal weights
        self.history = []   # store (date, weights, portfolio_return)

    def update(self, returns):
        """
        Update weights after observing returns (vector of length n_assets).
        returns: observed daily returns for each asset.
        """
        # Exponential gradient update
        # w_{t+1,i} = w_{t,i} * exp(eta * r_{t,i}) / sum_j w_{t,j} * exp(eta * r_{t,j})
        exp_terms = self.weights * np.exp(self.eta * returns)
        new_weights = exp_terms / np.sum(exp_terms)
        # If adaptive eta = 1 / sqrt(t) (starting from t=1)
        if self.adaptive:
            t = len(self.history) + 1
            self.eta = 0.05 / np.sqrt(t)
        self.weights = new_weights
        # Compute portfolio return for this period (using current weights? Actually we use previous weights)
        # The portfolio return for day t is dot(weights_before_update, returns)
        # We'll store the portfolio return after update but using pre‑update weights.
        port_ret = np.dot(self.weights, returns)
        return port_ret

    def run_online(self, returns_df):
        """
        Iterate over returns_df (index date, columns assets) in chronological order.
        Returns history of weights, portfolio returns, cumulative wealth.
        """
        dates = returns_df.index
        n = len(dates)
        self.history = []
        portfolio_returns = []
        # Reset weights to initial
        self.weights = np.ones(self.n) / self.n
        if self.adaptive:
            self.eta = 0.05   # will be adjusted inside update after first day
        for i in range(n):
            ret_vec = returns_df.iloc[i].values
            port_ret = self.update(ret_vec)
            portfolio_returns.append(port_ret)
            # Store snapshot
            self.history.append({
                'date': dates[i],
                'weights': self.weights.copy(),
                'portfolio_return': port_ret
            })
        # Compute cumulative wealth
        cum_wealth = np.cumprod(1 + np.array(portfolio_returns))
        return self.history, cum_wealth

    def current_weights(self):
        return self.weights.copy()

    def top_assets(self, asset_names, top_n=3):
        idx = np.argsort(self.weights)[::-1][:top_n]
        return [(asset_names[i], self.weights[i]) for i in idx]
