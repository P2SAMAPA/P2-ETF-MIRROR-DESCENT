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
        # Clip returns to safe range to avoid overflow in exp
        returns = np.clip(returns, -1.0, 1.0)
        # Remove NaN (set to 0) – but better to skip updates if any NaN?
        # If any return is NaN, we skip the update (keep weights unchanged)
        if np.any(np.isnan(returns)):
            # Keep previous weights
            return 0.0
        # Exponential gradient update
        exp_terms = self.weights * np.exp(self.eta * returns)
        Z = np.sum(exp_terms)
        if Z == 0:
            # fallback: equal weights
            new_weights = np.ones(self.n) / self.n
        else:
            new_weights = exp_terms / Z
        # If any weight becomes NaN (e.g., from 0/0), fallback to equal
        if np.any(np.isnan(new_weights)):
            new_weights = np.ones(self.n) / self.n
        # Update adaptive learning rate
        if self.adaptive:
            t = len(self.history) + 1
            self.eta = 0.05 / np.sqrt(t)
        # Compute portfolio return using old weights
        port_ret = np.dot(self.weights, returns)
        # Store new weights
        self.weights = new_weights
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
            # Skip days with all NaN
            if np.all(np.isnan(ret_vec)):
                # No update, portfolio return 0
                port_ret = 0.0
                # Keep previous weights
            else:
                # Replace NaN with 0 (or we could skip, but we already skip whole vector)
                ret_vec = np.nan_to_num(ret_vec, nan=0.0)
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
