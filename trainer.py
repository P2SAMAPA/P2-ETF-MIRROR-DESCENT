import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from online_portfolio import ExponentialGradientPortfolio

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Mirror Descent Online) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_assets": []}
            continue

        # Run online portfolio algorithm over the entire history
        n_assets = len(tickers)
        egp = ExponentialGradientPortfolio(n_assets, learning_rate=config.LEARNING_RATE, adaptive=config.USE_ADAPTIVE)
        history, cum_wealth = egp.run_online(returns)

        # Current weights after processing all days
        current_weights = egp.current_weights()
        top_assets = egp.top_assets(tickers, top_n=config.TOP_N)

        # Prepare output
        top_list = [{"ticker": ticker, "weight": float(w)} for ticker, w in top_assets]
        full_weights = {ticker: float(current_weights[i]) for i, ticker in enumerate(tickers)}
        # Also get final cumulative wealth and last portfolio return
        final_cum_wealth = float(cum_wealth[-1])
        last_port_return = float(history[-1]['portfolio_return'])

        print(f"  Top 3 assets by weight: {[e['ticker'] for e in top_list]}")
        print(f"  Final cumulative wealth: {final_cum_wealth:.3f} (starting at 1)")

        all_results[universe_name] = {
            "top_assets": top_list,
            "full_weights": full_weights,
            "final_cum_wealth": final_cum_wealth,
            "last_portfolio_return": last_port_return,
            "run_date": today
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/mirror_descent_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Mirror Descent Online Portfolio Engine complete ===")

if __name__ == "__main__":
    main()
