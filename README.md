# Mirror Descent Online Portfolio Engine

Exponential Gradient (EG) portfolio update – mirror descent with KL divergence. Online convex optimisation: after each day's returns, weights are updated multiplicatively. No walk‑forward; fully online.

- **Algorithm:** w_{t+1,i} = w_{t,i} * exp(η * r_{t,i}) / Z
- **Learning rate:** η=0.05, optionally adaptive (1/√t)
- **Output:** current portfolio weights, top 3 assets
- **Dashboard:** shows weights, cumulative wealth, full weight table

Runs daily on GitHub Actions (re‑runs on all historical data, but that's fine).

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
