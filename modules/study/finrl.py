from finrl import config
from finrl.env import StockTradingEnv
from finrl.agents.stablebaselines3_models import DRLAgent
from finrl.marketdata.yahoodownloader import YahooDownloader

# Download stock data
df = YahooDownloader(start_date="2020-01-01",
                     end_date="2021-01-01",
                     ticker_list=["AAPL", "MSFT", "AMZN"]).fetch_data()

# Build trading environment
env = StockTradingEnv(df=df, turbulence_threshold=300)

# Train PPO agent
agent = DRLAgent(env=env)
model = agent.get_model("ppo")
trained_model = agent.train_model(model=model, total_timesteps=10000)