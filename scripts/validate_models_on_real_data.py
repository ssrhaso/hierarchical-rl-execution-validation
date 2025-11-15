import sys
sys.path.append('.')

import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np

from src.data.alpaca_loader import AlpacaDataLoader
from src.environments.real_market_env import RealMarketEnv
from src.agents.load_trained_models import load_trained_models
from src.baselines.twap import TWAPPolicy
from src.baselines.vwap import VWAPPolicy

# Inline definition for the random policy baseline
class RandomPolicy:
    """Uniformly random action selection baseline."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        return self.action_space.sample(), None

    def reset(self):
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/validation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_episode(env, agent, agent_name: str, max_steps: int = 1000) -> dict:
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    actions_taken = []

    while not done and step < max_steps:
        try:
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            actions_taken.append(action)
            done = terminated or truncated
            step += 1
        except Exception as e:
            logger.error(f"Error during episode step {step} for {agent_name}: {e}")
            break

    metrics = {
        'agent_name': agent_name,
        'total_reward': total_reward,
        'slippage_bps': info.get('slippage_bps', np.nan),
        'exec_time_steps': step,
        'avg_exec_price': info.get('avg_execution_price', np.nan),
        'completion_rate': (1 - info.get('inventory_remaining', 1000) / 1000) * 100,
        'inventory_remaining': info.get('inventory_remaining', 1000),
        'shares_executed': info.get('shares_executed', 0),
        'action_distribution': {
            'wait': actions_taken.count(0),
            'execute_10pct': actions_taken.count(1),
            'execute_5pct': actions_taken.count(2)
        }
    }

    logger.info(f"[DONE] {agent_name:20s} | {step:4d} steps | {metrics['slippage_bps']:6.2f} bps | {metrics['completion_rate']:5.1f}% complete")
    return metrics

def main():
    SYMBOL = "AAPL"
    START_DATE = "2025-10-01"
    END_DATE = "2025-10-31"
    PARENT_ORDER_SIZE = 1000

    logger.info("=" * 80)
    logger.info("HIERARCHICAL RL OPTIMAL EXECUTION - VALIDATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Date Range: {START_DATE} to {END_DATE}")
    logger.info(f"Parent Order Size: {PARENT_ORDER_SIZE} shares")
    logger.info("=" * 80)
    logger.info("")

    # Data loader
    try:
        loader = AlpacaDataLoader()
        logger.info("[OK] Data loader initialized")
        logger.info("")
    except Exception as e:
        logger.error(f"[FAILED] Data loader initialization: {e}")
        logger.error("Check .env file for ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return

    # RL models
    try:
        rl_agents = load_trained_models(models_dir="models/")
        if not rl_agents:
            logger.warning("[WARNING] No RL models loaded - will test baselines only")
    except Exception as e:
        logger.error(f"[FAILED] Model loading error: {e}")
        rl_agents = {}

    # Date range
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    date_range = pd.date_range(start, end, freq='B')  # Business days only

    results = []
    logger.info(f"Testing on {len(date_range)} trading days")
    logger.info("")
    logger.info("=" * 80)
    logger.info("STARTING VALIDATION")
    logger.info("=" * 80)
    logger.info("")

    for idx, date in enumerate(date_range, 1):
        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"[Day {idx}/{len(date_range)}] {date_str}")
        try:
            market_data = loader.download_bars(
                symbol=SYMBOL,
                start_date=date_str,
                end_date=date_str,
                timeframe='1Min'
            )
            if market_data.empty:
                logger.warning(f"[SKIP] No market data available for {date_str}")
                logger.info("")
                continue
            logger.info(f"[OK] Loaded {len(market_data)} market bars")
        except Exception as e:
            logger.error(f"[FAILED] Data fetch for {date_str}: {e}")
            logger.info("")
            continue

        try:
            env = RealMarketEnv(
                market_data=market_data,
                parent_order_size=PARENT_ORDER_SIZE
            )
        except Exception as e:
            logger.error(f"[FAILED] Environment creation for {date_str}: {e}")
            logger.info("")
            continue

        # RL agents
        for model_name, agent in rl_agents.items():
            try:
                metrics = run_episode(env, agent, model_name)
                results.append({
                    'date': date_str,
                    'symbol': SYMBOL,
                    'model': model_name,
                    'model_type': 'RL',
                    **metrics
                })
            except Exception as e:
                logger.error(f"[FAILED] RL agent {model_name}: {e}")

        # Baseline policies
        baseline_agents = [
            ('twap', TWAPPolicy(PARENT_ORDER_SIZE, len(market_data))),
            ('vwap', VWAPPolicy(PARENT_ORDER_SIZE)),
            ('random', RandomPolicy(env.action_space))
        ]
        for baseline_name, baseline_agent in baseline_agents:
            try:
                if hasattr(baseline_agent, 'reset'):
                    baseline_agent.reset()
                metrics = run_episode(env, baseline_agent, baseline_name)
                results.append({
                    'date': date_str,
                    'symbol': SYMBOL,
                    'model': baseline_name,
                    'model_type': 'Baseline',
                    **metrics
                })
            except Exception as e:
                logger.error(f"[FAILED] Baseline {baseline_name}: {e}")

        logger.info("")

    if not results:
        logger.error("")
        logger.error("[FAILED] No results collected. Validation failed.")
        logger.error("Check:")
        logger.error("  - Alpaca API credentials")
        logger.error("  - Date range (must be historical data)")
        logger.error("  - Market hours (9:30-16:00 ET)")
        return

    results_df = pd.DataFrame(results)
    action_dist_df = pd.json_normalize(results_df['action_distribution'])
    action_dist_df.columns = ['actions_' + col for col in action_dist_df.columns]
    results_df = pd.concat([results_df.drop('action_distribution', axis=1), action_dist_df], axis=1)
    output_path = Path("results/validation_results.csv")
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)

    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved: {output_path}")
    logger.info(f"Total episodes: {len(results_df)}")
    logger.info(f"Models tested: {results_df['model'].nunique()}")
    logger.info(f"Trading days: {results_df['date'].nunique()}")
    logger.info("=" * 80)
    logger.info("")

    print("")
    print("=" * 80)
    print("SUMMARY STATISTICS (Mean +/- Std)")
    print("=" * 80)
    print("")
    summary = results_df.groupby(['model', 'model_type']).agg({
        'slippage_bps': ['mean', 'std', 'min', 'max'],
        'exec_time_steps': ['mean', 'std'],
        'completion_rate': ['mean', 'min']
    }).round(2)
    print(summary)
    print("")
    print("=" * 80)
    best_model = results_df.groupby('model')['slippage_bps'].mean().idxmin()
    best_slippage = results_df.groupby('model')['slippage_bps'].mean().min()
    print(f"")
    print(f"BEST MODEL: {best_model} ({best_slippage:.2f} bps average slippage)")
    print("=" * 80)
    print("")

if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    try:
        main()
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Validation interrupted by user")
    except Exception as e:
        logger.error("")
        logger.error(f"Fatal error: {e}", exc_info=True)
