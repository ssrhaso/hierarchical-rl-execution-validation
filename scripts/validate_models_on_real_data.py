"""
RL OPTIMAL EXECUTION - VALIDATION ON EXTENDED REAL MARKET DATA

Validates DQN and PPO on 6 months of real Alpaca market data.
Includes progress bar for better visibility.

Architecture:
    - DQN: Minute-level execution timing optimization
    - PPO: Market-aware decision making
    - Baselines: TWAP, VWAP, Random

Validation:
    - Data: 6 months (Jan-Jun 2025)
    - Symbols: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, ASML, ORCL (9 total symbols)
    - Parent Order Size: 1,000 shares
    - Episodes: ~2,500+ episodes
"""
import sys
sys.path.append('.')
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.data.alpaca_loader import AlpacaDataLoader
from src.environments.real_market_env import RealMarketEnv
from load_trained_models import load_trained_models
from src.baselines.twap import TWAPPolicy
from src.baselines.vwap import VWAPPolicy

# LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/validation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# RANDOM POLICY
class RandomPolicy:
    """Random baseline policy."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        return self.action_space.sample(), None

    def reset(self):
        pass

# EPISODE RUNNER
def run_episode(env, agent, agent_name: str, max_steps: int = 1000) -> dict:
    """Run single validation episode."""
    
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    actions_taken = []
    slippage_values = []

    while not done and step < max_steps:
        try:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            actions_taken.append(int(action))
            slippage_values.append(info.get('slippage_bps', 0.0))
            done = terminated or truncated
            step += 1
            
        except Exception as e:
            logger.error(f"Error during step {step} for {agent_name}: {e}")
            break

    inventory_remaining = info.get('inventory', 0)
    completion_rate = (1.0 - inventory_remaining / 1000.0) * 100.0
    avg_slippage = np.nanmean(slippage_values) if slippage_values else 0.0
    
    # METRICS COLLECTION
    metrics = {
        'agent_name': agent_name,
        'total_reward': total_reward,
        'slippage_bps': avg_slippage,
        'exec_time_steps': step,
        'completion_rate': completion_rate,
        'inventory_remaining': inventory_remaining,
        'shares_executed': 1000 - inventory_remaining,
        'action_distribution': {
            'action_0': actions_taken.count(0),
            'action_1': actions_taken.count(1),
            'action_2': actions_taken.count(2)
        }
    }
    
    return metrics


# MAIN VALIDATION
def main():
    """Main validation pipeline with extended data."""
    
    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "ASML", "ORCL" ]
    START_DATE = "2025-01-02"          
    END_DATE = "2025-06-30"
    PARENT_ORDER_SIZE = 1000
    
    logger.info("=" * 80)
    logger.info("RL OPTIMAL EXECUTION - EXTENDED VALIDATION (6 MONTHS)")
    logger.info("=" * 80)
    logger.info(f"Symbols: {', '.join(SYMBOLS)}")
    logger.info(f"Date Range: {START_DATE} to {END_DATE} (6 months)")
    logger.info(f"Parent Order Size: {PARENT_ORDER_SIZE} shares")
    logger.info(f"Expected Episodes: ~2,500+")
    logger.info("=" * 80)
    logger.info("")
    
    # Initialise Alpaca loader
    try:
        loader = AlpacaDataLoader()
        logger.info("[OK] Alpaca data loader initialized")
    except Exception as e:
        logger.error(f"[FAILED] Data loader: {e}")
        logger.error("Check .env file for ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return
    
    # Load RL models
    try:
        rl_agents = load_trained_models(models_dir="models/")
        if not rl_agents:
            logger.error("[ERROR] No RL models loaded!")
            return
        
        logger.info(f"[OK] Loaded {len(rl_agents)} RL models:")
        for name in rl_agents.keys():
            logger.info(f"      - {name}")
    except Exception as e:
        logger.error(f"[FAILED] Model loading: {e}")
        return
    
    logger.info("")
    
    # Generate date range
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    date_range = pd.date_range(start, end, freq='B')
    
    results = []
    total_days = len(date_range) * len(SYMBOLS)
    
    logger.info(f"Testing on {len(date_range)} trading days")
    logger.info(f"Total episodes to run: ~{total_days * 5} (5 models per day)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION STARTING")
    logger.info("=" * 80)
    logger.info("")
    
    
    # MAIN LOOP FOR VALIDATION
    with tqdm(total=total_days, desc="Overall Progress", unit="day") as pbar_overall:
        # Iterate symbols
        for symbol in SYMBOLS:
            logger.info(f"\n{'='*80}")
            logger.info(f"SYMBOL: {symbol}")
            logger.info(f"{'='*80}\n")
            
            # Iterate dates with progress bar
            with tqdm(total=len(date_range), desc=f"{symbol}", unit="day", leave=True) as pbar_symbol:
                for date in date_range:
                    date_str = date.strftime("%Y-%m-%d")
                    
                    try:
                        market_data = loader.download_bars(
                            symbol=symbol,
                            start_date=date_str,
                            end_date=date_str,
                            timeframe='1Min'
                        )
                        
                        if market_data.empty:
                            pbar_symbol.update(1)
                            pbar_overall.update(1)
                            continue
                        
                    except Exception as e:
                        logger.warning(f"[SKIP] {symbol} {date_str}: {e}")
                        pbar_symbol.update(1)
                        pbar_overall.update(1)
                        continue
                    
                    # Create environment
                    try:
                        env = RealMarketEnv(
                            market_data=market_data,
                            parent_order_size=PARENT_ORDER_SIZE
                        )
                    except Exception as e:
                        logger.error(f"[FAILED] Environment creation: {e}")
                        pbar_symbol.update(1)
                        pbar_overall.update(1)
                        continue
                    
                    # Test DQN
                    if 'dqn_guided_real_data' in rl_agents:
                        try:
                            metrics = run_episode(env, rl_agents['dqn_guided_real_data'], 'dqn_guided_real_data')
                            results.append({
                                'date': date_str,
                                'symbol': symbol,
                                'model': 'dqn_guided',
                                'model_type': 'RL (DQN)',
                                **metrics
                            })
                        except Exception as e:
                            logger.error(f"[FAILED] DQN: {e}")
                    
                    # Test PPO
                    if 'ppo_strategic_real_data' in rl_agents:
                        try:
                            metrics = run_episode(env, rl_agents['ppo_strategic_real_data'], 'ppo_strategic_real_data')
                            results.append({
                                'date': date_str,
                                'symbol': symbol,
                                'model': 'ppo_strategic',
                                'model_type': 'RL (PPO)',
                                **metrics
                            })
                        except Exception as e:
                            logger.error(f"[FAILED] PPO: {e}")
                    
                    # Test Baselines
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
                                'symbol': symbol,
                                'model': baseline_name,
                                'model_type': 'Baseline',
                                **metrics
                            })
                        except Exception as e:
                            logger.error(f"[FAILED] {baseline_name}: {e}")
                    
                    pbar_symbol.update(1)
                    pbar_overall.update(1)
    
    
    
    # SAVE RESULTS
    if not results:
        logger.error("[FAILED] No results collected")
        return
    results_df = pd.DataFrame(results)
    
    # Flatten action distribution
    action_dist_df = pd.json_normalize(results_df['action_distribution'])
    action_dist_df.columns = ['actions_' + col for col in action_dist_df.columns]
    results_df = pd.concat(
        [results_df.drop('action_distribution', axis=1), action_dist_df],
        axis=1
    )
    
    # Save
    output_path = Path("results/validation_results.csv")
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results: {output_path}")
    logger.info(f"Episodes: {len(results_df)}")
    logger.info(f"Models: {results_df['model'].nunique()}")
    logger.info(f"Symbols: {results_df['symbol'].nunique()}")
    logger.info(f"Trading Days: {results_df['date'].nunique()}")
    logger.info("=" * 80)
    logger.info("")
    
    
    # SUMMARY
    print("")
    print("=" * 80)
    print("PERFORMANCE SUMMARY (6-MONTH VALIDATION)")
    print("=" * 80)
    print("")
    
    summary = results_df.groupby(['model', 'model_type']).agg({
        'slippage_bps': ['mean', 'std', 'min', 'max', 'count'],
        'exec_time_steps': ['mean', 'std'],
        'completion_rate': ['mean']
    }).round(2)
    
    print(summary)
    print("")
    print("=" * 80)
    
    best_model = results_df.groupby('model')['slippage_bps'].mean().idxmin()
    best_slippage = results_df.groupby('model')['slippage_bps'].mean().min()
    vwap_slippage = results_df[results_df['model'] == 'vwap']['slippage_bps'].mean()
    
    print(f"BEST MODEL: {best_model} ({best_slippage:.2f} bps)")
    if vwap_slippage != 0:
        improvement = abs((vwap_slippage - best_slippage) / vwap_slippage) * 100
        print(f"vs VWAP: {improvement:.1f}% better")
    print("=" * 80)
    print("")


# ENTRY POINT
if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)