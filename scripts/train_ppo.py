"""
RETRAIN PPO FOR HIERARCHICAL EXECUTION
- Uses FULL 9-DIM observations (not wrapped)
- 500k timesteps (5x more than before)
- Better reward shaping for strategic decisions
- Saves checkpoint automatically
"""

import sys
sys.path.append('.')
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
import pandas as pd

from src.data.alpaca_loader import AlpacaDataLoader
from src.environments.real_market_env import RealMarketEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_ppo_real_data.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# TRAINING CONFIG
SYMBOLS = ["SPY"]   
START_DATE = "2025-07-01"
END_DATE = "2025-09-30"
PARENT_ORDER_SIZE = 1000
TOTAL_TIMESTEPS = 500_000  
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 128

def make_env():
    """Create training environment"""
    def _init():
        return RealMarketEnv(
            market_data=None,  # Will be set in loop
            parent_order_size=PARENT_ORDER_SIZE
        )
    return _init

def main():
    logger.info("=" * 80)
    logger.info("PPO RETRAINING FOR HIERARCHICAL EXECUTION - FULL 9-DIM OBSERVATIONS")
    logger.info("=" * 80)
    logger.info(f"Training on: {SYMBOLS}")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info("=" * 80)
    logger.info("")
    
    # Initialize loader
    try:
        loader = AlpacaDataLoader()
        logger.info("[OK] Alpaca data loader initialized")
    except Exception as e:
        logger.error(f"[FAILED] Data loader: {e}")
        return
    
    # Generate date range
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    date_range = pd.date_range(start, end, freq='B')
    
    logger.info(f"Loading {len(date_range)} trading days of data...")
    logger.info("")
    
    # Collect all market data
    all_market_data = []
    for symbol in SYMBOLS:
        for date in date_range:
            date_str = date.strftime("%Y-%m-%d")
            try:
                market_data = loader.download_bars(
                    symbol=symbol,
                    start_date=date_str,
                    end_date=date_str,
                    timeframe='1Min'
                )
                if not market_data.empty:
                    all_market_data.append(market_data)
                    logger.info(f"[OK] {symbol} {date_str}: {len(market_data)} bars")
            except Exception as e:
                logger.warning(f"[SKIP] {symbol} {date_str}: {e}")
    
    if not all_market_data:
        logger.error("[FAILED] No market data loaded!")
        return
    
    combined_market_data = pd.concat(all_market_data, ignore_index=True)
    logger.info(f"\n[OK] Total bars loaded: {len(combined_market_data):,}")
    logger.info("")
    
    # Create environment with FULL 9-DIM observations (NO WRAPPER)
    logger.info("=" * 80)
    logger.info("INITIALIZING TRAINING ENVIRONMENT")
    logger.info("=" * 80)
    logger.info("")
    
    env = RealMarketEnv(
        market_data=combined_market_data,
        parent_order_size=PARENT_ORDER_SIZE
    )
    
    logger.info(f"Environment observation space: {env.observation_space}")
    logger.info(f"Environment action space: {env.action_space}")
    logger.info("")
    
    # Create PPO model
    logger.info("=" * 80)
    logger.info("CREATING PPO MODEL")
    logger.info("=" * 80)
    logger.info("")
    
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="logs/tensorboard/ppo_retraining"
    )
    
    logger.info(f"PPO Model created:")
    logger.info(f"  Policy: MlpPolicy")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  N steps: {N_STEPS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info("")
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="models/checkpoints/ppo_retraining",
        name_prefix="ppo_strategic_real_data_checkpoint"
    )
    
    # Train model
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            log_interval=10
        )
        logger.info("")
        logger.info("[OK] Training completed!")
        
    except KeyboardInterrupt:
        logger.info("[INTERRUPTED] Training stopped by user")
    except Exception as e:
        logger.error(f"[FAILED] Training error: {e}", exc_info=True)
        return
    
    # Save final model
    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)
    logger.info("")
    
    output_path = Path("models/ppo_strategic_real_data_v2.zip")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        model.save(str(output_path))
        logger.info(f"[OK] Model saved to: {output_path}")
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"NEW MODEL: {output_path}")
        logger.info("Update load_trained_models.py to load from:")
        logger.info('  "path": models_path / "ppo_strategic_real_data_v2.zip"')
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"[FAILED] Model save: {e}")
        return

# ENTRY POINT
if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    Path("logs/tensorboard").mkdir(exist_ok=True)
    Path("models/checkpoints/ppo_retraining").mkdir(parents=True, exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
