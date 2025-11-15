"""
LOAD TRAINED RL MODELS FOR EXECUTION AGENTS 
"""
import logging 
from pathlib import Path
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)

class AgentWrapper:
    """Wrapper to standardise interface across different RL models"""
    
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
    
    def predict(self, observation: np.ndarray, deterministic: bool = True):
        return self.model.predict(observation, deterministic=deterministic)

def load_trained_models(models_dir: str = "models/") -> Dict[str, AgentWrapper]:
    """Load DQN and PPO models"""
    models_path = Path(models_dir)
    agents = {}
    
    if not models_path.exists():
        raise RuntimeError(f"Models directory does not exist: {models_path}")
    
    model_configs = {
        "dqn_guided_real_data": {
            "path": models_path / "dqn_guided_real_data.zip",
            "type": "dqn"
        },
        "ppo_strategic_real_data": {
            "path": models_path / "ppo_strategic_real_data_v2.zip",
            "type": "ppo"
        },
    }
    
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError:
        logger.error("stable-baselines3 not installed")
        return agents
    
    for model_name, config in model_configs.items():
        model_file = config["path"]
        model_type = config["type"]
        
        try:
            if not model_file.exists():
                logger.warning(f"Model file not found: {model_file}")
                continue
            
            if model_type == "dqn":
                model = DQN.load(str(model_file))
            elif model_type == "ppo":
                model = PPO.load(str(model_file))
            else:
                continue
            
            agents[model_name] = AgentWrapper(model, model_name)
            logger.info(f"[OK] Loaded {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
    
    if agents:
        logger.info(f"\n{'='*60}")
        logger.info(f"Loaded {len(agents)} models:")
        for name in agents.keys():
            logger.info(f"  - {name}")
        logger.info(f"{'='*60}\n")
    
    return agents
