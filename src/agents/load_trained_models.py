"""
LOAD TRAINED RL MODELS FOR EXECUTION AGENTS 
PROVIDE STANDARD INTERFACE TO ACCESS MODELS
"""
import logging 
from pathlib import Path
from typing import Dict
import numpy as np
logger = logging.getLogger(__name__)


class AgentWrapper:
    """ Wrapper to standardise inetface across different RL models
    Makes all models look like 'stable-baselines3' models """
    
    def __init__(
        self,
        model, 
        model_name: str,
    ):
        self.model = model
        self.model_name = model_name
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """ Standardised predict method """
        return self.model.predict(observation, deterministic=deterministic)
    
def load_trained_models(
    models_dir : str = "models/",
) -> Dict[str, AgentWrapper]:
    """Load trained RL models from file directory (models/)
    Return dictionary of AgentWrapper instances keyed by model name
    """
    models_path = Path(models_dir)
    agents = {}
    model_files = {
        "baseline_dqn": models_path / "dqn_baseline_final.zip",
        "tactical_dqn": models_path / "dqn_tactical_final.zip",
        "strategic_ppo": models_path / "ppo_strategic_final.zip",
    }    
    if not models_path.exists():
        raise RuntimeError(f"Models directory does not exist: {models_path}")

    

    
    # Define exact paths to model files
    model_configs = {
        "dqn_guided": {
            "path": models_path / "dqn_guided" / "dqn_guided.zip",
            "type": "dqn"
        },
        "ppo_strategic": {
            "path": models_path / "ppo_strategic" / "ppo_strategic_stage2_random_tactical.zip",
            "type": "ppo"
        }
    }
    
    # Import SB3 models
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError:
        logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3")
        return agents
    
    # Load each model
    for model_name, config in model_configs.items():
        model_file = config["path"]
        model_type = config["type"]
        
        try:
            if not model_file.exists():
                logger.warning(f" Model file not found: {model_file}")
                logger.warning(f"   Skipping {model_name}...")
                continue
            
            # Load based on type
            if model_type == "dqn":
                model = DQN.load(str(model_file))
            elif model_type == "ppo":
                model = PPO.load(str(model_file))
            else:
                logger.error(f"Unknown model type: {model_type}")
                continue
            
            # Wrap and store
            agents[model_name] = AgentWrapper(model, model_name)
            logger.info(f"✓ Loaded {model_name} from {model_file.name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load {model_name}: {e}")
            logger.error(f"   File: {model_file}")
    
    if not agents:
        raise RuntimeError("No models were loaded. Please check the models directory.")
    else:
        logger.info(f"\n{'='*60}")
        logger.info(f"Successfully loaded {len(agents)} RL models:")
        for name in agents.keys():
            logger.info(f"  - {name}")
        logger.info(f"{'='*60}\n")
    
    return agents

    
                