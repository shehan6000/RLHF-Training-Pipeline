import math
import logging
from typing import Tuple
from dataclasses import dataclass


@dataclass
class TrainingSteps:
    reward_model_steps: int
    reinforcement_learning_steps: int


class TrainingCalculator:
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def calculate_training_steps(
        self,
        preference_dataset_size: int,
        prompt_dataset_size: int,
        reward_num_epochs: int = 30,
        reinforcement_learning_num_epochs: int = 10
    ) -> TrainingSteps:
        """
        Calculate training steps for reward model and RL training.
        
        Args:
            preference_dataset_size: Size of preference dataset
            prompt_dataset_size: Size of prompt dataset
            reward_num_epochs: Number of epochs for reward model training
            reinforcement_learning_num_epochs: Number of epochs for RL training
            
        Returns:
            TrainingSteps object with calculated steps
        """
        try:
            # Calculate reward model training steps
            reward_steps_per_epoch = math.ceil(preference_dataset_size / self.batch_size)
            reward_model_steps = reward_steps_per_epoch * reward_num_epochs
            
            # Calculate RL training steps
            rl_steps_per_epoch = math.ceil(prompt_dataset_size / self.batch_size)
            reinforcement_learning_steps = rl_steps_per_epoch * reinforcement_learning_num_epochs
            
            self.logger.info(
                f"Calculated training steps: "
                f"Reward Model={reward_model_steps}, "
                f"RL={reinforcement_learning_steps}"
            )
            
            return TrainingSteps(
                reward_model_steps=reward_model_steps,
                reinforcement_learning_steps=reinforcement_learning_steps
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating training steps: {e}")
            raise
    
    def validate_training_parameters(
        self,
        preference_dataset_size: int,
        prompt_dataset_size: int,
        reward_num_epochs: int,
        reinforcement_learning_num_epochs: int
    ) -> Tuple[bool, str]:
        """
        Validate training parameters for reasonable values.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Validate dataset sizes
        if preference_dataset_size < 100:
            errors.append("Preference dataset too small (min 100 samples)")
        if prompt_dataset_size < 100:
            errors.append("Prompt dataset too small (min 100 samples)")
        
        # Validate epoch counts
        if reward_num_epochs < 1 or reward_num_epochs > 100:
            errors.append("Reward epochs should be between 1 and 100")
        if reinforcement_learning_num_epochs < 1 or reinforcement_learning_num_epochs > 50:
            errors.append("RL epochs should be between 1 and 50")
        
        # Check for potential overfitting
        if preference_dataset_size < 1000 and reward_num_epochs > 50:
            errors.append("High epoch count with small dataset may cause overfitting")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, "Parameters are valid"
    
    def recommend_epochs(
        self,
        preference_dataset_size: int,
        prompt_dataset_size: int
    ) -> Tuple[int, int]:
        """
        Recommend epoch counts based on dataset sizes.
        
        Returns:
            Tuple of (recommended_reward_epochs, recommended_rl_epochs)
        """
        # Base recommendations on dataset size
        if preference_dataset_size >= 10000:
            reward_epochs = 20
        elif preference_dataset_size >= 5000:
            reward_epochs = 25
        elif preference_dataset_size >= 1000:
            reward_epochs = 30
        else:
            reward_epochs = 35  # More epochs for small datasets
        
        if prompt_dataset_size >= 50000:
            rl_epochs = 8
        elif prompt_dataset_size >= 20000:
            rl_epochs = 10
        elif prompt_dataset_size >= 5000:
            rl_epochs = 12
        else:
            rl_epochs = 15
        
        return reward_epochs, rl_epochs