import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    LLAMA_2_7B = "llama-2-7b"
    LLAMA_2_13B = "llama-2-13b"
    T5X = "t5x"
    TEXT_BISON = "text-bison@001"


@dataclass
class DatasetConfig:
    preference_dataset: str
    prompt_dataset: str
    eval_dataset: str
    preference_size: int
    prompt_size: int


@dataclass
class TrainingConfig:
    large_model_reference: ModelType
    reward_model_train_steps: int
    reinforcement_learning_train_steps: int
    reward_model_learning_rate_multiplier: float
    reinforcement_learning_rate_multiplier: float
    kl_coeff: float
    instruction: str
    batch_size: int = 64


@dataclass
class VertexAIConfig:
    project_id: str
    region: str
    staging_bucket: str
    pipeline_display_name: str


@dataclass
class RLHFConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    vertex_ai: VertexAIConfig


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_data = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as file:
                self.config_data = yaml.safe_load(file)
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_sections = ['dataset', 'training', 'vertex_ai']
        for section in required_sections:
            if section not in self.config_data:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get_rlhf_config(self) -> RLHFConfig:
        """Get complete RLHF configuration."""
        dataset_config = DatasetConfig(
            preference_dataset=self.config_data['dataset']['preference_dataset'],
            prompt_dataset=self.config_data['dataset']['prompt_dataset'],
            eval_dataset=self.config_data['dataset']['eval_dataset'],
            preference_size=self.config_data['dataset']['preference_size'],
            prompt_size=self.config_data['dataset']['prompt_size']
        )
        
        training_config = TrainingConfig(
            large_model_reference=ModelType(self.config_data['training']['large_model_reference']),
            reward_model_train_steps=self.config_data['training']['reward_model_train_steps'],
            reinforcement_learning_train_steps=self.config_data['training']['reinforcement_learning_train_steps'],
            reward_model_learning_rate_multiplier=self.config_data['training']['reward_model_learning_rate_multiplier'],
            reinforcement_learning_rate_multiplier=self.config_data['training']['reinforcement_learning_rate_multiplier'],
            kl_coeff=self.config_data['training']['kl_coeff'],
            instruction=self.config_data['training']['instruction'],
            batch_size=self.config_data['training'].get('batch_size', 64)
        )
        
        vertex_ai_config = VertexAIConfig(
            project_id=self.config_data['vertex_ai']['project_id'],
            region=self.config_data['vertex_ai']['region'],
            staging_bucket=self.config_data['vertex_ai']['staging_bucket'],
            pipeline_display_name=self.config_data['vertex_ai']['pipeline_display_name']
        )
        
        return RLHFConfig(
            dataset=dataset_config,
            training=training_config,
            vertex_ai=vertex_ai_config
        )
    
    def get_parameter_values(self) -> Dict[str, Any]:
        """Get parameter values for Vertex AI pipeline."""
        config = self.get_rlhf_config()
        
        return {
            "preference_dataset": config.dataset.preference_dataset,
            "prompt_dataset": config.dataset.prompt_dataset,
            "eval_dataset": config.dataset.eval_dataset,
            "large_model_reference": config.training.large_model_reference.value,
            "reward_model_train_steps": config.training.reward_model_train_steps,
            "reinforcement_learning_train_steps": config.training.reinforcement_learning_train_steps,
            "reward_model_learning_rate_multiplier": config.training.reward_model_learning_rate_multiplier,
            "reinforcement_learning_rate_multiplier": config.training.reinforcement_learning_rate_multiplier,
            "kl_coeff": config.training.kl_coeff,
            "instruction": config.training.instruction
        }