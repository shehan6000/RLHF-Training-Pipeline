#!/usr/bin/env python3
"""
RLHF Training Pipeline - Main Entry Point
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import ConfigManager
from src.training_calculator import TrainingCalculator
from src.pipeline_compiler import PipelineCompiler
from src.vertex_ai_client import VertexAIClient
from src.monitoring import PipelineMonitor
from src.utils.auth import AuthManager


class RLHFTrainingPipeline:
    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
        self.rlhf_config = self.config_manager.get_rlhf_config()
        self.setup_logging()
        
        self.training_calculator = TrainingCalculator(
            batch_size=self.rlhf_config.training.batch_size
        )
        self.pipeline_compiler = PipelineCompiler()
        self.auth_manager = AuthManager()
        
        # Authenticate and initialize Vertex AI client
        credentials, project_id, staging_bucket = self.auth_manager.authenticate(
            self.rlhf_config.vertex_ai.project_id
        )
        self.vertex_ai_client = VertexAIClient(self.rlhf_config, credentials)
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('rlhf_pipeline.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_optimal_training_steps(self):
        """Calculate optimal training steps based on dataset sizes."""
        self.logger.info("Calculating optimal training steps...")
        
        # Validate parameters
        is_valid, error_msg = self.training_calculator.validate_training_parameters(
            preference_dataset_size=self.rlhf_config.dataset.preference_size,
            prompt_dataset_size=self.rlhf_config.dataset.prompt_size,
            reward_num_epochs=30,  # Default values
            reinforcement_learning_num_epochs=10
        )
        
        if not is_valid:
            self.logger.warning(f"Parameter validation warnings: {error_msg}")
        
        # Get recommended epochs
        reward_epochs, rl_epochs = self.training_calculator.recommend_epochs(
            self.rlhf_config.dataset.preference_size,
            self.rlhf_config.dataset.prompt_size
        )
        
        self.logger.info(f"Recommended epochs: Reward={reward_epochs}, RL={rl_epochs}")
        
        # Calculate steps
        training_steps = self.training_calculator.calculate_training_steps(
            preference_dataset_size=self.rlhf_config.dataset.preference_size,
            prompt_dataset_size=self.rlhf_config.dataset.prompt_size,
            reward_num_epochs=reward_epochs,
            reinforcement_learning_num_epochs=rl_epochs
        )
        
        return training_steps
    
    def run_pipeline(self, wait_for_completion: bool = False):
        """Run the complete RLHF training pipeline."""
        try:
            self.logger.info("Starting RLHF Training Pipeline")
            
            # Step 1: Compile pipeline
            self.logger.info("Step 1: Compiling pipeline...")
            pipeline_path = self.pipeline_compiler.compile_pipeline()
            
            if not self.pipeline_compiler.validate_pipeline_yaml(pipeline_path):
                raise ValueError("Pipeline compilation validation failed")
            
            # Step 2: Get parameter values
            self.logger.info("Step 2: Preparing parameters...")
            parameter_values = self.config_manager.get_parameter_values()
            
            # Step 3: Create and run pipeline job
            self.logger.info("Step 3: Creating pipeline job...")
            job = self.vertex_ai_client.create_pipeline_job(
                template_path=pipeline_path,
                parameter_values=parameter_values
            )
            
            # Step 4: Run the job
            self.logger.info("Step 4: Running pipeline job...")
            result = self.vertex_ai_client.run_pipeline_job(
                job=job,
                wait_for_completion=wait_for_completion
            )
            
            if result["success"]:
                self.logger.info(f"Pipeline job started successfully: {result['job_id']}")
                
                if wait_for_completion:
                    # Monitor the job
                    monitor = PipelineMonitor(self.vertex_ai_client)
                    final_status = monitor.monitor_job_with_updates(result["job_id"])
                    
                    # Generate report
                    report = monitor.generate_monitoring_report(final_status)
                    self.logger.info(f"Final job status:\n{report}")
                    
                    return final_status
                else:
                    return result
            else:
                self.logger.error(f"Pipeline job failed to start: {result.get('error')}")
                return result
                
        except Exception as e:
            self.logger.error(f"RLHF pipeline execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def monitor_existing_job(self, job_id: str):
        """Monitor an existing pipeline job."""
        try:
            monitor = PipelineMonitor(self.vertex_ai_client)
            status = monitor.monitor_job_with_updates(job_id)
            report = monitor.generate_monitoring_report(status)
            print(report)
            return status
        except Exception as e:
            self.logger.error(f"Failed to monitor job {job_id}: {e}")
            return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="RLHF Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for pipeline completion"
    )
    parser.add_argument(
        "--monitor",
        type=str,
        help="Monitor existing job by ID"
    )
    parser.add_argument(
        "--calculate-steps",
        action="store_true",
        help="Calculate optimal training steps and exit"
    )
    
    args = parser.parse_args()
    
    try:
        pipeline = RLHFTrainingPipeline(args.config)
        
        if args.monitor:
            pipeline.monitor_existing_job(args.monitor)
        elif args.calculate_steps:
            steps = pipeline.calculate_optimal_training_steps()
            print(f"Optimal training steps: {steps}")
        else:
            result = pipeline.run_pipeline(wait_for_completion=args.wait)
            if not result.get("success"):
                sys.exit(1)
                
    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()