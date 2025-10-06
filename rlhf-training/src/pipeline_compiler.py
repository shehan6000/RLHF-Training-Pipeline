import logging
import tempfile
import os
from pathlib import Path
from kfp import compiler
from google_cloud_pipeline_components.preview.llm import rlhf_pipeline


class PipelineCompiler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compile_pipeline(
        self,
        output_path: str = None,
        pipeline_name: str = "rlhf_pipeline"
    ) -> str:
        """
        Compile the RLHF pipeline to YAML.
        
        Args:
            output_path: Path to save compiled pipeline
            pipeline_name: Name for the pipeline file
            
        Returns:
            Path to compiled pipeline YAML
        """
        try:
            if output_path is None:
                # Create temporary directory if no output path specified
                temp_dir = tempfile.mkdtemp()
                output_path = os.path.join(temp_dir, f"{pipeline_name}.yaml")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Compiling RLHF pipeline to: {output_path}")
            
            # Compile the pipeline
            compiler.Compiler().compile(
                pipeline_func=rlhf_pipeline,
                package_path=output_path
            )
            
            # Verify compilation was successful
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Pipeline compilation failed: {output_path} not created")
            
            file_size = os.path.getsize(output_path)
            self.logger.info(f"Pipeline compiled successfully. File size: {file_size} bytes")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Pipeline compilation failed: {e}")
            raise
    
    def validate_pipeline_yaml(self, pipeline_path: str) -> bool:
        """
        Validate the compiled pipeline YAML file.
        
        Args:
            pipeline_path: Path to pipeline YAML file
            
        Returns:
            True if validation successful
        """
        try:
            if not os.path.exists(pipeline_path):
                self.logger.error(f"Pipeline file not found: {pipeline_path}")
                return False
            
            with open(pipeline_path, 'r') as file:
                content = file.read()
            
            # Basic validation checks
            required_strings = [
                "apiVersion: argoproj.io/v1alpha1",
                "kind: Workflow",
                "rlhf-pipeline"
            ]
            
            for required in required_strings:
                if required not in content:
                    self.logger.error(f"Pipeline validation failed: missing '{required}'")
                    return False
            
            self.logger.info("Pipeline YAML validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline validation error: {e}")
            return False