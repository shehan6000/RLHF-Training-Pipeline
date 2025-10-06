import logging
import time
from typing import Dict, Any, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from .config import RLHFConfig


class VertexAIClient:
    def __init__(self, config: RLHFConfig, credentials=None):
        self.config = config
        self.credentials = credentials
        self.logger = logging.getLogger(__name__)
        self._initialize_aiplatform()
    
    def _initialize_aiplatform(self):
        """Initialize the AI Platform client."""
        try:
            aiplatform.init(
                project=self.config.vertex_ai.project_id,
                location=self.config.vertex_ai.region,
                credentials=self.credentials,
                staging_bucket=self.config.vertex_ai.staging_bucket
            )
            self.logger.info(
                f"AI Platform initialized: project={self.config.vertex_ai.project_id}, "
                f"region={self.config.vertex_ai.region}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Platform: {e}")
            raise
    
    def create_pipeline_job(
        self,
        template_path: str,
        parameter_values: Dict[str, Any],
        enable_caching: bool = False
    ) -> pipeline_jobs.PipelineJob:
        """
        Create a Vertex AI Pipeline job.
        
        Args:
            template_path: Path to pipeline template
            parameter_values: Pipeline parameter values
            enable_caching: Whether to enable pipeline caching
            
        Returns:
            PipelineJob object
        """
        try:
            self.logger.info(f"Creating pipeline job: {self.config.vertex_ai.pipeline_display_name}")
            
            job = aiplatform.PipelineJob(
                display_name=self.config.vertex_ai.pipeline_display_name,
                template_path=template_path,
                parameter_values=parameter_values,
                enable_caching=enable_caching
            )
            
            self.logger.info("Pipeline job created successfully")
            return job
            
        except Exception as e:
            self.logger.error(f"Failed to create pipeline job: {e}")
            raise
    
    def run_pipeline_job(
        self,
        job: pipeline_jobs.PipelineJob,
        wait_for_completion: bool = False,
        timeout: int = 86400  # 24 hours
    ) -> Dict[str, Any]:
        """
        Run a Vertex AI Pipeline job.
        
        Args:
            job: PipelineJob to run
            wait_for_completion: Whether to wait for job completion
            timeout: Timeout in seconds for waiting
            
        Returns:
            Dictionary with job status and information
        """
        try:
            self.logger.info("Starting pipeline job execution")
            
            # Start the job
            job.run()
            
            result = {
                "job_id": job.name,
                "display_name": job.display_name,
                "state": job.state,
                "start_time": job.start_time,
                "success": True
            }
            
            if wait_for_completion:
                self.logger.info("Waiting for pipeline job completion...")
                job.wait(timeout=timeout)
                
                result.update({
                    "end_time": job.end_time,
                    "state": job.state,
                    "error": job.error
                })
                
                if job.state == aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED:
                    self.logger.info("Pipeline job completed successfully")
                else:
                    self.logger.error(f"Pipeline job failed: {job.error}")
                    result["success"] = False
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline job execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def monitor_pipeline_job(self, job_id: str) -> Dict[str, Any]:
        """
        Monitor an existing pipeline job.
        
        Args:
            job_id: ID of the pipeline job to monitor
            
        Returns:
            Dictionary with current job status
        """
        try:
            # Get the job by ID
            job = aiplatform.PipelineJob.get(resource_name=job_id)
            
            return {
                "job_id": job.name,
                "display_name": job.display_name,
                "state": job.state,
                "start_time": job.start_time,
                "end_time": job.end_time,
                "error": job.error,
                "success": job.state == aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED
            }
            
        except Exception as e:
            self.logger.error(f"Failed to monitor pipeline job {job_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def cancel_pipeline_job(self, job_id: str) -> bool:
        """
        Cancel a running pipeline job.
        
        Args:
            job_id: ID of the pipeline job to cancel
            
        Returns:
            True if cancellation was successful
        """
        try:
            job = aiplatform.PipelineJob.get(resource_name=job_id)
            job.cancel()
            self.logger.info(f"Pipeline job {job_id} cancellation requested")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel pipeline job {job_id}: {e}")
            return False