import logging
import time
from typing import Dict, Any, List
from datetime import datetime
import json


class PipelineMonitor:
    def __init__(self, vertex_ai_client, check_interval: int = 300):  # 5 minutes
        self.vertex_ai_client = vertex_ai_client
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
    
    def monitor_job_with_updates(
        self,
        job_id: str,
        timeout: int = 86400,  # 24 hours
        callback=None
    ) -> Dict[str, Any]:
        """
        Monitor a pipeline job with periodic updates.
        
        Args:
            job_id: Job ID to monitor
            timeout: Maximum monitoring time in seconds
            callback: Optional callback function for status updates
            
        Returns:
            Final job status
        """
        start_time = time.time()
        last_state = None
        
        self.logger.info(f"Starting monitoring for job: {job_id}")
        
        while time.time() - start_time < timeout:
            try:
                status = self.vertex_ai_client.monitor_pipeline_job(job_id)
                current_state = status.get("state", "UNKNOWN")
                
                # Log state changes
                if current_state != last_state:
                    self.logger.info(f"Job state changed: {last_state} -> {current_state}")
                    last_state = current_state
                
                # Call callback if provided
                if callback:
                    callback(status)
                
                # Check if job is complete
                if status["success"] or status.get("error"):
                    self.logger.info(f"Job completed with state: {current_state}")
                    return status
                
                # Wait before next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error during job monitoring: {e}")
                return {
                    "success": False,
                    "error": f"Monitoring error: {str(e)}"
                }
        
        # Timeout reached
        self.logger.warning(f"Job monitoring timeout reached after {timeout} seconds")
        return {
            "success": False,
            "error": f"Monitoring timeout after {timeout} seconds",
            "state": "TIMEOUT"
        }
    
    def generate_monitoring_report(self, job_status: Dict[str, Any]) -> str:
        """
        Generate a human-readable monitoring report.
        
        Args:
            job_status: Job status dictionary
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "RLHF Pipeline Job Monitoring Report",
            "=" * 50,
            f"Job ID: {job_status.get('job_id', 'N/A')}",
            f"Display Name: {job_status.get('display_name', 'N/A')}",
            f"State: {job_status.get('state', 'N/A')}",
            f"Start Time: {job_status.get('start_time', 'N/A')}",
            f"End Time: {job_status.get('end_time', 'N/A')}",
        ]
        
        if job_status.get('error'):
            report_lines.append(f"Error: {job_status['error']}")
        
        report_lines.append(f"Success: {job_status.get('success', False)}")
        
        return "\n".join(report_lines)


class MetricsLogger:
    def __init__(self, log_file: str = "rlhf_metrics.jsonl"):
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
    
    def log_training_metrics(
        self,
        job_id: str,
        metrics: Dict[str, Any],
        stage: str
    ):
        """
        Log training metrics to JSONL file.
        
        Args:
            job_id: Pipeline job ID
            metrics: Dictionary of metrics
            stage: Training stage (reward_model, rl_training, etc.)
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": job_id,
            "stage": stage,
            "metrics": metrics
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            self.logger.debug(f"Logged metrics for {stage} stage")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")