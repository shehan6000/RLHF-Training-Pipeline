import logging
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
from typing import Tuple, Optional


class AuthManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def authenticate(
        self,
        project_id: Optional[str] = None
    ) -> Tuple[any, str, str]:
        """
        Authenticate with Google Cloud and get credentials.
        
        Args:
            project_id: Optional project ID override
            
        Returns:
            Tuple of (credentials, project_id, staging_bucket)
        """
        try:
            # Get default credentials
            credentials, default_project = default()
            
            # Use provided project ID or default
            final_project_id = project_id or default_project
            if not final_project_id:
                raise ValueError("No project ID specified and no default project found")
            
            # Generate staging bucket name
            staging_bucket = f"gs://{final_project_id}-rlhf-staging"
            
            self.logger.info(f"Authentication successful for project: {final_project_id}")
            
            return credentials, final_project_id, staging_bucket
            
        except DefaultCredentialsError as e:
            self.logger.error(
                "Default credentials not found. Please ensure:\n"
                "1. You are running on Google Cloud (GCE, GKE, Cloud Run)\n"
                "2. Or set GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                "3. Or run 'gcloud auth application-default login'"
            )
            raise
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            raise