# RLHF Training Pipeline 
## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Configuration](#configuration)
5. [Usage Guide](#usage-guide)
6. [Pipeline Components](#pipeline-components)
7. [Monitoring & Logging](#monitoring--logging)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [API Reference](#api-reference)

## Overview

The RLHF (Reinforcement Learning from Human Feedback) Training Pipeline is a production-ready system for fine-tuning large language models using human preference data. This implementation leverages Google Cloud's Vertex AI Pipelines and Kubeflow to provide a scalable, monitored, and reproducible training process.

### Key Features

- **Automated Pipeline**: End-to-end RLHF training workflow
- **Multiple Model Support**: Compatible with Llama-2, T5X, and other foundation models
- **Configurable Training**: Flexible hyperparameter configuration
- **Production Monitoring**: Real-time job monitoring and metrics tracking
- **Error Handling**: Comprehensive error handling and recovery mechanisms
- **Cost Optimization**: Efficient resource utilization and training step calculation

### Supported Models

- Llama-2-7B
- Llama-2-13B  
- T5X
- text-bison@001

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Configuration  │◄──►│   Pipeline        │◄──►│   Vertex AI      │
│     Manager      │    │   Compiler       │    │   Client        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Training        │    │   Monitoring &   │    │   Google Cloud  │
│ Calculator      │    │   Logging        │    │   Services      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow

1. **Configuration Loading**: YAML configuration files are parsed and validated
2. **Training Calculation**: Optimal training steps are calculated based on dataset sizes
3. **Pipeline Compilation**: RLHF pipeline is compiled to Kubeflow YAML format
4. **Job Submission**: Pipeline job is submitted to Vertex AI
5. **Execution & Monitoring**: Job execution is monitored with real-time updates
6. **Metrics Collection**: Training metrics are logged for analysis

### Component Interactions

```
User Input
    │
    ▼
Configuration Manager
    │
    ▼
Training Calculator
    │
    ▼
Pipeline Compiler
    │
    ▼
Vertex AI Client
    │
    ▼
Pipeline Monitor
    │
    ▼
Metrics Logger
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Google Cloud Project with billing enabled
- Vertex AI API enabled
- Storage bucket for staging files
- Appropriate IAM permissions

### Required IAM Permissions

```yaml
- aiplatform.pipelineJobs.create
- aiplatform.pipelineJobs.get
- aiplatform.pipelineJobs.list
- aiplatform.pipelineJobs.cancel
- storage.objects.create
- storage.objects.get
- storage.objects.list
```

### Step-by-Step Installation

#### 1. Clone and Setup Environment

```bash
# Create project directory
git clone <repository-url>
cd rlhf-training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Google Cloud Setup

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable compute.googleapis.com
```

#### 3. Environment Configuration

Create a configuration file for your environment:

```yaml
# configs/production_config.yaml
dataset:
  preference_dataset: "gs://your-bucket/rlhf/data/preferences/*.jsonl"
  prompt_dataset: "gs://your-bucket/rlhf/data/prompts/*.jsonl"
  eval_dataset: "gs://your-bucket/rlhf/data/evaluation/*.jsonl"
  preference_size: 50000
  prompt_size: 100000

training:
  large_model_reference: "llama-2-7b"
  reward_model_train_steps: 10000
  reinforcement_learning_train_steps: 8000
  reward_model_learning_rate_multiplier: 1.0
  reinforcement_learning_rate_multiplier: 0.2
  kl_coeff: 0.1
  instruction: "Summarize in less than 50 words"
  batch_size: 64

vertex_ai:
  project_id: "your-project-id"
  region: "us-central1"
  staging_bucket: "gs://your-project-rlhf-staging"
  pipeline_display_name: "rlhf-production-training"
```

#### 4. Verify Installation

```bash
# Test configuration loading
python main.py --config configs/base_config.yaml --calculate-steps

# Expected output:
# Optimal training steps: TrainingSteps(reward_model_steps=1410, reinforcement_learning_steps=320)
```

## Configuration

### Configuration Structure

The system uses a hierarchical configuration system with the following structure:

```yaml
dataset:
  preference_dataset: string    # GCS path to preference data
  prompt_dataset: string        # GCS path to prompt data  
  eval_dataset: string          # GCS path to evaluation data
  preference_size: integer      # Number of preference samples
  prompt_size: integer          # Number of prompt samples

training:
  large_model_reference: string # Model identifier
  reward_model_train_steps: integer
  reinforcement_learning_train_steps: integer
  reward_model_learning_rate_multiplier: float
  reinforcement_learning_rate_multiplier: float
  kl_coeff: float              # KL divergence coefficient
  instruction: string          # Task instruction
  batch_size: integer          # Training batch size

vertex_ai:
  project_id: string           # GCP project ID
  region: string              # GCP region
  staging_bucket: string      # GCS bucket for staging
  pipeline_display_name: string # Pipeline display name
```

### Environment-Specific Configurations

#### Development Configuration

```yaml
# configs/development_config.yaml
dataset:
  preference_dataset: "gs://vertex-ai/generative-ai/rlhf/text_small/summarize_from_feedback_tfds/comparisons/train/*.jsonl"
  prompt_dataset: "gs://vertex-ai/generative-ai/rlhf/text_small/reddit_tfds/train/*.jsonl"
  eval_dataset: "gs://vertex-ai/generative-ai/rlhf/text_small/reddit_tfds/val/*.jsonl"
  preference_size: 3000
  prompt_size: 2000

training:
  reward_model_train_steps: 1410
  reinforcement_learning_train_steps: 320
  # ... other training parameters
```

#### Production Configuration

```yaml
# configs/production_config.yaml
dataset:
  preference_dataset: "gs://vertex-ai/generative-ai/rlhf/text/summarize_from_feedback_tfds/comparisons/train/*.jsonl"
  prompt_dataset: "gs://vertex-ai/generative-ai/rlhf/text/reddit_tfds/train/*.jsonl"
  eval_dataset: "gs://vertex-ai/generative-ai/rlhf/text/reddit_tfds/val/*.jsonl"
  preference_size: 100000
  prompt_size: 50000

training:
  reward_model_train_steps: 10000
  reinforcement_learning_train_steps: 10000
  reinforcement_learning_rate_multiplier: 0.2
  # ... other training parameters
```

### Training Parameter Guidelines

#### Reward Model Training

- **Epochs**: 20-30 for optimal performance
- **Learning Rate**: Start with 1.0 multiplier, adjust based on convergence
- **Batch Size**: Fixed at 64 in pipeline

#### Reinforcement Learning Training

- **Epochs**: 10-20 to avoid reward hacking
- **Learning Rate**: 0.1-0.5 multiplier of base learning rate
- **KL Coefficient**: 0.1 for balanced regularization

## Usage Guide

### Command Line Interface

#### Basic Training

```bash
# Run with small dataset (development)
python main.py --config configs/small_dataset_config.yaml

# Run with full dataset (production)
python main.py --config configs/full_dataset_config.yaml --wait

# Run with custom configuration
python main.py --config path/to/your_config.yaml
```

#### Job Monitoring

```bash
# Monitor existing job
python main.py --config configs/base_config.yaml --monitor projects/123/locations/us-central1/pipelineJobs/456

# Monitor with real-time updates
python main.py --monitor JOB_ID --config configs/base_config.yaml
```

#### Training Analysis

```bash
# Calculate optimal training steps
python main.py --config configs/base_config.yaml --calculate-steps

# Validate configuration
python main.py --config configs/base_config.yaml --validate
```

### Programmatic Usage

```python
from src.config import ConfigManager
from src.training_calculator import TrainingCalculator
from src.vertex_ai_client import VertexAIClient

# Initialize pipeline
config_manager = ConfigManager("configs/production_config.yaml")
rlhf_config = config_manager.get_rlhf_config()

# Calculate training steps
calculator = TrainingCalculator()
steps = calculator.calculate_training_steps(
    preference_dataset_size=rlhf_config.dataset.preference_size,
    prompt_dataset_size=rlhf_config.dataset.prompt_size
)

# Run pipeline
client = VertexAIClient(rlhf_config)
job = client.create_pipeline_job(
    template_path="rlhf_pipeline.yaml",
    parameter_values=config_manager.get_parameter_values()
)

result = client.run_pipeline_job(job, wait_for_completion=True)
```

### Common Workflows

#### Development Workflow

1. Start with small dataset configuration
2. Calculate optimal training steps
3. Run pipeline without waiting for completion
4. Monitor job progress
5. Analyze results and adjust parameters

#### Production Workflow

1. Use full dataset configuration
2. Validate all parameters
3. Run pipeline with completion waiting
4. Monitor with automated alerts
5. Collect and analyze metrics

## Pipeline Components

### Training Step Calculation

The system automatically calculates optimal training steps based on:

```python
steps_per_epoch = ceil(dataset_size / batch_size)
total_steps = steps_per_epoch * num_epochs
```

#### Example Calculation

```python
# For preference dataset of 3000 samples
preference_size = 3000
batch_size = 64
reward_epochs = 30

steps_per_epoch = ceil(3000 / 64) = 47
reward_model_steps = 47 * 30 = 1410
```

### Model Configuration

#### Supported Models and Parameters

| Model | Recommended Steps | Learning Rate | KL Coefficient |
|-------|------------------|---------------|----------------|
| Llama-2-7B | 1000-5000 | 1.0 | 0.1 |
| Llama-2-13B | 2000-8000 | 0.5 | 0.1 |
| T5X | 500-2000 | 1.0 | 0.05 |
| text-bison@001 | 1000-3000 | 0.8 | 0.1 |

### Dataset Requirements

#### Preference Dataset Format

```jsonl
{"prompt": "Original text to summarize", "chosen": "Good summary", "rejected": "Bad summary"}
{"prompt": "Another original text", "chosen": "Better response", "rejected": "Worse response"}
```

#### Prompt Dataset Format

```jsonl
{"prompt": "Text that needs summarization"}
{"prompt": "Another text for summarization"}
```

## Monitoring & Logging

### Real-time Monitoring

The pipeline includes comprehensive monitoring capabilities:

```python
# Monitor job with custom callback
def status_callback(status):
    print(f"Job {status['job_id']} is {status['state']}")
    
monitor.monitor_job_with_updates(job_id, callback=status_callback)
```

### Log Files

#### Application Logs

```
2024-01-15 10:30:45 - RLHFTrainingPipeline - INFO - Starting RLHF Training Pipeline
2024-01-15 10:31:02 - PipelineCompiler - INFO - Compiling RLHF pipeline
2024-01-15 10:31:15 - VertexAIClient - INFO - Pipeline job created: projects/123/locations/us-central1/pipelineJobs/456
```

#### Metrics Logs (JSONL Format)

```json
{
  "timestamp": "2024-01-15T10:35:22.123456",
  "job_id": "projects/123/locations/us-central1/pipelineJobs/456",
  "stage": "reward_model",
  "metrics": {
    "loss": 0.234,
    "accuracy": 0.892,
    "learning_rate": 0.001
  }
}
```

### Health Checks

```bash
# Check pipeline health
python main.py --config configs/base_config.yaml --health-check

# Monitor resource utilization
python -c "from src.monitoring import HealthChecker; HealthChecker().check_resources()"
```

## Troubleshooting

### Common Issues

#### Authentication Errors

**Problem**: `DefaultCredentialsError: Could not automatically determine credentials`

**Solution**:
```bash
# Set up application default credentials
gcloud auth application-default login

# Or set service account key
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

#### Pipeline Compilation Errors

**Problem**: `Pipeline compilation failed: ...`

**Solution**:
1. Check Kubeflow and Vertex AI component versions
2. Verify pipeline dependencies are installed
3. Check for syntax errors in pipeline definition

#### Training Failures

**Problem**: `Training job failed with OOM error`

**Solution**:
- Reduce batch size in configuration
- Use smaller model variant
- Increase machine type in pipeline configuration

### Error Codes and Solutions

| Error Code | Cause | Solution |
|------------|-------|----------|
| AUTH_001 | Missing credentials | Run `gcloud auth application-default login` |
| CONFIG_002 | Invalid configuration | Validate YAML syntax and required fields |
| PIPELINE_003 | Compilation failure | Check component versions and dependencies |
| TRAINING_004 | Resource exhaustion | Adjust batch size or use larger machine type |

### Debug Mode

Enable detailed debugging:

```bash
# Set debug environment variable
export LOG_LEVEL=DEBUG

# Run with verbose output
python main.py --config configs/base_config.yaml --wait
```

## Best Practices

### Configuration Management

1. **Use Environment-Specific Configs**: Maintain separate configurations for dev, staging, and prod
2. **Version Control**: Keep configurations in version control with sensitive data excluded
3. **Validation**: Always validate configurations before running pipelines

### Training Optimization

1. **Dataset Size**: Ensure sufficient data for stable training (min 1000 preference pairs)
2. **Epoch Calculation**: Use 20-30 epochs for reward model, 10-20 for RL training
3. **Learning Rates**: Start with recommended values and adjust based on convergence

### Cost Management

1. **Resource Monitoring**: Use Cloud Monitoring to track costs
2. **Early Stopping**: Implement checkpointing and early stopping
3. **Pipeline Optimization**: Use cached results when possible

### Security

1. **IAM Roles**: Follow principle of least privilege
2. **Data Encryption**: Ensure data at rest and in transit is encrypted
3. **Access Controls**: Restrict bucket and pipeline access

## API Reference

### ConfigManager

#### `get_rlhf_config() -> RLHFConfig`
Returns complete RLHF configuration object.

#### `get_parameter_values() -> Dict[str, Any]`
Returns parameter values for Vertex AI pipeline.

### TrainingCalculator

#### `calculate_training_steps(preference_size, prompt_size, reward_epochs, rl_epochs) -> TrainingSteps`
Calculates optimal training steps.

#### `validate_training_parameters(preference_size, prompt_size, reward_epochs, rl_epochs) -> Tuple[bool, str]`
Validates training parameters.

### VertexAIClient

#### `create_pipeline_job(template_path, parameter_values, enable_caching) -> PipelineJob`
Creates a new pipeline job.

#### `run_pipeline_job(job, wait_for_completion, timeout) -> Dict[str, Any]`
Executes a pipeline job.

#### `monitor_pipeline_job(job_id) -> Dict[str, Any]`
Monitors an existing pipeline job.

### PipelineMonitor

#### `monitor_job_with_updates(job_id, timeout, callback) -> Dict[str, Any]`
Monitors job with periodic updates and optional callback.

#### `generate_monitoring_report(job_status) -> str`
Generates human-readable monitoring report.

## Examples

### Basic Training Example

```python
from src.config import ConfigManager
from src.vertex_ai_client import VertexAIClient

# Load configuration
config_manager = ConfigManager("configs/production_config.yaml")
rlhf_config = config_manager.get_rlhf_config()

# Initialize client
client = VertexAIClient(rlhf_config)

# Create and run job
job = client.create_pipeline_job(
    template_path="rlhf_pipeline.yaml",
    parameter_values=config_manager.get_parameter_values()
)

result = client.run_pipeline_job(job, wait_for_completion=True)

if result["success"]:
    print("Training completed successfully!")
else:
    print(f"Training failed: {result['error']}")
```

### Custom Monitoring Example

```python
from src.monitoring import PipelineMonitor

def custom_callback(status):
    # Send alert if job fails
    if status["state"] == "FAILED":
        send_alert(f"Pipeline failed: {status['error']}")
    
    # Log progress every 10%
    if "progress" in status:
        print(f"Progress: {status['progress']}%")

monitor = PipelineMonitor(vertex_ai_client)
status = monitor.monitor_job_with_updates(
    job_id="your-job-id",
    callback=custom_callback
)
```

