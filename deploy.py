"""
Deployment Script for Agent Tuning Optimization Framework

This script prepares the framework for deployment to production environments
and Hugging Face Spaces.
"""

import os
import shutil
import argparse
import subprocess
import json
from pathlib import Path

def prepare_for_deployment(source_dir, output_dir, config_path=None):
    """
    Prepare the framework for deployment.
    
    Args:
        source_dir: Source directory containing the framework
        output_dir: Output directory for deployment package
        config_path: Path to configuration file (optional)
    """
    print(f"Preparing deployment package from {source_dir} to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy core modules
    core_modules = [
        "models",
        "data",
        "training",
        "evaluation",
        "main.py",
        "README.md"
    ]
    
    for module in core_modules:
        source_path = os.path.join(source_dir, module)
        target_path = os.path.join(output_dir, module)
        
        if os.path.isdir(source_path):
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)
    
    # Copy configuration file if provided
    if config_path:
        shutil.copy2(config_path, os.path.join(output_dir, "config.json"))
    else:
        # Use example config
        example_config_path = os.path.join(source_dir, "example_config.json")
        if os.path.exists(example_config_path):
            shutil.copy2(example_config_path, os.path.join(output_dir, "config.json"))
    
    # Create requirements.txt
    requirements = [
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "scikit-learn>=1.0.0",
        "peft>=0.2.0"
    ]
    
    with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
        f.write("\n".join(requirements))
    
    # Create setup.py
    setup_py = """
from setuptools import setup, find_packages

setup(
    name="agent_tuning_framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "scikit-learn>=1.0.0",
        "peft>=0.2.0"
    ],
    author="MBZUAI Technical Interview Preparation",
    author_email="example@example.com",
    description="Agent Tuning Optimization Framework with Negative and Synthetic Samples",
    keywords="nlp, machine learning, agent tuning, language models",
    url="https://github.com/username/agent_tuning_framework",
)
"""
    
    with open(os.path.join(output_dir, "setup.py"), "w") as f:
        f.write(setup_py)
    
    # Create app.py for web interface
    app_py = """
import os
import json
import gradio as gr
import torch
from models.llm_interface import LLMInterface
from data.trajectory_data import TrajectoryDataset, Trajectory
from training.negative_samples import create_negative_sample_generator
from training.synthetic_trajectories import create_synthetic_trajectory_generator

# Initialize model
def load_model(model_path):
    if os.path.exists(model_path):
        return LLMInterface(
            model_name=model_path,
            model_type="causal",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        return LLMInterface(
            model_name="gpt2",
            model_type="causal",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

# Initialize components
model = load_model("./tuned_model")
negative_generator = create_negative_sample_generator("response_degradation")
synthetic_generator = create_synthetic_trajectory_generator("template")

# Define interface functions
def generate_response(task, user_message):
    prompt = f"Task: {task}\\n\\nUser: {user_message}\\nAgent:"
    response = model.generate(prompt)
    return response["response"]

def generate_negative_sample(task, user_message, agent_message):
    trajectory = Trajectory(
        task_description=task,
        interactions=[{"user": user_message, "agent": agent_message}]
    )
    negative_trajectory = negative_generator.generate(trajectory)
    return negative_trajectory.interactions[0]["agent"]

def generate_synthetic_trajectory(task):
    trajectory = synthetic_generator.generate(task)
    result = ""
    for i, interaction in enumerate(trajectory.interactions):
        result += f"Turn {i+1}:\\nUser: {interaction['user']}\\nAgent: {interaction['agent']}\\n\\n"
    return result

# Create Gradio interface
with gr.Blocks(title="Agent Tuning Framework Demo") as demo:
    gr.Markdown("# Agent Tuning Optimization Framework Demo")
    
    with gr.Tab("Generate Response"):
        with gr.Row():
            with gr.Column():
                task_input = gr.Textbox(label="Task Description")
                user_input = gr.Textbox(label="User Message")
                generate_btn = gr.Button("Generate Response")
            with gr.Column():
                response_output = gr.Textbox(label="Agent Response")
        
        generate_btn.click(
            generate_response,
            inputs=[task_input, user_input],
            outputs=response_output
        )
    
    with gr.Tab("Generate Negative Sample"):
        with gr.Row():
            with gr.Column():
                neg_task_input = gr.Textbox(label="Task Description")
                neg_user_input = gr.Textbox(label="User Message")
                neg_agent_input = gr.Textbox(label="Agent Message (Positive Example)")
                neg_generate_btn = gr.Button("Generate Negative Sample")
            with gr.Column():
                neg_output = gr.Textbox(label="Negative Sample")
        
        neg_generate_btn.click(
            generate_negative_sample,
            inputs=[neg_task_input, neg_user_input, neg_agent_input],
            outputs=neg_output
        )
    
    with gr.Tab("Generate Synthetic Trajectory"):
        with gr.Row():
            with gr.Column():
                synth_task_input = gr.Textbox(label="Task Description")
                synth_generate_btn = gr.Button("Generate Synthetic Trajectory")
            with gr.Column():
                synth_output = gr.Textbox(label="Synthetic Trajectory")
        
        synth_generate_btn.click(
            generate_synthetic_trajectory,
            inputs=[synth_task_input],
            outputs=synth_output
        )

if __name__ == "__main__":
    demo.launch()
"""
    
    with open(os.path.join(output_dir, "app.py"), "w") as f:
        f.write(app_py)
    
    # Create Dockerfile
    dockerfile = """
FROM python:3.9-slim

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gradio>=3.0.0

EXPOSE 7860

CMD ["python", "app.py"]
"""
    
    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile)
    
    # Create README for deployment
    deployment_readme = """
# Agent Tuning Optimization Framework

This package contains the Agent Tuning Optimization Framework with Negative and Synthetic Samples, a comprehensive solution for efficiently tuning large language models into specialized agents.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

```bash
python main.py --config config.json --output ./experiment_results
```

### Web Interface

```bash
pip install gradio
python app.py
```

## Deployment Options

### Docker

```bash
docker build -t agent-tuning-framework .
docker run -p 7860:7860 agent-tuning-framework
```

### Hugging Face Spaces

This project can be deployed to Hugging Face Spaces by following these steps:

1. Create a new Space on Hugging Face (https://huggingface.co/spaces)
2. Select "Gradio" as the SDK
3. Upload all files from this directory to the Space
4. The Space will automatically build and deploy the application

## Configuration

See `config.json` for configuration options.

## License

MIT
"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(deployment_readme)
    
    # Create Hugging Face Space files
    os.makedirs(os.path.join(output_dir, "huggingface"), exist_ok=True)
    
    # Create requirements.txt for Hugging Face
    hf_requirements = requirements + ["gradio>=3.0.0"]
    
    with open(os.path.join(output_dir, "huggingface", "requirements.txt"), "w") as f:
        f.write("\n".join(hf_requirements))
    
    # Copy app.py
    shutil.copy2(os.path.join(output_dir, "app.py"), os.path.join(output_dir, "huggingface", "app.py"))
    
    # Create README for Hugging Face
    hf_readme = """
---
title: Agent Tuning Optimization Framework
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.36.1
app_file: app.py
pinned: false
license: mit
---

# Agent Tuning Optimization Framework

This Space demonstrates the Agent Tuning Optimization Framework with Negative and Synthetic Samples, a comprehensive solution for efficiently tuning large language models into specialized agents.

## Features

- Generate agent responses for given tasks and user messages
- Create negative samples from positive examples
- Generate synthetic interaction trajectories

## Usage

1. Select a tab for the desired functionality
2. Enter the required information
3. Click the button to generate results

## Learn More

For more information, visit the [GitHub repository](https://github.com/username/agent_tuning_framework).
"""
    
    with open(os.path.join(output_dir, "huggingface", "README.md"), "w") as f:
        f.write(hf_readme)
    
    print(f"Deployment package prepared in {output_dir}")
    print(f"Hugging Face Space files prepared in {os.path.join(output_dir, 'huggingface')}")

def main():
    """Main function for preparing deployment package."""
    parser = argparse.ArgumentParser(description="Prepare deployment package for Agent Tuning Framework")
    parser.add_argument("--source", type=str, default=".", help="Source directory containing the framework")
    parser.add_argument("--output", type=str, default="./deployment", help="Output directory for deployment package")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    prepare_for_deployment(args.source, args.output, args.config)

if __name__ == "__main__":
    main()
