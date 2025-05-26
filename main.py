"""
Main Integration Module for Agent Tuning Optimization Framework

This module provides functionality for integrating all components of the framework
and running end-to-end experiments.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Union, Optional, Tuple

from models.llm_interface import LLMInterface
from data.trajectory_data import Trajectory, TrajectoryDataset, create_synthetic_dataset
from training.negative_samples import create_negative_sample_generator
from training.synthetic_trajectories import create_synthetic_trajectory_generator
from training.agent_tuner import create_agent_tuner
from evaluation.evaluators import create_agent_evaluator

def run_experiment(
    experiment_config: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Run an end-to-end experiment with the framework.
    
    Args:
        experiment_config: Experiment configuration
        output_dir: Directory to save results
        
    Returns:
        Dictionary of experiment results
    """
    print(f"Starting experiment: {experiment_config['name']}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save experiment configuration
    with open(f"{output_dir}/experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)
    
    # Initialize LLM interface
    print("Initializing LLM interface...")
    llm_config = experiment_config.get("llm", {})
    llm_interface = LLMInterface(
        model_name=llm_config.get("model_name", "gpt2"),
        model_type=llm_config.get("model_type", "causal"),
        device=llm_config.get("device", "cpu"),
        max_length=llm_config.get("max_length", 512),
        temperature=llm_config.get("temperature", 0.7)
    )
    
    # Load or create dataset
    print("Preparing dataset...")
    dataset_config = experiment_config.get("dataset", {})
    
    if dataset_config.get("path"):
        # Load existing dataset
        dataset = TrajectoryDataset(dataset_config.get("name", "experiment_dataset"))
        dataset.load_from_json(dataset_config["path"])
    else:
        # Create synthetic dataset
        dataset = create_synthetic_dataset(dataset_config.get("num_trajectories", 20))
    
    print(f"Dataset loaded with {len(dataset.trajectories)} trajectories")
    
    # Generate negative samples
    print("Generating negative samples...")
    negative_config = experiment_config.get("negative_samples", {})
    
    if negative_config.get("enabled", True):
        negative_generator = create_negative_sample_generator(
            negative_config.get("method", "response_degradation")
        )
        
        positive_trajectories = dataset.get_trajectories(positive_only=True)
        negative_trajectories = negative_generator.batch_generate(
            positive_trajectories,
            **negative_config.get("params", {})
        )
        
        # Add negative trajectories to dataset
        for trajectory in negative_trajectories:
            dataset.add_trajectory(trajectory)
        
        print(f"Added {len(negative_trajectories)} negative trajectories")
    
    # Generate synthetic trajectories
    print("Generating synthetic trajectories...")
    synthetic_config = experiment_config.get("synthetic_trajectories", {})
    
    if synthetic_config.get("enabled", True):
        synthetic_generator = create_synthetic_trajectory_generator(
            synthetic_config.get("method", "template"),
            llm_interface if synthetic_config.get("method") in ["llm", "hybrid"] else None
        )
        
        # Generate from task descriptions
        task_descriptions = [t.task_description for t in dataset.get_trajectories(positive_only=True)]
        task_descriptions = list(set(task_descriptions))  # Remove duplicates
        
        synthetic_trajectories = synthetic_generator.batch_generate(
            task_descriptions,
            **synthetic_config.get("params", {})
        )
        
        # Add synthetic trajectories to dataset
        for trajectory in synthetic_trajectories:
            dataset.add_trajectory(trajectory)
        
        print(f"Added {len(synthetic_trajectories)} synthetic trajectories")
    
    # Save the enhanced dataset
    dataset.save_to_json(f"{output_dir}/enhanced_dataset.json")
    
    # Analyze dataset
    dataset_stats = dataset.analyze_dataset()
    with open(f"{output_dir}/dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)
    
    # Split dataset for training and evaluation
    all_trajectories = dataset.get_trajectories()
    split_idx = int(len(all_trajectories) * 0.8)  # 80% for training
    
    train_trajectories = all_trajectories[:split_idx]
    eval_trajectories = all_trajectories[split_idx:]
    
    print(f"Split dataset: {len(train_trajectories)} for training, {len(eval_trajectories)} for evaluation")
    
    # Tune agent
    print("Tuning agent...")
    tuning_config = experiment_config.get("tuning", {})
    
    tuner = create_agent_tuner(tuning_config.get("method", "supervised"))
    
    tuned_model, tuning_metrics = tuner.tune(
        model_name=llm_config.get("model_name", "gpt2"),
        trajectories=train_trajectories,
        output_dir=f"{output_dir}/tuned_model",
        **tuning_config.get("params", {})
    )
    
    # Save tuning metrics
    with open(f"{output_dir}/tuning_metrics.json", "w") as f:
        # Convert any non-serializable values to strings
        serializable_metrics = {}
        for k, v in tuning_metrics.items():
            if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                serializable_metrics[k] = v
            else:
                serializable_metrics[k] = str(v)
        
        json.dump(serializable_metrics, f, indent=2)
    
    # Create tuned model interface
    tuned_llm_interface = LLMInterface(
        model_name=f"{output_dir}/tuned_model",
        model_type=llm_config.get("model_type", "causal"),
        device=llm_config.get("device", "cpu"),
        max_length=llm_config.get("max_length", 512),
        temperature=llm_config.get("temperature", 0.7)
    )
    
    # Evaluate agent
    print("Evaluating agent...")
    eval_config = experiment_config.get("evaluation", {})
    
    evaluator = create_agent_evaluator(eval_config.get("method", "quality"))
    
    eval_results = evaluator.evaluate(
        llm_interface=tuned_llm_interface,
        test_trajectories=eval_trajectories,
        **eval_config.get("params", {})
    )
    
    # Visualize evaluation results
    evaluator.visualize_results(
        results=eval_results,
        output_dir=f"{output_dir}/evaluation"
    )
    
    # Save evaluation results
    with open(f"{output_dir}/evaluation_results.json", "w") as f:
        # Create a simplified version without large data
        simplified_results = {}
        
        if "aggregated" in eval_results:
            simplified_results["aggregated"] = eval_results["aggregated"]
        
        if "metrics" in eval_results:
            # Include only essential metrics
            simplified_results["metrics"] = [
                {k: v for k, v in m.items() if k not in ["generated_responses"]}
                for m in eval_results["metrics"]
            ]
        
        json.dump(simplified_results, f, indent=2)
    
    # Comparative evaluation (if configured)
    if eval_config.get("comparative", {}).get("enabled", False):
        print("Performing comparative evaluation...")
        
        # Create baseline model interface
        baseline_llm_interface = LLMInterface(
            model_name=llm_config.get("model_name", "gpt2"),
            model_type=llm_config.get("model_type", "causal"),
            device=llm_config.get("device", "cpu"),
            max_length=llm_config.get("max_length", 512),
            temperature=llm_config.get("temperature", 0.7)
        )
        
        # Create comparative evaluator
        comparative_evaluator = create_agent_evaluator("comparative")
        
        # Evaluate and compare
        comparative_results = comparative_evaluator.evaluate(
            llm_interfaces={
                "baseline": baseline_llm_interface,
                "tuned": tuned_llm_interface
            },
            test_trajectories=eval_trajectories,
            **eval_config.get("comparative", {}).get("params", {})
        )
        
        # Visualize comparative results
        comparative_evaluator.visualize_results(
            results=comparative_results,
            output_dir=f"{output_dir}/comparative"
        )
        
        # Save comparative results
        with open(f"{output_dir}/comparative_results.json", "w") as f:
            # Create a simplified version
            simplified_comparative = {
                "comparative": comparative_results.get("comparative", {})
            }
            
            json.dump(simplified_comparative, f, indent=2)
    
    print(f"Experiment completed. Results saved to {output_dir}")
    
    return {
        "dataset_stats": dataset_stats,
        "tuning_metrics": tuning_metrics,
        "evaluation_results": eval_results
    }

def main():
    """Main function for running the framework from command line."""
    parser = argparse.ArgumentParser(description="Agent Tuning Optimization Framework")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment configuration file")
    parser.add_argument("--output", type=str, default="./experiment_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Load experiment configuration
    with open(args.config, "r") as f:
        experiment_config = json.load(f)
    
    # Run experiment
    run_experiment(experiment_config, args.output)

if __name__ == "__main__":
    main()
