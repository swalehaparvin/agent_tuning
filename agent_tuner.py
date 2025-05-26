"""
Agent Tuning Module for Agent Tuning Optimization Framework

This module provides functionality for efficiently tuning large language models
into specialized agents using a combination of positive examples, negative examples,
and synthetically generated interaction trajectories.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from tqdm import tqdm
from transformers import (
    Trainer, TrainingArguments, 
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM, AutoTokenizer
)
from datasets import Dataset

from data.trajectory_data import Trajectory, TrajectoryDataset
from models.llm_interface import LLMInterface

class AgentTuner:
    """Base class for agent tuning methods."""
    
    def __init__(self, name: str):
        """
        Initialize the agent tuner.
        
        Args:
            name: Name of the tuning method
        """
        self.name = name
    
    def tune(
        self, 
        model_name: str,
        trajectories: List[Trajectory],
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Tune a model into a specialized agent.
        
        Args:
            model_name: Name of the base model
            trajectories: List of training trajectories
            **kwargs: Additional tuning parameters
            
        Returns:
            Tuple of (tuned_model, training_metrics)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_model(self, model: Any, path: str) -> None:
        """
        Save the tuned model.
        
        Args:
            model: Tuned model
            path: Path to save the model
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def load_model(self, path: str) -> Any:
        """
        Load a tuned model.
        
        Args:
            path: Path to the model
            
        Returns:
            Loaded model
        """
        raise NotImplementedError("Subclasses must implement this method")


class SupervisedFineTuner(AgentTuner):
    """Tune agents using supervised fine-tuning."""
    
    def __init__(self):
        """Initialize the supervised fine-tuner."""
        super().__init__("supervised_fine_tuning")
    
    def tune(
        self, 
        model_name: str,
        trajectories: List[Trajectory],
        output_dir: str = "./tuned_model",
        num_train_epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 512,
        format_type: str = "interleaved",
        positive_weight: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Tune a model using supervised fine-tuning.
        
        Args:
            model_name: Name of the base model
            trajectories: List of training trajectories
            output_dir: Directory to save the model
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            max_seq_length: Maximum sequence length
            format_type: Format type for trajectories
            positive_weight: Weight for positive examples
            device: Device to use for training
            **kwargs: Additional tuning parameters
            
        Returns:
            Tuple of (tuned_model, training_metrics)
        """
        print(f"Starting supervised fine-tuning of {model_name}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare training data
        print("Preparing training data...")
        
        # Separate positive and negative trajectories
        positive_trajectories = [t for t in trajectories if t.is_positive]
        negative_trajectories = [t for t in trajectories if not t.is_positive]
        
        print(f"Found {len(positive_trajectories)} positive and {len(negative_trajectories)} negative trajectories")
        
        # Calculate sample counts based on positive weight
        total_samples = len(trajectories)
        target_positive = int(total_samples * positive_weight)
        target_negative = total_samples - target_positive
        
        # Sample trajectories to achieve desired ratio
        if len(positive_trajectories) > target_positive:
            positive_trajectories = np.random.choice(positive_trajectories, target_positive, replace=False).tolist()
        
        if len(negative_trajectories) > target_negative:
            negative_trajectories = np.random.choice(negative_trajectories, target_negative, replace=False).tolist()
        
        # Combine trajectories
        sampled_trajectories = positive_trajectories + negative_trajectories
        np.random.shuffle(sampled_trajectories)
        
        print(f"Using {len(positive_trajectories)} positive and {len(negative_trajectories)} negative trajectories for training")
        
        # Format trajectories for training
        training_texts = []
        
        for trajectory in tqdm(sampled_trajectories, desc="Formatting trajectories"):
            formatted = trajectory.to_training_format(format_type)
            training_texts.append(formatted)
        
        # Tokenize training data
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length
            )
        
        # Create dataset
        dataset = Dataset.from_dict({"text": training_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=2,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            report_to="none"
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        # Train the model
        print("Starting training...")
        train_result = trainer.train()
        
        # Save the model
        print(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Return the model and metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "samples_per_second": train_result.metrics["train_samples_per_second"],
            "num_train_samples": len(tokenized_dataset)
        }
        
        return model, metrics
    
    def save_model(self, model: Any, path: str) -> None:
        """
        Save the tuned model.
        
        Args:
            model: Tuned model
            path: Path to save the model
        """
        model.save_pretrained(path)
    
    def load_model(self, path: str) -> Any:
        """
        Load a tuned model.
        
        Args:
            path: Path to the model
            
        Returns:
            Loaded model
        """
        return AutoModelForCausalLM.from_pretrained(path)


class ParameterEfficientFineTuner(AgentTuner):
    """Tune agents using parameter-efficient fine-tuning methods."""
    
    def __init__(self):
        """Initialize the parameter-efficient fine-tuner."""
        super().__init__("parameter_efficient_fine_tuning")
    
    def tune(
        self, 
        model_name: str,
        trajectories: List[Trajectory],
        output_dir: str = "./tuned_model",
        method: str = "lora",  # 'lora', 'prefix', 'prompt_tuning'
        num_train_epochs: int = 3,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 512,
        format_type: str = "interleaved",
        positive_weight: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Tune a model using parameter-efficient methods.
        
        Args:
            model_name: Name of the base model
            trajectories: List of training trajectories
            output_dir: Directory to save the model
            method: PEFT method to use
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            max_seq_length: Maximum sequence length
            format_type: Format type for trajectories
            positive_weight: Weight for positive examples
            device: Device to use for training
            **kwargs: Additional tuning parameters
            
        Returns:
            Tuple of (tuned_model, training_metrics)
        """
        try:
            from peft import (
                get_peft_model, LoraConfig, PrefixTuningConfig, 
                PromptTuningConfig, TaskType, PeftModel
            )
        except ImportError:
            raise ImportError("PEFT library is required for parameter-efficient fine-tuning. Install it with 'pip install peft'.")
        
        print(f"Starting parameter-efficient fine-tuning of {model_name} using {method}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure PEFT method
        if method == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            )
        elif method == "prefix":
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=20,
                prefix_projection=True
            )
        elif method == "prompt_tuning":
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=20,
                tokenizer_name_or_path=model_name
            )
        else:
            raise ValueError(f"Unsupported PEFT method: {method}")
        
        # Create PEFT model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Prepare training data (same as SupervisedFineTuner)
        print("Preparing training data...")
        
        # Separate positive and negative trajectories
        positive_trajectories = [t for t in trajectories if t.is_positive]
        negative_trajectories = [t for t in trajectories if not t.is_positive]
        
        print(f"Found {len(positive_trajectories)} positive and {len(negative_trajectories)} negative trajectories")
        
        # Calculate sample counts based on positive weight
        total_samples = len(trajectories)
        target_positive = int(total_samples * positive_weight)
        target_negative = total_samples - target_positive
        
        # Sample trajectories to achieve desired ratio
        if len(positive_trajectories) > target_positive:
            positive_trajectories = np.random.choice(positive_trajectories, target_positive, replace=False).tolist()
        
        if len(negative_trajectories) > target_negative:
            negative_trajectories = np.random.choice(negative_trajectories, target_negative, replace=False).tolist()
        
        # Combine trajectories
        sampled_trajectories = positive_trajectories + negative_trajectories
        np.random.shuffle(sampled_trajectories)
        
        print(f"Using {len(positive_trajectories)} positive and {len(negative_trajectories)} negative trajectories for training")
        
        # Format trajectories for training
        training_texts = []
        
        for trajectory in tqdm(sampled_trajectories, desc="Formatting trajectories"):
            formatted = trajectory.to_training_format(format_type)
            training_texts.append(formatted)
        
        # Tokenize training data
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length
            )
        
        # Create dataset
        dataset = Dataset.from_dict({"text": training_texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=2,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            report_to="none"
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        # Train the model
        print("Starting training...")
        train_result = trainer.train()
        
        # Save the model
        print(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Return the model and metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "samples_per_second": train_result.metrics["train_samples_per_second"],
            "num_train_samples": len(tokenized_dataset),
            "peft_method": method
        }
        
        return model, metrics
    
    def save_model(self, model: Any, path: str) -> None:
        """
        Save the tuned model.
        
        Args:
            model: Tuned model
            path: Path to save the model
        """
        model.save_pretrained(path)
    
  
(Content truncated due to size limit. Use line ranges to read in chunks)