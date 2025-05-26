"""
Trajectory Data Management Module for Agent Tuning Optimization Framework

This module provides functionality for loading, processing, and managing agent interaction
trajectories for training and evaluation purposes.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from tqdm import tqdm

class Trajectory:
    """Class representing a single agent interaction trajectory."""
    
    def __init__(
        self, 
        task_description: str,
        interactions: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a trajectory.
        
        Args:
            task_description: Description of the task
            interactions: List of interaction turns (each with 'user' and 'agent' keys)
            metadata: Additional metadata about the trajectory
        """
        self.task_description = task_description
        self.interactions = interactions
        self.metadata = metadata or {}
        self.quality_score = self.metadata.get('quality_score', None)
        self.is_positive = self.metadata.get('is_positive', True)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trajectory to dictionary.
        
        Returns:
            Dictionary representation of the trajectory
        """
        return {
            'task_description': self.task_description,
            'interactions': self.interactions,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trajectory':
        """
        Create trajectory from dictionary.
        
        Args:
            data: Dictionary representation of the trajectory
            
        Returns:
            Trajectory instance
        """
        return cls(
            task_description=data['task_description'],
            interactions=data['interactions'],
            metadata=data.get('metadata', {})
        )
    
    def to_training_format(self, format_type: str = 'interleaved') -> str:
        """
        Convert trajectory to training format.
        
        Args:
            format_type: Format type ('interleaved', 'completion', etc.)
            
        Returns:
            Formatted trajectory as string
        """
        if format_type == 'interleaved':
            # Format as interleaved conversation
            result = f"Task: {self.task_description}\n\n"
            
            for i, interaction in enumerate(self.interactions):
                result += f"User: {interaction['user']}\n"
                result += f"Agent: {interaction['agent']}\n\n"
            
            return result.strip()
        
        elif format_type == 'completion':
            # Format as completion task (last agent response is the target)
            if not self.interactions:
                return ""
            
            result = f"Task: {self.task_description}\n\n"
            
            for i, interaction in enumerate(self.interactions[:-1]):
                result += f"User: {interaction['user']}\n"
                result += f"Agent: {interaction['agent']}\n\n"
            
            # Add last user query without agent response
            result += f"User: {self.interactions[-1]['user']}\n"
            result += f"Agent:"
            
            return result.strip(), self.interactions[-1]['agent'].strip()
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def get_quality_score(self) -> float:
        """
        Get quality score for the trajectory.
        
        Returns:
            Quality score (0.0 to 1.0)
        """
        if self.quality_score is not None:
            return self.quality_score
        
        # Calculate simple quality score based on response length and complexity
        score = 0.0
        
        if not self.interactions:
            return score
        
        # Average response length (normalized)
        avg_length = np.mean([len(turn['agent']) for turn in self.interactions])
        length_score = min(avg_length / 500, 1.0)  # Normalize to max of 500 chars
        
        # Response complexity (simple heuristic based on unique words)
        all_responses = " ".join([turn['agent'] for turn in self.interactions])
        unique_words = len(set(all_responses.lower().split()))
        complexity_score = min(unique_words / 200, 1.0)  # Normalize to max of 200 unique words
        
        # Combine scores
        score = 0.6 * length_score + 0.4 * complexity_score
        
        # Cache the score
        self.quality_score = score
        self.metadata['quality_score'] = score
        
        return score


class TrajectoryDataset:
    """Dataset for managing collections of agent interaction trajectories."""
    
    def __init__(self, name: str):
        """
        Initialize the trajectory dataset.
        
        Args:
            name: Name of the dataset
        """
        self.name = name
        self.trajectories: List[Trajectory] = []
        self.positive_trajectories: List[Trajectory] = []
        self.negative_trajectories: List[Trajectory] = []
    
    def add_trajectory(self, trajectory: Trajectory) -> None:
        """
        Add a trajectory to the dataset.
        
        Args:
            trajectory: Trajectory to add
        """
        self.trajectories.append(trajectory)
        
        # Add to positive or negative list based on metadata
        if trajectory.is_positive:
            self.positive_trajectories.append(trajectory)
        else:
            self.negative_trajectories.append(trajectory)
    
    def load_from_json(self, file_path: str) -> None:
        """
        Load trajectories from JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # List of trajectories
            for item in data:
                self.add_trajectory(Trajectory.from_dict(item))
        elif isinstance(data, dict) and 'trajectories' in data:
            # Dictionary with trajectories key
            for item in data['trajectories']:
                self.add_trajectory(Trajectory.from_dict(item))
        else:
            raise ValueError(f"Unsupported JSON format in {file_path}")
    
    def save_to_json(self, file_path: str) -> None:
        """
        Save trajectories to JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        data = {
            'name': self.name,
            'trajectories': [t.to_dict() for t in self.trajectories]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_trajectories(
        self, 
        positive_only: bool = False,
        negative_only: bool = False,
        min_quality: Optional[float] = None,
        max_samples: Optional[int] = None
    ) -> List[Trajectory]:
        """
        Get trajectories based on filtering criteria.
        
        Args:
            positive_only: Whether to return only positive trajectories
            negative_only: Whether to return only negative trajectories
            min_quality: Minimum quality score threshold
            max_samples: Maximum number of samples to return
            
        Returns:
            Filtered list of trajectories
        """
        if positive_only and negative_only:
            raise ValueError("Cannot set both positive_only and negative_only to True")
        
        # Select base list
        if positive_only:
            trajectories = self.positive_trajectories.copy()
        elif negative_only:
            trajectories = self.negative_trajectories.copy()
        else:
            trajectories = self.trajectories.copy()
        
        # Apply quality filter
        if min_quality is not None:
            trajectories = [t for t in trajectories if t.get_quality_score() >= min_quality]
        
        # Apply max samples limit
        if max_samples is not None and max_samples < len(trajectories):
            trajectories = trajectories[:max_samples]
        
        return trajectories
    
    def get_training_examples(
        self, 
        format_type: str = 'interleaved',
        positive_ratio: float = 0.8,
        min_quality: Optional[float] = 0.5,
        max_samples: Optional[int] = None
    ) -> Union[List[str], Tuple[List[str], List[str]]]:
        """
        Get formatted training examples from trajectories.
        
        Args:
            format_type: Format type ('interleaved', 'completion', etc.)
            positive_ratio: Ratio of positive to total examples
            min_quality: Minimum quality score threshold
            max_samples: Maximum number of samples to return
            
        Returns:
            Formatted training examples (format depends on format_type)
        """
        # Get positive and negative trajectories
        positive = self.get_trajectories(positive_only=True, min_quality=min_quality)
        negative = self.get_trajectories(negative_only=True)
        
        # Calculate sample counts
        if max_samples is not None:
            pos_count = int(max_samples * positive_ratio)
            neg_count = max_samples - pos_count
        else:
            pos_count = len(positive)
            neg_count = len(negative)
        
        # Sample trajectories
        if pos_count < len(positive):
            positive = np.random.choice(positive, pos_count, replace=False).tolist()
        
        if neg_count < len(negative):
            negative = np.random.choice(negative, neg_count, replace=False).tolist()
        
        # Format trajectories
        if format_type == 'interleaved':
            pos_examples = [t.to_training_format(format_type) for t in positive]
            neg_examples = [t.to_training_format(format_type) for t in negative]
            return pos_examples + neg_examples
        
        elif format_type == 'completion':
            pos_inputs = []
            pos_targets = []
            
            for t in positive:
                inp, target = t.to_training_format(format_type)
                pos_inputs.append(inp)
                pos_targets.append(target)
            
            neg_inputs = []
            neg_targets = []
            
            for t in negative:
                inp, target = t.to_training_format(format_type)
                neg_inputs.append(inp)
                neg_targets.append(target)
            
            return pos_inputs + neg_inputs, pos_targets + neg_targets
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """
        Analyze the dataset and return statistics.
        
        Returns:
            Dictionary of dataset statistics
        """
        if not self.trajectories:
            return {
                'total_trajectories': 0,
                'positive_count': 0,
                'negative_count': 0
            }
        
        # Basic counts
        total = len(self.trajectories)
        positive_count = len(self.positive_trajectories)
        negative_count = len(self.negative_trajectories)
        
        # Quality statistics
        quality_scores = [t.get_quality_score() for t in self.trajectories]
        avg_quality = np.mean(quality_scores)
        min_quality = np.min(quality_scores)
        max_quality = np.max(quality_scores)
        
        # Interaction statistics
        interaction_counts = [len(t.interactions) for t in self.trajectories]
        avg_interactions = np.mean(interaction_counts)
        max_interactions = np.max(interaction_counts)
        
        # Task diversity (simple heuristic based on unique task descriptions)
        unique_tasks = len(set([t.task_description for t in self.trajectories]))
        
        return {
            'total_trajectories': total,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_count / total if total > 0 else 0,
            'avg_quality': avg_quality,
            'min_quality': min_quality,
            'max_quality': max_quality,
            'avg_interactions': avg_interactions,
            'max_interactions': max_interactions,
            'unique_tasks': unique_tasks
        }


def create_synthetic_dataset(num_trajectories: int = 10) -> TrajectoryDataset:
    """
    Create a synthetic dataset for testing purposes.
    
    Args:
        num_trajectories: Number of trajectories to create
        
    Returns:
        Synthetic trajectory dataset
    """
    dataset = TrajectoryDataset("synthetic_dataset")
    
    # Sample task descriptions
    task_descriptions = [
        "Book a flight from New York to London for next week",
        "Find a vegetarian restaurant near downtown",
        "Schedule a meeting with the marketing team for tomorrow",
        "Order a new laptop with at least 16GB RAM",
        "Write a congratulatory email to a colleague who got promoted",
        "Research the best electric cars available in the market",
        "Create a weekly meal plan with shopping list",
        "Find information about tourist attractions in Barcelona",
        "Help me debug a Python script that's giving an IndexError",
        "Summarize the main points from the attached research paper"
    ]
    
    # Create trajectories
    for i in range(num_trajectories):
        # Select task
        task_idx = i % len(task_descriptions)
        task = task_descriptions[task_idx]
        
        # Create interactions (2-4 turns)
        num_turns = np.random.randint(2, 5)
        interactions = []
        
        for j in range(num_turns):
            if j == 0:
                user_msg = f"I need help with this task: {task}"
                agent_msg = f"I'd be happy to help you {task.lower()}. Could you provide more details about your preferences?"
            elif j == num_turns - 1:
                user_msg = "That sounds good. Please proceed with the final steps."
                agent_msg = f"I've completed the task to {task.lower()}. Here's a summary of what I did..."
            else:
                user_msg = f"I prefer options that are {['affordable', 'convenient', 'high-quality'][j % 3]}."
                agent_msg = f"Based on your preference for {['affordable', 'convenient', 'high-quality'][j % 3]} options, I recommend..."
            
            interactions.append({
                'user': user_msg,
                'agent': agent_msg
            })
        
        # Determine if positive or negative example
        is_positive = (i % 4 != 0)  # 75% positive, 25% negative
        
        # Create metadata
        metadata = {
            'is_positive': is_positive,
            'quality_score': np.random.uniform(0.7, 0.9) if is_positive else np.random.uniform(0.3, 0.5),
            'created_at': '2025-05-21'
        }
        
        # Create and add trajectory
        trajectory = Trajectory(
            task_description=task,
            interactions=interactions,
            metadata=metadata
        )
        
        dataset.add_trajectory(trajectory)
    
    return dataset
