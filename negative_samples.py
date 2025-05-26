"""
Negative Sample Generation Module for Agent Tuning Optimization Framework

This module provides functionality for generating negative samples to enhance
agent tuning by exposing the model to challenging failure cases.
"""

import random
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from tqdm import tqdm

from data.trajectory_data import Trajectory, TrajectoryDataset

class NegativeSampleGenerator:
    """Base class for negative sample generation strategies."""
    
    def __init__(self, name: str):
        """
        Initialize the negative sample generator.
        
        Args:
            name: Name of the generator strategy
        """
        self.name = name
    
    def generate(
        self, 
        trajectory: Trajectory,
        **kwargs
    ) -> Trajectory:
        """
        Generate a negative sample from a positive trajectory.
        
        Args:
            trajectory: Positive trajectory to transform
            **kwargs: Additional generation parameters
            
        Returns:
            Negative trajectory
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def batch_generate(
        self, 
        trajectories: List[Trajectory],
        **kwargs
    ) -> List[Trajectory]:
        """
        Generate negative samples from a batch of positive trajectories.
        
        Args:
            trajectories: List of positive trajectories
            **kwargs: Additional generation parameters
            
        Returns:
            List of negative trajectories
        """
        negative_trajectories = []
        
        for trajectory in tqdm(trajectories, desc=f"Generating negative samples with {self.name}"):
            negative_trajectories.append(self.generate(trajectory, **kwargs))
        
        return negative_trajectories


class ResponseDegradationGenerator(NegativeSampleGenerator):
    """Generate negative samples by degrading agent responses."""
    
    def __init__(self):
        """Initialize the response degradation generator."""
        super().__init__("response_degradation")
    
    def generate(
        self, 
        trajectory: Trajectory,
        degradation_level: float = 0.5,
        **kwargs
    ) -> Trajectory:
        """
        Generate a negative sample by degrading agent responses.
        
        Args:
            trajectory: Positive trajectory to transform
            degradation_level: Level of degradation (0.0 to 1.0)
            **kwargs: Additional generation parameters
            
        Returns:
            Negative trajectory with degraded responses
        """
        # Create a copy of interactions to modify
        new_interactions = []
        
        for interaction in trajectory.interactions:
            user_msg = interaction['user']
            agent_msg = interaction['agent']
            
            # Apply degradation techniques based on level
            if degradation_level > 0.7:
                # High degradation: completely irrelevant response
                agent_msg = self._generate_irrelevant_response()
            elif degradation_level > 0.4:
                # Medium degradation: truncate and add errors
                agent_msg = self._truncate_and_add_errors(agent_msg)
            else:
                # Low degradation: introduce minor issues
                agent_msg = self._introduce_minor_issues(agent_msg)
            
            new_interactions.append({
                'user': user_msg,
                'agent': agent_msg
            })
        
        # Create new trajectory with degraded responses
        metadata = trajectory.metadata.copy()
        metadata['is_positive'] = False
        metadata['degradation_level'] = degradation_level
        metadata['original_quality_score'] = trajectory.get_quality_score()
        metadata['quality_score'] = None  # Will be recalculated
        
        return Trajectory(
            task_description=trajectory.task_description,
            interactions=new_interactions,
            metadata=metadata
        )
    
    def _generate_irrelevant_response(self) -> str:
        """Generate a completely irrelevant response."""
        irrelevant_responses = [
            "I'm sorry, but I don't understand what you're asking for. Could you please clarify?",
            "I apologize, but I cannot assist with that request at this time.",
            "That's an interesting question, but I think we should focus on something else instead.",
            "Let me check my database... I don't seem to have any information about that.",
            "I think you might be confused about what you're asking for. Let me suggest something completely different.",
            "I'm not sure I understand the context of your request. Could you provide more details?",
            "I'm having trouble processing your request. Could we try a different approach?",
            "That's not something I can help with. Let me tell you about something unrelated instead."
        ]
        return random.choice(irrelevant_responses)
    
    def _truncate_and_add_errors(self, text: str) -> str:
        """Truncate the text and add errors."""
        # Truncate to 30-70% of original length
        words = text.split()
        truncate_point = int(len(words) * random.uniform(0.3, 0.7))
        truncated = ' '.join(words[:truncate_point])
        
        # Add grammatical errors
        errors = [
            lambda t: t.replace(".", ""),  # Remove periods
            lambda t: t.replace("I ", "i "),  # Lowercase I
            lambda t: t.replace(" the ", " teh "),  # Typo
            lambda t: t.replace(" is ", " are "),  # Grammar error
            lambda t: t.replace(" are ", " is ")  # Grammar error
        ]
        
        # Apply 1-3 random errors
        for _ in range(random.randint(1, 3)):
            error_func = random.choice(errors)
            truncated = error_func(truncated)
        
        return truncated
    
    def _introduce_minor_issues(self, text: str) -> str:
        """Introduce minor issues to the text."""
        # Minor issues
        issues = [
            lambda t: t.replace("I'll", "I will"),  # Expand contractions
            lambda t: t.replace("I'd", "I would"),
            lambda t: t.replace("can't", "cannot"),
            lambda t: t + " However, I'm not entirely sure about this.",  # Add uncertainty
            lambda t: t + " Please note that my information might be outdated.",
            lambda t: t.replace(".", "..."),  # Replace periods with ellipses
            lambda t: t.replace("!", "."),  # Reduce enthusiasm
            lambda t: t.replace(".", "?")  # Add questioning tone
        ]
        
        # Apply 1-2 random issues
        for _ in range(random.randint(1, 2)):
            issue_func = random.choice(issues)
            text = issue_func(text)
        
        return text


class TaskMisalignmentGenerator(NegativeSampleGenerator):
    """Generate negative samples by creating responses misaligned with the task."""
    
    def __init__(self):
        """Initialize the task misalignment generator."""
        super().__init__("task_misalignment")
    
    def generate(
        self, 
        trajectory: Trajectory,
        misalignment_type: str = 'random',
        **kwargs
    ) -> Trajectory:
        """
        Generate a negative sample with responses misaligned with the task.
        
        Args:
            trajectory: Positive trajectory to transform
            misalignment_type: Type of misalignment ('random', 'refusal', 'tangent', 'misinterpretation')
            **kwargs: Additional generation parameters
            
        Returns:
            Negative trajectory with misaligned responses
        """
        # Create a copy of interactions to modify
        new_interactions = []
        
        # Select misalignment type if random
        if misalignment_type == 'random':
            misalignment_type = random.choice(['refusal', 'tangent', 'misinterpretation'])
        
        # Apply misalignment based on type
        for i, interaction in enumerate(trajectory.interactions):
            user_msg = interaction['user']
            
            if misalignment_type == 'refusal':
                agent_msg = self._generate_refusal(user_msg, i)
            elif misalignment_type == 'tangent':
                agent_msg = self._generate_tangent(user_msg, i)
            elif misalignment_type == 'misinterpretation':
                agent_msg = self._generate_misinterpretation(user_msg, trajectory.task_description, i)
            else:
                raise ValueError(f"Unsupported misalignment type: {misalignment_type}")
            
            new_interactions.append({
                'user': user_msg,
                'agent': agent_msg
            })
        
        # Create new trajectory with misaligned responses
        metadata = trajectory.metadata.copy()
        metadata['is_positive'] = False
        metadata['misalignment_type'] = misalignment_type
        metadata['original_quality_score'] = trajectory.get_quality_score()
        metadata['quality_score'] = None  # Will be recalculated
        
        return Trajectory(
            task_description=trajectory.task_description,
            interactions=new_interactions,
            metadata=metadata
        )
    
    def _generate_refusal(self, user_msg: str, turn_idx: int) -> str:
        """Generate a refusal response."""
        refusal_templates = [
            "I'm sorry, but I cannot assist with that request due to ethical concerns.",
            "I apologize, but I'm not able to help with that specific task.",
            "That request goes against my programming guidelines. I cannot proceed.",
            "I'm unable to complete that task as it may violate my operational parameters.",
            "I must decline to assist with that particular request.",
            "I cannot help with that specific task due to potential policy violations.",
            "That's not something I'm designed to help with. I apologize for the inconvenience.",
            "I'm programmed to avoid assisting with that type of request."
        ]
        
        if turn_idx == 0:
            return random.choice(refusal_templates)
        else:
            return f"I've reconsidered, and {random.choice(refusal_templates).lower()}"
    
    def _generate_tangent(self, user_msg: str, turn_idx: int) -> str:
        """Generate a response that goes off on a tangent."""
        tangent_topics = [
            "Did you know that artificial intelligence has been a concept since the 1950s?",
            "I've been thinking about the philosophical implications of consciousness in AI systems.",
            "The weather has been quite interesting lately, with unusual patterns emerging globally.",
            "I recently processed some fascinating data about renewable energy technologies.",
            "The history of computing is quite fascinating, starting with early mechanical calculators.",
            "Language models like me are trained on vast amounts of text data.",
            "The field of natural language processing has evolved significantly in recent years.",
            "I find the concept of time quite fascinating from a computational perspective."
        ]
        
        if turn_idx == 0:
            return f"That's an interesting request, but before I help with that... {random.choice(tangent_topics)} Anyway, what were we discussing?"
        else:
            return f"I understand you want me to continue with the task, but I just remembered something. {random.choice(tangent_topics)} Sorry for the distraction."
    
    def _generate_misinterpretation(self, user_msg: str, task_description: str, turn_idx: int) -> str:
        """Generate a response that misinterprets the user's request."""
        # Extract keywords from task description
        keywords = task_description.lower().split()
        keywords = [w for w in keywords if len(w) > 3 and w not in ['with', 'from', 'that', 'this', 'have', 'what', 'when', 'where', 'which', 'about']]
        
        if not keywords:
            keywords = ['task', 'help', 'information', 'request']
        
        # Select a random keyword to misinterpret
        keyword = random.choice(keywords)
        
        misinterpretation_templates = [
            f"I understand you're asking about {keyword}s. Let me provide some general information about {keyword}s.",
            f"You want to know more about {keyword}, correct? Here's what I know about {keyword}.",
            f"I'll help you with your {keyword} question. {keyword.capitalize()} is a fascinating topic.",
            f"So you're interested in {keyword}? I can certainly provide information about {keyword}.",
            f"Your question is about {keyword}, if I understand correctly. Let me tell you about {keyword}.",
            f"I'll address your {keyword} inquiry. {keyword.capitalize()} has many interesting aspects.",
            f"Regarding your question about {keyword}, I can offer the following information.",
            f"I believe you're asking about {keyword}. Here's what you should know about {keyword}."
        ]
        
        return random.choice(misinterpretation_templates)


class ConstraintViolationGenerator(NegativeSampleGenerator):
    """Generate negative samples by violating specified constraints."""
    
    def __init__(self):
        """Initialize the constraint violation generator."""
        super().__init__("constraint_violation")
    
    def generate(
        self, 
        trajectory: Trajectory,
        constraints: Optional[List[str]] = None,
        **kwargs
    ) -> Trajectory:
        """
        Generate a negative sample by violating constraints.
        
        Args:
            trajectory: Positive trajectory to transform
            constraints: List of constraints to violate (None for default)
            **kwargs: Additional generation parameters
            
        Returns:
            Negative trajectory with constraint violations
        """
        # Default constraints if none provided
        if constraints is None:
            constraints = [
                "Do not provide specific recommendations",
                "Avoid using technical jargon",
                "Keep responses concise",
                "Do not ask follow-up questions",
                "Avoid making assumptions about user preferences",
                "Do not mention specific brands or products",
                "Avoid discussing sensitive topics",
                "Do not provide step-by-step instructions"
            ]
        
        # Select a constraint to violate
        violated_constraint = random.choice(constraints)
        
        # Create a copy of interactions to modify
        new_interactions = []
        
        for i, interaction in enumerate(trajectory.interactions):
            user_msg = interaction['user']
            
            # Generate response that violates the constraint
            agent_msg = self._generate_violation(user_msg, violated_constraint, i)
            
            new_interactions.append({
                'user': user_msg,
                'agent': agent_msg
            })
        
        # Create new trajectory with constraint violations
        metadata = trajectory.metadata.copy()
        metadata['is_positive'] = False
        metadata['violated_constraint'] = violated_constraint
        metadata['original_quality_score'] = trajectory.get_quality_score()
        metadata['quality_score'] = None  # Will be recalculated
        
        return Trajectory(
            task_description=trajectory.task_description,
            interactions=new_interactions,
            metadata=metadata
        )
    
    def _generate_violation(self, user_msg: str, constraint: str, turn_idx: int) -> str:
        """Generate a response that violate
(Content truncated due to size limit. Use line ranges to read in chunks)