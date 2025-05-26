"""
Uncertainty Quantification Module for LLMs

This module implements various uncertainty quantification methods for large language models,
including softmax confidence, Monte Carlo dropout, ensemble disagreement, and calibration metrics.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Union, Optional
from scipy.special import softmax
import torch.nn.functional as F

class UncertaintyQuantifier:
    """Base class for uncertainty quantification methods."""
    
    def __init__(self, name: str):
        """
        Initialize the uncertainty quantifier.
        
        Args:
            name: Name of the uncertainty quantification method
        """
        self.name = name
    
    def quantify(self, model_outputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Quantify uncertainty in model outputs.
        
        Args:
            model_outputs: Outputs from the LLM interface
            
        Returns:
            Dictionary of uncertainty metrics
        """
        raise NotImplementedError("Subclasses must implement this method")


class SoftmaxConfidence(UncertaintyQuantifier):
    """Uncertainty quantification based on softmax confidence scores."""
    
    def __init__(self):
        """Initialize the softmax confidence quantifier."""
        super().__init__("softmax_confidence")
    
    def quantify(self, model_outputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Quantify uncertainty using softmax confidence scores.
        
        Args:
            model_outputs: Outputs from the LLM interface, must include logits
            
        Returns:
            Dictionary of uncertainty metrics:
                - mean_confidence: Average confidence across tokens
                - min_confidence: Minimum confidence across tokens
                - entropy: Average entropy of token distributions
        """
        if "logits" not in model_outputs:
            raise ValueError("Model outputs must include logits for softmax confidence")
        
        logits = model_outputs["logits"][0]  # Use first sample's logits
        
        # Calculate softmax probabilities and confidence metrics
        confidences = []
        entropies = []
        
        for token_logits in logits:
            probs = softmax(token_logits, axis=-1)
            max_prob = np.max(probs)
            confidences.append(max_prob)
            
            # Calculate entropy of the probability distribution
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        
        return {
            "mean_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "entropy": float(np.mean(entropies))
        }


class MonteCarloDropout(UncertaintyQuantifier):
    """Uncertainty quantification based on Monte Carlo dropout sampling."""
    
    def __init__(self):
        """Initialize the Monte Carlo dropout quantifier."""
        super().__init__("mc_dropout")
    
    def quantify(self, model_outputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Quantify uncertainty using Monte Carlo dropout sampling.
        
        Args:
            model_outputs: Outputs from the LLM interface, must include multiple samples
            
        Returns:
            Dictionary of uncertainty metrics:
                - sample_variance: Variance across different samples
                - sample_diversity: Lexical diversity across samples
        """
        if "samples" not in model_outputs or len(model_outputs["samples"]) <= 1:
            raise ValueError("Model outputs must include multiple samples for MC dropout")
        
        samples = model_outputs["samples"]
        
        # Calculate sample diversity using token overlap
        from nltk.tokenize import word_tokenize
        try:
            tokenized_samples = [set(word_tokenize(sample.lower())) for sample in samples]
        except:
            # Fallback to simple whitespace tokenization if nltk is not available
            tokenized_samples = [set(sample.lower().split()) for sample in samples]
        
        # Calculate Jaccard similarity between all pairs of samples
        similarities = []
        for i in range(len(tokenized_samples)):
            for j in range(i+1, len(tokenized_samples)):
                intersection = len(tokenized_samples[i].intersection(tokenized_samples[j]))
                union = len(tokenized_samples[i].union(tokenized_samples[j]))
                if union > 0:
                    similarities.append(intersection / union)
                else:
                    similarities.append(1.0)  # Empty sets are considered identical
        
        # Convert similarity to diversity (1 - similarity)
        diversity = 1.0 - np.mean(similarities) if similarities else 0.0
        
        # Calculate variance in sample lengths as another diversity metric
        sample_lengths = [len(sample) for sample in samples]
        length_variance = np.var(sample_lengths) if len(sample_lengths) > 1 else 0.0
        
        return {
            "sample_diversity": float(diversity),
            "length_variance": float(length_variance),
            "num_samples": len(samples)
        }


class EnsembleDisagreement(UncertaintyQuantifier):
    """Uncertainty quantification based on ensemble disagreement."""
    
    def __init__(self):
        """Initialize the ensemble disagreement quantifier."""
        super().__init__("ensemble_disagreement")
    
    def quantify(self, ensemble_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Quantify uncertainty using ensemble disagreement.
        
        Args:
            ensemble_outputs: List of outputs from different models
            
        Returns:
            Dictionary of uncertainty metrics:
                - response_diversity: Lexical diversity across model responses
                - confidence_variance: Variance in confidence scores across models
        """
        if not ensemble_outputs or len(ensemble_outputs) <= 1:
            raise ValueError("Ensemble outputs must include results from multiple models")
        
        # Extract primary responses from each model
        responses = [output["response"] for output in ensemble_outputs]
        
        # Calculate response diversity using token overlap (similar to MC dropout)
        from nltk.tokenize import word_tokenize
        try:
            tokenized_responses = [set(word_tokenize(response.lower())) for response in responses]
        except:
            # Fallback to simple whitespace tokenization if nltk is not available
            tokenized_responses = [set(response.lower().split()) for response in responses]
        
        # Calculate Jaccard similarity between all pairs of responses
        similarities = []
        for i in range(len(tokenized_responses)):
            for j in range(i+1, len(tokenized_responses)):
                intersection = len(tokenized_responses[i].intersection(tokenized_responses[j]))
                union = len(tokenized_responses[i].union(tokenized_responses[j]))
                if union > 0:
                    similarities.append(intersection / union)
                else:
                    similarities.append(1.0)  # Empty sets are considered identical
        
        # Convert similarity to diversity (1 - similarity)
        diversity = 1.0 - np.mean(similarities) if similarities else 0.0
        
        # Extract confidence scores if available
        confidences = []
        for output in ensemble_outputs:
            if "mean_confidence" in output:
                confidences.append(output["mean_confidence"])
        
        # Calculate variance in confidence scores
        confidence_variance = np.var(confidences) if len(confidences) > 1 else 0.0
        
        return {
            "response_diversity": float(diversity),
            "confidence_variance": float(confidence_variance),
            "num_models": len(ensemble_outputs)
        }


class CalibrationMetrics(UncertaintyQuantifier):
    """Uncertainty quantification based on calibration metrics."""
    
    def __init__(self):
        """Initialize the calibration metrics quantifier."""
        super().__init__("calibration_metrics")
    
    def expected_calibration_error(
        self, 
        confidences: List[float], 
        accuracies: List[bool], 
        num_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            confidences: List of confidence scores
            accuracies: List of boolean accuracy indicators
            num_bins: Number of bins for binning confidences
            
        Returns:
            Expected Calibration Error
        """
        if len(confidences) != len(accuracies):
            raise ValueError("Confidences and accuracies must have the same length")
        
        if not confidences:
            return 0.0
        
        # Create bins and calculate ECE
        bin_indices = np.digitize(confidences, np.linspace(0, 1, num_bins))
        ece = 0.0
        
        for bin_idx in range(1, num_bins + 1):
            bin_mask = (bin_indices == bin_idx)
            if np.any(bin_mask):
                bin_confidences = np.array(confidences)[bin_mask]
                bin_accuracies = np.array(accuracies)[bin_mask]
                bin_confidence = np.mean(bin_confidences)
                bin_accuracy = np.mean(bin_accuracies)
                bin_size = np.sum(bin_mask)
                
                # Weighted absolute difference between confidence and accuracy
                ece += (bin_size / len(confidences)) * np.abs(bin_confidence - bin_accuracy)
        
        return float(ece)
    
    def maximum_calibration_error(
        self, 
        confidences: List[float], 
        accuracies: List[bool], 
        num_bins: int = 10
    ) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        
        Args:
            confidences: List of confidence scores
            accuracies: List of boolean accuracy indicators
            num_bins: Number of bins for binning confidences
            
        Returns:
            Maximum Calibration Error
        """
        if len(confidences) != len(accuracies):
            raise ValueError("Confidences and accuracies must have the same length")
        
        if not confidences:
            return 0.0
        
        # Create bins and calculate MCE
        bin_indices = np.digitize(confidences, np.linspace(0, 1, num_bins))
        max_ce = 0.0
        
        for bin_idx in range(1, num_bins + 1):
            bin_mask = (bin_indices == bin_idx)
            if np.any(bin_mask):
                bin_confidences = np.array(confidences)[bin_mask]
                bin_accuracies = np.array(accuracies)[bin_mask]
                bin_confidence = np.mean(bin_confidences)
                bin_accuracy = np.mean(bin_accuracies)
                
                # Absolute difference between confidence and accuracy
                ce = np.abs(bin_confidence - bin_accuracy)
                max_ce = max(max_ce, ce)
        
        return float(max_ce)
    
    def quantify(
        self, 
        confidences: List[float], 
        accuracies: List[bool]
    ) -> Dict[str, float]:
        """
        Quantify uncertainty using calibration metrics.
        
        Args:
            confidences: List of confidence scores
            accuracies: List of boolean accuracy indicators
            
        Returns:
            Dictionary of calibration metrics:
                - ece: Expected Calibration Error
                - mce: Maximum Calibration Error
        """
        return {
            "ece": self.expected_calibration_error(confidences, accuracies),
            "mce": self.maximum_calibration_error(confidences, accuracies)
        }


# Factory function to create uncertainty quantifiers
def create_uncertainty_quantifier(method: str) -> UncertaintyQuantifier:
    """
    Create an uncertainty quantifier based on the specified method.
    
    Args:
        method: Name of the uncertainty quantification method
        
    Returns:
        Uncertainty quantifier instance
    """
    if method == "softmax_confidence":
        return SoftmaxConfidence()
    elif method == "mc_dropout":
        return MonteCarloDropout()
    elif method == "ensemble_disagreement":
        return EnsembleDisagreement()
    elif method == "calibration_metrics":
        return CalibrationMetrics()
    else:
        raise ValueError(f"Unsupported uncertainty quantification method: {method}")
