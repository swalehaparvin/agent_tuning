"""
Domain-Specific Calibration Module for LLMs

This module implements calibration techniques for improving uncertainty estimates
across different domains, focusing on temperature scaling and domain adaptation.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Union, Optional, Tuple
from scipy.optimize import minimize_scalar

class Calibrator:
    """Base class for calibration methods."""
    
    def __init__(self, name: str):
        """
        Initialize the calibrator.
        
        Args:
            name: Name of the calibration method
        """
        self.name = name
        self.is_fitted = False
    
    def fit(self, confidences: List[float], accuracies: List[bool]) -> None:
        """
        Fit the calibrator to the provided data.
        
        Args:
            confidences: List of confidence scores
            accuracies: List of boolean accuracy indicators
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def calibrate(self, confidences: List[float]) -> List[float]:
        """
        Calibrate the provided confidence scores.
        
        Args:
            confidences: List of confidence scores
            
        Returns:
            Calibrated confidence scores
        """
        raise NotImplementedError("Subclasses must implement this method")


class TemperatureScaling(Calibrator):
    """Calibration using temperature scaling."""
    
    def __init__(self):
        """Initialize the temperature scaling calibrator."""
        super().__init__("temperature_scaling")
        self.temperature = 1.0
    
    def _nll_loss(self, temperature: float, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """
        Calculate negative log likelihood loss for temperature scaling.
        
        Args:
            temperature: Temperature parameter
            confidences: Array of confidence scores
            accuracies: Array of boolean accuracy indicators
            
        Returns:
            Negative log likelihood loss
        """
        # Apply temperature scaling
        scaled_confidences = np.clip(confidences / temperature, 1e-10, 1.0 - 1e-10)
        
        # Calculate binary cross-entropy loss
        loss = -np.mean(
            accuracies * np.log(scaled_confidences) + 
            (1 - accuracies) * np.log(1 - scaled_confidences)
        )
        
        return loss
    
    def fit(self, confidences: List[float], accuracies: List[bool]) -> None:
        """
        Fit the temperature parameter to the provided data.
        
        Args:
            confidences: List of confidence scores
            accuracies: List of boolean accuracy indicators
        """
        if not confidences or len(confidences) != len(accuracies):
            raise ValueError("Confidences and accuracies must have the same non-zero length")
        
        # Convert to numpy arrays
        conf_array = np.array(confidences)
        acc_array = np.array(accuracies, dtype=float)
        
        # Optimize temperature parameter
        result = minimize_scalar(
            lambda t: self._nll_loss(t, conf_array, acc_array),
            bounds=(0.1, 10.0),
            method='bounded'
        )
        
        self.temperature = result.x
        self.is_fitted = True
        
        print(f"Fitted temperature parameter: {self.temperature:.4f}")
    
    def calibrate(self, confidences: List[float]) -> List[float]:
        """
        Calibrate the provided confidence scores using temperature scaling.
        
        Args:
            confidences: List of confidence scores
            
        Returns:
            Calibrated confidence scores
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        
        # Apply temperature scaling
        calibrated = [min(max(conf / self.temperature, 1e-10), 1.0 - 1e-10) for conf in confidences]
        
        return calibrated


class DomainAdaptiveCalibration(Calibrator):
    """Calibration using domain-adaptive techniques."""
    
    def __init__(self, source_domain: str, target_domain: str):
        """
        Initialize the domain-adaptive calibrator.
        
        Args:
            source_domain: Source domain name
            target_domain: Target domain name
        """
        super().__init__("domain_adaptive_calibration")
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.source_temperature = 1.0
        self.target_temperature = 1.0
        self.domain_shift_factor = 1.0
    
    def fit(
        self, 
        source_confidences: List[float], 
        source_accuracies: List[bool],
        target_confidences: Optional[List[float]] = None,
        target_accuracies: Optional[List[bool]] = None
    ) -> None:
        """
        Fit the domain-adaptive calibrator to the provided data.
        
        Args:
            source_confidences: List of confidence scores from source domain
            source_accuracies: List of boolean accuracy indicators from source domain
            target_confidences: List of confidence scores from target domain (if available)
            target_accuracies: List of boolean accuracy indicators from target domain (if available)
        """
        # Fit source domain temperature
        source_calibrator = TemperatureScaling()
        source_calibrator.fit(source_confidences, source_accuracies)
        self.source_temperature = source_calibrator.temperature
        
        # If target domain data is available, fit target temperature
        if target_confidences and target_accuracies:
            target_calibrator = TemperatureScaling()
            target_calibrator.fit(target_confidences, target_accuracies)
            self.target_temperature = target_calibrator.temperature
            
            # Calculate domain shift factor
            self.domain_shift_factor = self.target_temperature / self.source_temperature
        else:
            # Default domain shift factor based on heuristics
            # This is a simplified approach; in a real system, this would be more sophisticated
            self.domain_shift_factor = 1.2  # Assuming target domain is slightly more uncertain
            self.target_temperature = self.source_temperature * self.domain_shift_factor
        
        self.is_fitted = True
        
        print(f"Fitted source temperature: {self.source_temperature:.4f}")
        print(f"Fitted target temperature: {self.target_temperature:.4f}")
        print(f"Domain shift factor: {self.domain_shift_factor:.4f}")
    
    def calibrate(self, confidences: List[float], domain: str = None) -> List[float]:
        """
        Calibrate the provided confidence scores using domain-adaptive calibration.
        
        Args:
            confidences: List of confidence scores
            domain: Domain of the confidences ('source' or 'target', defaults to target)
            
        Returns:
            Calibrated confidence scores
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        
        # Determine which temperature to use
        if domain == "source":
            temperature = self.source_temperature
        else:
            temperature = self.target_temperature
        
        # Apply temperature scaling
        calibrated = [min(max(conf / temperature, 1e-10), 1.0 - 1e-10) for conf in confidences]
        
        return calibrated


class EnsembleCalibration(Calibrator):
    """Calibration using an ensemble of calibration methods."""
    
    def __init__(self, calibrators: List[Calibrator], weights: Optional[List[float]] = None):
        """
        Initialize the ensemble calibrator.
        
        Args:
            calibrators: List of calibrator instances
            weights: List of weights for each calibrator (None for equal weights)
        """
        super().__init__("ensemble_calibration")
        self.calibrators = calibrators
        
        # Initialize weights
        if weights is None:
            self.weights = [1.0 / len(calibrators)] * len(calibrators)
        else:
            if len(weights) != len(calibrators):
                raise ValueError("Number of weights must match number of calibrators")
            
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def fit(self, confidences: List[float], accuracies: List[bool]) -> None:
        """
        Fit all calibrators in the ensemble.
        
        Args:
            confidences: List of confidence scores
            accuracies: List of boolean accuracy indicators
        """
        for calibrator in self.calibrators:
            calibrator.fit(confidences, accuracies)
        
        self.is_fitted = True
    
    def calibrate(self, confidences: List[float]) -> List[float]:
        """
        Calibrate the provided confidence scores using the ensemble.
        
        Args:
            confidences: List of confidence scores
            
        Returns:
            Calibrated confidence scores
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        
        # Get calibrated confidences from each calibrator
        all_calibrated = []
        for calibrator in self.calibrators:
            all_calibrated.append(calibrator.calibrate(confidences))
        
        # Combine calibrated confidences using weights
        calibrated = []
        for i in range(len(confidences)):
            weighted_sum = sum(self.weights[j] * all_calibrated[j][i] for j in range(len(self.calibrators)))
            calibrated.append(weighted_sum)
        
        return calibrated


# Factory function to create calibrators
def create_calibrator(method: str, **kwargs) -> Calibrator:
    """
    Create a calibrator based on the specified method.
    
    Args:
        method: Name of the calibration method
        **kwargs: Additional arguments for the calibrator
        
    Returns:
        Calibrator instance
    """
    if method == "temperature_scaling":
        return TemperatureScaling()
    elif method == "domain_adaptive":
        if "source_domain" not in kwargs or "target_domain" not in kwargs:
            raise ValueError("Domain-adaptive calibration requires source_domain and target_domain")
        return DomainAdaptiveCalibration(kwargs["source_domain"], kwargs["target_domain"])
    elif method == "ensemble":
        if "calibrators" not in kwargs:
            raise ValueError("Ensemble calibration requires a list of calibrators")
        return EnsembleCalibration(kwargs["calibrators"], kwargs.get("weights"))
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
