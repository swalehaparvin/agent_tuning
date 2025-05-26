"""
Evaluation Framework for Cross-Domain Uncertainty Quantification

This module provides functionality for evaluating uncertainty quantification methods
across different domains, including metrics for uncertainty quality and cross-domain performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Union, Optional, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class UncertaintyEvaluator:
    """Evaluator for uncertainty quantification methods."""
    
    def __init__(self, name: str):
        """
        Initialize the uncertainty evaluator.
        
        Args:
            name: Name of the evaluation method
        """
        self.name = name
    
    def evaluate(
        self, 
        uncertainties: List[float], 
        correctness: List[bool]
    ) -> Dict[str, float]:
        """
        Evaluate uncertainty estimates against correctness.
        
        Args:
            uncertainties: List of uncertainty scores (higher means more uncertain)
            correctness: List of boolean correctness indicators
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement this method")


class CalibrationEvaluator(UncertaintyEvaluator):
    """Evaluator for calibration quality."""
    
    def __init__(self):
        """Initialize the calibration evaluator."""
        super().__init__("calibration_evaluator")
    
    def expected_calibration_error(
        self, 
        confidences: List[float], 
        correctness: List[bool], 
        num_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            confidences: List of confidence scores
            correctness: List of boolean correctness indicators
            num_bins: Number of bins for binning confidences
            
        Returns:
            Expected Calibration Error
        """
        if len(confidences) != len(correctness):
            raise ValueError("Confidences and correctness must have the same length")
        
        if not confidences:
            return 0.0
        
        # Create bins and calculate ECE
        bin_indices = np.digitize(confidences, np.linspace(0, 1, num_bins))
        ece = 0.0
        
        for bin_idx in range(1, num_bins + 1):
            bin_mask = (bin_indices == bin_idx)
            if np.any(bin_mask):
                bin_confidences = np.array(confidences)[bin_mask]
                bin_correctness = np.array(correctness)[bin_mask]
                bin_confidence = np.mean(bin_confidences)
                bin_accuracy = np.mean(bin_correctness)
                bin_size = np.sum(bin_mask)
                
                # Weighted absolute difference between confidence and accuracy
                ece += (bin_size / len(confidences)) * np.abs(bin_confidence - bin_accuracy)
        
        return float(ece)
    
    def maximum_calibration_error(
        self, 
        confidences: List[float], 
        correctness: List[bool], 
        num_bins: int = 10
    ) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        
        Args:
            confidences: List of confidence scores
            correctness: List of boolean correctness indicators
            num_bins: Number of bins for binning confidences
            
        Returns:
            Maximum Calibration Error
        """
        if len(confidences) != len(correctness):
            raise ValueError("Confidences and correctness must have the same length")
        
        if not confidences:
            return 0.0
        
        # Create bins and calculate MCE
        bin_indices = np.digitize(confidences, np.linspace(0, 1, num_bins))
        max_ce = 0.0
        
        for bin_idx in range(1, num_bins + 1):
            bin_mask = (bin_indices == bin_idx)
            if np.any(bin_mask):
                bin_confidences = np.array(confidences)[bin_mask]
                bin_correctness = np.array(correctness)[bin_mask]
                bin_confidence = np.mean(bin_confidences)
                bin_accuracy = np.mean(bin_correctness)
                
                # Absolute difference between confidence and accuracy
                ce = np.abs(bin_confidence - bin_accuracy)
                max_ce = max(max_ce, ce)
        
        return float(max_ce)
    
    def evaluate(
        self, 
        confidences: List[float], 
        correctness: List[bool]
    ) -> Dict[str, float]:
        """
        Evaluate calibration quality.
        
        Args:
            confidences: List of confidence scores
            correctness: List of boolean correctness indicators
            
        Returns:
            Dictionary of calibration metrics:
                - ece: Expected Calibration Error
                - mce: Maximum Calibration Error
        """
        return {
            "ece": self.expected_calibration_error(confidences, correctness),
            "mce": self.maximum_calibration_error(confidences, correctness)
        }
    
    def plot_reliability_diagram(
        self, 
        confidences: List[float], 
        correctness: List[bool], 
        num_bins: int = 10,
        title: str = "Reliability Diagram",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot a reliability diagram for calibration visualization.
        
        Args:
            confidences: List of confidence scores
            correctness: List of boolean correctness indicators
            num_bins: Number of bins for binning confidences
            title: Title for the plot
            save_path: Path to save the plot (None to display)
        """
        if len(confidences) != len(correctness):
            raise ValueError("Confidences and correctness must have the same length")
        
        # Create bins
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges[:-1])
        
        # Calculate accuracy and confidence for each bin
        bin_accuracies = []
        bin_confidences = []
        bin_sizes = []
        
        for bin_idx in range(1, num_bins + 1):
            bin_mask = (bin_indices == bin_idx)
            if np.any(bin_mask):
                bin_confidences.append(np.mean(np.array(confidences)[bin_mask]))
                bin_accuracies.append(np.mean(np.array(correctness)[bin_mask]))
                bin_sizes.append(np.sum(bin_mask))
            else:
                bin_confidences.append(0)
                bin_accuracies.append(0)
                bin_sizes.append(0)
        
        # Plot reliability diagram
        plt.figure(figsize=(10, 6))
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Plot bin accuracies vs. confidences
        plt.bar(
            bin_edges[:-1], 
            bin_accuracies, 
            width=1/num_bins, 
            align='edge', 
            alpha=0.7, 
            label='Observed Accuracy'
        )
        
        # Plot confidence histogram
        ax2 = plt.twinx()
        ax2.hist(
            confidences, 
            bins=bin_edges, 
            alpha=0.3, 
            color='gray', 
            label='Confidence Histogram'
        )
        
        # Calculate ECE and MCE
        ece = self.expected_calibration_error(confidences, correctness, num_bins)
        mce = self.maximum_calibration_error(confidences, correctness, num_bins)
        
        # Add ECE and MCE to title
        plt.title(f"{title}\nECE: {ece:.4f}, MCE: {mce:.4f}")
        
        # Add labels and legend
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        ax2.set_ylabel('Count')
        
        # Add legend
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')
        
        # Save or display the plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


class SelectivePredictionEvaluator(UncertaintyEvaluator):
    """Evaluator for selective prediction performance."""
    
    def __init__(self):
        """Initialize the selective prediction evaluator."""
        super().__init__("selective_prediction_evaluator")
    
    def evaluate(
        self, 
        uncertainties: List[float], 
        correctness: List[bool]
    ) -> Dict[str, float]:
        """
        Evaluate selective prediction performance.
        
        Args:
            uncertainties: List of uncertainty scores (higher means more uncertain)
            correctness: List of boolean correctness indicators
            
        Returns:
            Dictionary of selective prediction metrics:
                - auroc: Area Under ROC Curve for predicting errors
                - auprc: Area Under Precision-Recall Curve for predicting errors
                - uncertainty_error_correlation: Correlation between uncertainty and errors
        """
        if len(uncertainties) != len(correctness):
            raise ValueError("Uncertainties and correctness must have the same length")
        
        if not uncertainties:
            return {
                "auroc": 0.5,
                "auprc": 0.5,
                "uncertainty_error_correlation": 0.0
            }
        
        # Convert correctness to errors (1 for error, 0 for correct)
        errors = [1 - int(c) for c in correctness]
        
        # Calculate AUROC for predicting errors
        try:
            auroc = roc_auc_score(errors, uncertainties)
        except:
            # Handle case where all predictions are correct or all are wrong
            auroc = 0.5
        
        # Calculate AUPRC for predicting errors
        try:
            precision, recall, _ = precision_recall_curve(errors, uncertainties)
            auprc = auc(recall, precision)
        except:
            # Handle case where all predictions are correct or all are wrong
            auprc = 0.5
        
        # Calculate correlation between uncertainty and errors
        uncertainty_error_correlation = np.corrcoef(uncertainties, errors)[0, 1]
        
        return {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "uncertainty_error_correlation": float(uncertainty_error_correlation)
        }
    
    def plot_selective_prediction_curve(
        self, 
        uncertainties: List[float], 
        correctness: List[bool],
        title: str = "Selective Prediction Performance",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot a selective prediction curve.
        
        Args:
            uncertainties: List of uncertainty scores (higher means more uncertain)
            correctness: List of boolean correctness indicators
            title: Title for the plot
            save_path: Path to save the plot (None to display)
        """
        if len(uncertainties) != len(correctness):
            raise ValueError("Uncertainties and correctness must have the same length")
        
        # Sort by uncertainty (ascending)
        sorted_indices = np.argsort(uncertainties)
        sorted_correctness = np.array(correctness)[sorted_indices]
        
        # Calculate cumulative accuracy at different coverage levels
        coverages = np.linspace(0, 1, 100)
        accuracies = []
        
        for coverage in coverages:
            if coverage == 0:
                accuracies.append(1.0)  # Perfect accuracy at 0% coverage
            else:
                n_samples = int(coverage * len(sorted_correctness))
                if n_samples == 0:
                    accuracies.append(1.0)
                else:
                    accuracies.append(np.mean(sorted_correctness[:n_samples]))
        
        # Plot selective prediction curve
        plt.figure(figsize=(10, 6))
        plt.plot(coverages, accuracies, 'b-', linewidth=2)
        
        # Add reference line for random selection
        plt.plot([0, 1], [np.mean(correctness), np.mean(correctness)], 'k--', label='Random Selection')
        
        # Calculate AUROC
        metrics = self.evaluate(uncertainties, correctness)
        
        # Add AUROC to title
        plt.title(f"{title}\nAUROC: {metrics['auroc']:.4f}")
        
        # Add labels and legend
        plt.xlabel('Coverage')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        
        # Save or display the plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


class CrossDomainEvaluator:
    """Evaluator for cross-domain uncertainty performance."""
    
    def __init__(self):
        """Initialize the cross-domain evaluator."""
        self.name = "cross_domain_evaluator"
        self.calibration_evaluator = CalibrationEvaluator()
        self.selective_prediction_evaluator = SelectivePredictionEvaluator()
    
    def evaluate_domain_transfer(
        self, 
        source_uncertainties: List[float],
        source_correctness: List[bool],
        target_uncertainties: List[float],
        target_correctness: List[bool]
    ) -> Dict[str, float]:
        """
        Evaluate domain transfer performance.
        
        Args:
            source_uncertainties: List of uncertainty scores from source domain
            source_correctness: List of boolean correctness indicators from source domain
            target_uncertainties: List of uncertainty scores from target domain
            target_correctness: List of boolean correctness indicators from target domain
            
        Returns:
            Dictionary of domain transfer metrics:
                - source_auroc: AUROC in source domain
                - target_auroc: AUROC in target domain
                - transfer_degradation: Degradation in AUROC from source to target
                - source_ece: ECE in source domain
                - target_ece: ECE in target domain
                - calibration_shift: Shift in calibration from source to target
        """
        # Evaluate source domain
        source_selective = self.selective_prediction_evaluator.evaluate(
            source_uncertainties, source_correctness
        )
        source_calibration = self.calibration_evaluator.evaluate(
            [1 - u for u in source_uncertainties], source_correctness
        )
        
        # Evaluate target domain
        target_selective = self.selective_prediction_evaluator.evaluate(
            target_uncertainties, target_correctness
        )
        target_calibration = self.calibration_evaluator.evaluate(
            [1 - u for u in target_uncertainties], target_correctness
        )
        
        # Calculate transfer metrics
        transfer_degradation = source_selective["auroc"] - target_selective["auroc"]
        calibration_shift = target_calibration["ece"] - source_calibration["ece"]
        
        return {
            "source_auroc": source_selective["auroc"],
            "target_auroc": target_selective["auroc"],
            "transfer_degradation": float(transfer_degradation),
            "source_ece": source_calibration["ece"],
            "target_ece": target_calibration["ece"],
            "calibration_shift": float(calibration_shift)
        }
    
    def evaluate_all_domains(
        self, 
        domain_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate uncertainty performance across all domains.
        
        Args:
            domain_results: Dictionary mapping domain names to results
                Each result should contain:
                - uncertainties: List of uncertai
(Content truncated due to size limit. Use line ranges to read in chunks)