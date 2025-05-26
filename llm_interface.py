"""
LLM Interface Module for Cross-Domain Uncertainty Quantification

This module provides a unified interface for interacting with large language models,
supporting multiple model architectures and uncertainty quantification methods.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

class LLMInterface:
    """Interface for interacting with large language models with uncertainty quantification."""
    
    def __init__(
        self, 
        model_name: str,
        model_type: str = "causal",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_beams: int = 1
    ):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: Name of the Hugging Face model to use
            model_type: Type of model ('causal' or 'seq2seq')
            device: Device to run the model on ('cpu' or 'cuda')
            cache_dir: Directory to cache models
            max_length: Maximum length of generated sequences
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_beams: Number of beams for beam search
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )
        
        # Load model based on type
        if model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
        elif model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Response cache for efficiency
        self.response_cache = {}
    
    def generate(
        self, 
        prompt: str,
        num_samples: int = 1,
        return_logits: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate responses from the model with uncertainty quantification.
        
        Args:
            prompt: Input text prompt
            num_samples: Number of samples to generate (for MC methods)
            return_logits: Whether to return token logits
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing:
                - response: The generated text
                - samples: Multiple samples if num_samples > 1
                - logits: Token logits if return_logits is True
        """
        # Check cache first
        cache_key = (prompt, num_samples, return_logits, str(kwargs))
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_beams": self.num_beams,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        gen_kwargs.update(kwargs)
        
        # Generate multiple samples if requested
        samples = []
        all_logits = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    output_scores=return_logits,
                    return_dict_in_generate=True,
                    **gen_kwargs
                )
            
            # Extract generated tokens
            if self.model_type == "causal":
                gen_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
            else:
                gen_tokens = outputs.sequences[0]
            
            # Decode tokens to text
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            samples.append(gen_text)
            
            # Extract logits if requested
            if return_logits and hasattr(outputs, "scores"):
                all_logits.append([score.cpu().numpy() for score in outputs.scores])
        
        # Prepare result
        result = {
            "response": samples[0],  # Primary response is first sample
            "samples": samples
        }
        
        if return_logits:
            result["logits"] = all_logits
        
        # Cache result
        self.response_cache[cache_key] = result
        return result
    
    def batch_generate(
        self, 
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input text prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generation results for each prompt
        """
        results = []
        for prompt in tqdm(prompts, desc="Generating responses"):
            results.append(self.generate(prompt, **kwargs))
        return results
    
    def clear_cache(self):
        """Clear the response cache."""
        self.response_cache = {}
