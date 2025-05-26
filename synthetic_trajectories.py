"""
Synthetic Trajectory Generation Module for Agent Tuning Optimization Framework

This module provides functionality for generating synthetic agent interaction trajectories
based on task specifications to enhance the training data for agent tuning.
"""

import random
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from tqdm import tqdm

from data.trajectory_data import Trajectory, TrajectoryDataset
from models.llm_interface import LLMInterface

class SyntheticTrajectoryGenerator:
    """Base class for synthetic trajectory generation strategies."""
    
    def __init__(self, name: str):
        """
        Initialize the synthetic trajectory generator.
        
        Args:
            name: Name of the generator strategy
        """
        self.name = name
    
    def generate(
        self, 
        task_description: str,
        num_interactions: int = 3,
        **kwargs
    ) -> Trajectory:
        """
        Generate a synthetic trajectory for a given task.
        
        Args:
            task_description: Description of the task
            num_interactions: Number of interaction turns to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Synthetic trajectory
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def batch_generate(
        self, 
        task_descriptions: List[str],
        num_interactions: int = 3,
        **kwargs
    ) -> List[Trajectory]:
        """
        Generate synthetic trajectories for a batch of tasks.
        
        Args:
            task_descriptions: List of task descriptions
            num_interactions: Number of interaction turns to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of synthetic trajectories
        """
        synthetic_trajectories = []
        
        for task in tqdm(task_descriptions, desc=f"Generating synthetic trajectories with {self.name}"):
            synthetic_trajectories.append(self.generate(task, num_interactions, **kwargs))
        
        return synthetic_trajectories


class TemplateBasedGenerator(SyntheticTrajectoryGenerator):
    """Generate synthetic trajectories using predefined templates."""
    
    def __init__(self):
        """Initialize the template-based generator."""
        super().__init__("template_based")
        
        # User message templates
        self.initial_user_templates = [
            "I need help with {task}.",
            "Can you assist me with {task}?",
            "I'm trying to {task}. Can you help?",
            "I'd like your help with {task}.",
            "I'm working on {task} and need assistance."
        ]
        
        self.followup_user_templates = [
            "That sounds good. Can you provide more details?",
            "I like your approach. What's the next step?",
            "Thanks for the information. Can you elaborate on {aspect}?",
            "I appreciate your help. How should I proceed with {aspect}?",
            "That's helpful. Can you tell me more about {aspect}?"
        ]
        
        self.final_user_templates = [
            "This is exactly what I needed. Thank you!",
            "Perfect, that solves my problem. Thanks for your help!",
            "Great, I'll follow your advice. Thanks!",
            "That's very helpful. I appreciate your assistance!",
            "Thanks for walking me through this. I understand now."
        ]
        
        # Agent message templates
        self.initial_agent_templates = [
            "I'd be happy to help you with {task}. Could you provide more details about your specific requirements?",
            "I can definitely assist with {task}. Let me ask a few questions to better understand your needs.",
            "I'll help you with {task}. To get started, I'll need to gather some information.",
            "I can guide you through {task}. First, let's clarify what you're looking to accomplish.",
            "I'm here to help with {task}. Let's break this down into manageable steps."
        ]
        
        self.middle_agent_templates = [
            "Based on what you've shared, I recommend {recommendation}. This approach has several advantages: {advantages}.",
            "Given your requirements, the best option would be {recommendation}. Here's why: {advantages}.",
            "After analyzing your needs, I suggest {recommendation}. The benefits include {advantages}.",
            "Taking into account what you've mentioned, I'd recommend {recommendation}. This will help because {advantages}.",
            "From what I understand, {recommendation} would be the most suitable approach. The key benefits are {advantages}."
        ]
        
        self.final_agent_templates = [
            "To summarize, we've discussed {summary}. The next steps are {next_steps}. Is there anything else you'd like me to clarify?",
            "In conclusion, we've covered {summary}. You should now {next_steps}. Feel free to reach out if you have any questions.",
            "To wrap up, we've gone through {summary}. Moving forward, you can {next_steps}. Let me know if you need further assistance.",
            "In summary, we've addressed {summary}. Your action items are {next_steps}. Don't hesitate to ask if anything is unclear.",
            "To recap our discussion, we've explored {summary}. The recommended actions are {next_steps}. Is there anything else you'd like to know?"
        ]
        
        # Task aspects for template filling
        self.task_aspects = {
            "travel": ["destination", "budget", "duration", "accommodation", "transportation"],
            "shopping": ["product type", "price range", "features", "brands", "delivery options"],
            "technology": ["device specifications", "software requirements", "compatibility", "performance", "user interface"],
            "education": ["learning objectives", "resources", "schedule", "assessment methods", "prerequisites"],
            "finance": ["investment options", "risk tolerance", "time horizon", "financial goals", "tax implications"],
            "health": ["symptoms", "treatment options", "preventive measures", "specialists", "recovery timeline"],
            "career": ["job requirements", "application process", "interview preparation", "skill development", "networking"],
            "home": ["design elements", "materials", "budget constraints", "timeline", "contractor selection"]
        }
        
        # Recommendations for template filling
        self.recommendations = {
            "travel": [
                "creating a detailed itinerary that balances sightseeing with relaxation",
                "booking accommodations in central locations to minimize travel time",
                "using a mix of public transportation and walking to explore the destination",
                "allocating buffer days in your schedule for unexpected discoveries",
                "researching local customs and phrases before your trip"
            ],
            "shopping": [
                "comparing features across multiple brands before making a decision",
                "reading user reviews focusing on long-term reliability",
                "considering last year's model for better value",
                "checking return policies and warranty terms",
                "waiting for seasonal sales for significant discounts"
            ],
            "technology": [
                "prioritizing future-proof specifications over current needs",
                "ensuring compatibility with your existing devices and software",
                "allocating more budget to critical components that affect performance",
                "considering open-source alternatives to proprietary solutions",
                "implementing a phased approach to system upgrades"
            ],
            "education": [
                "creating a structured study plan with specific milestones",
                "using varied learning resources to reinforce concepts",
                "implementing spaced repetition techniques for better retention",
                "joining study groups or forums for collaborative learning",
                "scheduling regular self-assessments to identify knowledge gaps"
            ],
            "finance": [
                "diversifying your portfolio across different asset classes",
                "automating regular contributions to your investment accounts",
                "rebalancing your portfolio annually to maintain your target allocation",
                "maximizing tax-advantaged accounts before investing in taxable accounts",
                "maintaining an emergency fund before making higher-risk investments"
            ],
            "health": [
                "combining lifestyle modifications with medical treatments",
                "tracking relevant health metrics to monitor progress",
                "consulting specialists for comprehensive evaluation",
                "implementing gradual changes for sustainable results",
                "addressing root causes rather than just symptoms"
            ],
            "career": [
                "tailoring your resume and cover letter for each application",
                "developing a personal brand that highlights your unique value proposition",
                "networking strategically within your target industry",
                "pursuing relevant certifications to validate your skills",
                "preparing specific examples that demonstrate your capabilities"
            ],
            "home": [
                "focusing on high-impact improvements that add the most value",
                "getting multiple quotes from contractors for comparison",
                "creating a detailed project timeline with contingencies",
                "prioritizing structural integrity over aesthetic enhancements",
                "investing in quality materials for high-use areas"
            ]
        }
        
        # Advantages for template filling
        self.advantages = {
            "travel": [
                "maximizing your experience while minimizing stress",
                "ensuring you see the most important sights while still having time to relax",
                "immersing yourself in the local culture more effectively",
                "saving money on unnecessary expenses",
                "avoiding common tourist pitfalls"
            ],
            "shopping": [
                "ensuring you get the best value for your money",
                "avoiding buyer's remorse from hasty decisions",
                "finding the optimal balance between price and quality",
                "identifying products with the best longevity",
                "protecting yourself from potential issues down the line"
            ],
            "technology": [
                "reducing the need for frequent upgrades",
                "ensuring smooth integration with your workflow",
                "optimizing performance for your specific use cases",
                "minimizing compatibility issues",
                "creating a scalable solution that grows with your needs"
            ],
            "education": [
                "maintaining consistent progress toward your learning goals",
                "developing deeper understanding through multiple perspectives",
                "improving long-term retention of key concepts",
                "benefiting from collective knowledge and insights",
                "addressing weaknesses before they become problematic"
            ],
            "finance": [
                "reducing risk while maintaining growth potential",
                "building wealth consistently through dollar-cost averaging",
                "maintaining your target risk profile as markets change",
                "minimizing tax burden on your investments",
                "ensuring financial stability during unexpected events"
            ],
            "health": [
                "creating sustainable improvements rather than quick fixes",
                "objectively measuring your progress",
                "benefiting from specialized expertise",
                "building habits that last",
                "preventing recurrence of issues"
            ],
            "career": [
                "increasing your chances of getting interview invitations",
                "standing out in a competitive job market",
                "accessing opportunities through personal connections",
                "demonstrating your commitment to professional growth",
                "providing concrete evidence of your capabilities"
            ],
            "home": [
                "maximizing return on investment for your renovation budget",
                "ensuring fair pricing and quality workmanship",
                "managing expectations and reducing delays",
                "preventing costly repairs in the future",
                "ensuring durability in areas with high usage"
            ]
        }
        
        # Next steps for template filling
        self.next_steps = {
            "travel": [
                "finalize your itinerary, book accommodations, and arrange transportation",
                "research local attractions, create a packing list, and notify your bank of travel plans",
                "download offline maps, make copies of important documents, and learn basic local phrases",
                "check visa requirements, get necessary vaccinations, and purchase travel insurance",
                "book priority attractions in advance and create a flexible daily schedule"
            ],
            "shopping": [
                "create a comparison spreadsheet, read expert reviews, and check for upcoming sales",
                "visit stores to test products in person and ask about return policies",
                "check compatibility with your existing items and calculate total cost including accessories",
                "look for coupon codes, cashback opportunities, and loyalty program benefits",
                "verify warranty terms and availability of customer support"
            ],
            "technology": [
                "create a detailed requirements document and research compatible solutions",
                "test demo versions, read technical documentation, and consult user forums",
                "develop an implementation plan with clear phases and milestones",
                "allocate budget for training and support, not just acquisition",
                "create backup procedures and contingency plans before making changes"
            ],
            "education": [
                "create a structured study schedule and gather necessary learning materials",
                "set up a dedicated learning environment and eliminate potential distractions",
                "join relevant study groups and identify accountability partners",
                "schedule regular review sessions and practice assessments",
                "establish clear milestones and reward yourself for achieving them"
            ],
            "finance": [
                "open necessary accounts and set up automatic contributions",
                "review and adjust your budget to accommodate your financial goals",
                "create a system for tracking expenses and monitoring investments",
                "schedule annual portfolio reviews and tax planning sessions",
                "develop a comprehensive financial plan with short and long-term objectives"
            ],
            "health": [
                "schedule necessary appointments and create a tracking system for your health metrics",
                "modify your environment to support your health goals and reduce temptations",
     
(Content truncated due to size limit. Use line ranges to read in chunks)