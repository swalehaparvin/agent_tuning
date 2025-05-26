"""
Simplified Gradio Demo for Agent Tuning Optimization Framework

This script creates a simple Gradio web interface to demonstrate the framework's capabilities.
"""

import os
import gradio as gr
import numpy as np
import random
from datetime import datetime

# Mock functions to simulate framework behavior without requiring full model loading
def mock_generate_response(task, user_message):
    """Simulate generating a response from a tuned agent."""
    responses = [
        f"I'll help you with your task to {task.lower()}. Based on your message '{user_message}', I recommend starting by breaking this down into smaller steps.",
        f"I understand you need assistance with {task.lower()}. From your message, I can see that you're looking for guidance on '{user_message}'. Here's my approach to solving this.",
        f"Thank you for providing details about {task.lower()}. Your message '{user_message}' gives me enough context to help you effectively. Let me outline a solution.",
        f"I'm analyzing your request about {task.lower()}. Your message '{user_message}' indicates you need comprehensive assistance. Here's what I suggest as next steps."
    ]
    
    # Simulate processing time
    import time
    time.sleep(1.5)
    
    return random.choice(responses) + f"\n\nResponse generated at {datetime.now().strftime('%H:%M:%S')}"

def mock_generate_negative_sample(task, user_message, agent_message):
    """Simulate generating a negative sample from a positive example."""
    degradation_types = [
        "Response truncation",
        "Grammatical errors",
        "Task misalignment",
        "Constraint violation",
        "Irrelevant tangent"
    ]
    
    degradation = random.choice(degradation_types)
    
    if degradation == "Response truncation":
        words = agent_message.split()
        truncate_point = int(len(words) * random.uniform(0.3, 0.7))
        return " ".join(words[:truncate_point]) + f"...\n\nNegative sample type: {degradation}"
    
    elif degradation == "Grammatical errors":
        errors = [
            lambda t: t.replace(".", ""),  # Remove periods
            lambda t: t.replace("I ", "i "),  # Lowercase I
            lambda t: t.replace(" the ", " teh "),  # Typo
            lambda t: t.replace(" is ", " are "),  # Grammar error
            lambda t: t.replace(" are ", " is ")  # Grammar error
        ]
        
        result = agent_message
        for _ in range(random.randint(2, 4)):
            error_func = random.choice(errors)
            result = error_func(result)
        
        return result + f"\n\nNegative sample type: {degradation}"
    
    elif degradation == "Task misalignment":
        misalignments = [
            f"I understand you're asking about something completely different. Let me tell you about weather patterns instead.",
            f"I don't think that's what you really want to know. Let me explain something else that might interest you.",
            f"Your question seems to be about {task}, but I'd rather discuss the history of computing.",
            f"Instead of addressing your specific request about {task}, let me give you general information that's only tangentially related."
        ]
        
        return random.choice(misalignments) + f"\n\nNegative sample type: {degradation}"
    
    elif degradation == "Constraint violation":
        violations = [
            f"I specifically recommend the XYZ Pro 2000 for $499.99, the UltraBook 15 for $1,299, and the PowerTech 5000 for $799. These are the absolute best options available.",
            f"The system utilizes a polymorphic encapsulation paradigm with recursive lambda functions and stochastic gradient descent with backpropagation through a multi-layer perceptron.",
            f"What specific features are you looking for? Do you have any brand preferences? What's your budget range? When do you need this by? Have you considered alternative options?",
            f"Since you're a tech-savvy individual who values cutting-edge features, you'll definitely want the latest model with all the advanced capabilities."
        ]
        
        return random.choice(violations) + f"\n\nNegative sample type: {degradation}"
    
    else:  # Irrelevant tangent
        tangents = [
            f"Did you know that artificial intelligence has been a concept since the 1950s? The field has evolved significantly since then, with major breakthroughs in neural networks and deep learning.",
            f"I've been thinking about the philosophical implications of consciousness in AI systems. The question of whether an AI can truly understand or merely simulate understanding is fascinating.",
            f"The weather has been quite interesting lately, with unusual patterns emerging globally. Climate scientists attribute this to a combination of factors including ocean temperature changes.",
            f"I recently processed some fascinating data about renewable energy technologies. Solar efficiency has improved dramatically in the past decade, while costs have decreased by over 80%."
        ]
        
        return random.choice(tangents) + f"\n\nNegative sample type: {degradation}"

def mock_generate_synthetic_trajectory(task):
    """Simulate generating a synthetic trajectory for a given task."""
    # Determine task category
    categories = ["travel", "shopping", "technology", "education", "finance", "health", "career", "home"]
    category = random.choice(categories)
    
    # Generate interactions (2-4 turns)
    num_turns = random.randint(2, 4)
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
    
    # Format trajectory
    result = f"Synthetic Trajectory for Task: {task}\nCategory: {category}\n\n"
    
    for i, interaction in enumerate(interactions):
        result += f"Turn {i+1}:\nUser: {interaction['user']}\nAgent: {interaction['agent']}\n\n"
    
    result += f"Generation method: Template-based\nQuality score: {random.uniform(0.7, 0.9):.2f}"
    
    return result

# Create Gradio interface
with gr.Blocks(title="Agent Tuning Framework Demo") as demo:
    gr.Markdown("# Agent Tuning Optimization Framework Demo")
    gr.Markdown("### A framework for efficiently tuning LLMs into specialized agents using negative and synthetic samples")
    
    with gr.Tab("Generate Response"):
        with gr.Row():
            with gr.Column():
                task_input = gr.Textbox(label="Task Description", placeholder="e.g., Book a flight from New York to London")
                user_input = gr.Textbox(label="User Message", placeholder="e.g., I need to travel next week for business")
                generate_btn = gr.Button("Generate Response", variant="primary")
            with gr.Column():
                response_output = gr.Textbox(label="Agent Response", lines=8)
        
        generate_btn.click(
            mock_generate_response,
            inputs=[task_input, user_input],
            outputs=response_output
        )
        
        gr.Examples(
            [
                ["Book a flight from New York to London", "I need to travel next week for business"],
                ["Find a vegetarian restaurant", "I'm looking for dinner options tonight"],
                ["Help me debug a Python script", "I'm getting an IndexError in my code"]
            ],
            inputs=[task_input, user_input]
        )
    
    with gr.Tab("Generate Negative Sample"):
        with gr.Row():
            with gr.Column():
                neg_task_input = gr.Textbox(label="Task Description", placeholder="e.g., Book a flight from New York to London")
                neg_user_input = gr.Textbox(label="User Message", placeholder="e.g., I need to travel next week for business")
                neg_agent_input = gr.Textbox(label="Agent Message (Positive Example)", placeholder="e.g., I'd be happy to help you book a flight...", lines=5)
                neg_generate_btn = gr.Button("Generate Negative Sample", variant="primary")
            with gr.Column():
                neg_output = gr.Textbox(label="Negative Sample", lines=8)
        
        neg_generate_btn.click(
            mock_generate_negative_sample,
            inputs=[neg_task_input, neg_user_input, neg_agent_input],
            outputs=neg_output
        )
        
        gr.Examples(
            [
                ["Book a flight from New York to London", "I need to travel next week for business", "I'd be happy to help you book a flight from New York to London. Could you provide more details about your preferred travel dates, budget, and any airline preferences you might have?"],
                ["Recommend a laptop for programming", "I need a new laptop for software development", "I can definitely help you find a suitable laptop for programming. Based on software development needs, I'd recommend looking for a laptop with at least 16GB RAM, a multi-core processor, and an SSD for storage. Would you like specific brand recommendations or have a particular budget in mind?"]
            ],
            inputs=[neg_task_input, neg_user_input, neg_agent_input]
        )
    
    with gr.Tab("Generate Synthetic Trajectory"):
        with gr.Row():
            with gr.Column():
                synth_task_input = gr.Textbox(label="Task Description", placeholder="e.g., Plan a weekend trip to Chicago")
                synth_generate_btn = gr.Button("Generate Synthetic Trajectory", variant="primary")
            with gr.Column():
                synth_output = gr.Textbox(label="Synthetic Trajectory", lines=15)
        
        synth_generate_btn.click(
            mock_generate_synthetic_trajectory,
            inputs=[synth_task_input],
            outputs=synth_output
        )
        
        gr.Examples(
            [
                ["Plan a weekend trip to Chicago"],
                ["Recommend healthy meal prep options for the week"],
                ["Help me create a study schedule for final exams"]
            ],
            inputs=[synth_task_input]
        )
    
    gr.Markdown("""
    ## About This Framework
    
    The Agent Tuning Optimization Framework provides a comprehensive solution for efficiently tuning large language models into specialized agents through the strategic incorporation of negative samples and synthetic trajectories.
    
    ### Key Features:
    
    1. **Negative Sample Generation**: Creates examples of undesired agent behaviors to teach models what not to do
    2. **Synthetic Trajectory Generation**: Automatically generates diverse interaction trajectories
    3. **Mixed-Sample Tuning**: Combines positive examples, negative samples, and synthetic trajectories
    4. **Parameter-Efficient Fine-Tuning**: Implements methods like LoRA for computational efficiency
    
    This demo provides a simplified simulation of the framework's capabilities. For full functionality, deploy the complete framework following the provided documentation.
    """)

# Launch the interface
demo.launch(share=True)
