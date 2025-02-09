# 📚 Project Title

This project implements a Retrieval-Augmented Generation (RAG) pipeline with a Proximal Policy Optimization (PPO) agent for question-answering using Gradio for the interface.

## Table of Contents

- [Overview](#overview)
- [Proximal Policy Optimization (PPO) Agent](#proximal-policy-optimization-ppo-agent)
- [Feedback Mechanism](#feedback-mechanism)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project is designed to answer user questions by retrieving relevant information from various sources such as web URLs and PDF files. It integrates a PPO agent to learn from user feedback and improve the quality of responses over time.

## 🤖 Proximal Policy Optimization (PPO) Agent

The PPO agent is responsible for learning from user feedback and updating the state-value function. Here is a brief overview of how the PPO agent works in this project:

- **Initialization**: The PPO agent is initialized with a learning rate (`alpha`), a clip parameter (`epsilon`), and a decay rate for the learning rate.

    ```python
    class PPOAgent:
        def __init__(self):
            self.alpha = ALPHA  # Initial learning rate
            self.epsilon = EPSILON  # Clip parameter
            self.values = {}
            self.decay_rate = 0.99  # Learning rate decay

        def update(self, state, reward):
            if state not in self.values:
                self.values[state] = 0

            old_value = self.values[state]
            new_value = old_value + self.alpha * (reward - old_value)

            # Apply the clipping
            clipped_value = np.clip(new_value, old_value - self.epsilon, old_value + self.epsilon)

            self.values[state] = clipped_value
            self.alpha *= self.decay_rate  # Decay learning rate

        def get_value(self, state):
            return self.values.get(state, 0)
    ```

- **Updating State-Value Function**: The `update` method updates the state-value function based on the reward received. The new value is clipped to ensure stable updates.

- **Learning Rate Decay**: The learning rate decays over time to ensure more precise adjustments during later stages of training.

## 📝 Feedback Mechanism

The project incorporates a feedback mechanism to collect user feedback and update the PPO agent's state-value function. Here's how the feedback mechanism works:

- **Recording Feedback**: User feedback is recorded and stored in a repository. The feedback includes the rating and the response.

    ```python
    def feedback(question: str, rating: int, response: str):
        feedback_repository[question] = {'rating': rating, 'response': response}
        reward = rating / 5  # Normalize rating to 0-1
        agent.update(question, reward)
        return {"message": f"Feedback recorded: {feedback_repository[question].get('rating')} and normalized value {reward}"}
    ```

- **Normalizing Rating**: The rating is normalized to a value between 0 and 1 by dividing by 5. This normalized rating is used as the reward for updating the PPO agent.

- **Updating PPO Agent**: The PPO agent's `update` method is called to update the state-value function with the normalized reward.

## ✨ Features

- **🤖 PPO Agent**: Learns from user feedback and updates the state-value function.
- **📄 Document Retrieval**: Loads and retrieves documents from web URLs and PDFs.
- **❓ Question Answering**: Processes source documents and answers user questions.
- **🌐 Gradio Interface**: Provides an interactive web interface for users to input URLs, upload PDFs, ask questions, and provide feedback.

