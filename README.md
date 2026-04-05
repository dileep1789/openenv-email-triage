# OpenEnv Email Triage AI Environment

This project implements a competition-level AI training environment for email triage, classification, and response generation, following the OpenEnv specification.

## 🚀 Overview

The environment simulates a real-world email management system where an AI agent must:
1. **Observe**: Read incoming emails (subject, body, sender, history).
2. **Classify**: Categorize emails (Support, Spam, Sales, Complaint, Job Application).
3. **Act**: Decide on the next step (Reply, Escalate, Ignore).
4. **Respond**: Generate a professional and accurate response if needed.

## 📦 Project Structure

```text
openenv-email-triage/
├── env/
│   ├── environment.py  # Core OpenEnv class
│   ├── models.py       # Pydantic data models
│   ├── tasks.py        # Task data (Easy, Medium, Hard)
│   └── graders.py      # Reward calculation logic
├── openenv.yaml        # Environment metadata
├── inference.py        # Standard agent execution loop
├── Dockerfile          # Containerization
├── requirements.txt    # dependencies
└── README.md           # Documentation
```

## 🎯 Reward System

Rewards are granted based on three criteria:
- **Classification Accuracy (40%)**: Correctly identifying the email category.
- **Action Selection (30%)**: Choosing the correct action (Reply vs. Escalate).
- **Response Content (30%)**: Including relevant keywords in the reply.

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- OpenAI API Key (or compatible provider)

### Local Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your-key"
   export API_BASE_URL="api-base-url"
   export MODEL_NAME="model-name"
   ```
3. Run inference:
   ```bash
   python inference.py
   ```

### Docker
Build and run with Docker:
```bash
docker build -t email-triage .
docker run -e OPENAI_API_KEY="your-key" email-triage
```

## 🏆 Competition Tips

- **High Fidelity**: The environment uses structured Pydantic models for type safety.
- **Task Scaling**: Tasks range from simple classification to complex "hard" workflows.
- **Deterministic Grading**: High-scoring submissions use clear, objective scoring metrics.
