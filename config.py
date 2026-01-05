import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4o"  # Or your specific fine-tuned model

SIMILARITY_THRESHOLD = 0.75  # For SBERT check 
MAX_LOOP_COUNT = 2           # For Introspective Loop 
MAX_DEBATE_ROUNDS = 3        # For Multi-Agent Debate 
