"""Constants for the Mistral AI integration."""

DOMAIN = "mistral_ai"

# Configuration keys
CONF_API_KEY = "api_key"
CONF_MAX_TOKENS = "max_tokens"
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_TEMPERATURE = "temperature"

# Default values
DEFAULT_MAX_TOKENS = 1000
DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TEMPERATURE = 0.7

# Subentry types, and the default title given to each new subentry
SUBENTRY_TYPE_AI_TASK_DATA = "ai_task_data"
SUBENTRY_TYPE_CONVERSATION = "conversation"

DEFAULT_AI_TASK_NAME = "Mistral AI task"
DEFAULT_CONVERSATION_NAME = "Mistral AI conversation"

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

# How long to wait on the API before giving up, in seconds
TIMEOUT = 30
