"""Constants for the Mistral AI integration."""

DOMAIN = "mistral_ai"

# Configuration keys
CONF_API_KEY = "api_key"
CONF_MAX_TOKENS = "max_tokens"
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_TEMPERATURE = "temperature"
CONF_VOICE = "voice"
CONF_WEB_SEARCH = "web_search"

# Default values
#
# DEFAULT_MAX_TOKENS matches openai_conversation and
# google_generative_ai_conversation, which both use 3000. It is a ceiling
# rather than a target, and the tool calling loop spends it a request at a
# time, so a lower figure buys nothing and truncates mid-sentence.
DEFAULT_MAX_TOKENS = 3000
DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TEMPERATURE = 0.7

# Transcription is asked to report what was said, not to be interesting about
# it, so it gets its own default rather than the conversational 0.7. The
# parameter is a licence to guess at unclear audio, and guessing is the failure
# mode that makes a voice assistant act on something nobody said.
DEFAULT_STT_TEMPERATURE = 0.0

# Subentry types, and the default title given to each new subentry
SUBENTRY_TYPE_AI_TASK_DATA = "ai_task_data"
SUBENTRY_TYPE_CONVERSATION = "conversation"
SUBENTRY_TYPE_STT = "stt"
SUBENTRY_TYPE_TTS = "tts"

# These become the subentry title, which becomes the device name, which --
# because the entities set _attr_name to None -- becomes the entity name and
# the entity id. Matching how Home Assistant's own LLM integrations name
# theirs: "<Brand> Conversation", "<Brand> AI Task", and STT/TTS abbreviated
# as google_generative_ai_conversation does.
DEFAULT_AI_TASK_NAME = "Mistral AI Task"
DEFAULT_CONVERSATION_NAME = "Mistral AI Conversation"
DEFAULT_STT_NAME = "Mistral AI STT"
DEFAULT_TTS_NAME = "Mistral AI TTS"

# Model capability flags, as reported by the models endpoint. The config flow
# filters the model list by these rather than hard-coding model names, which
# go stale every time Mistral ships or retires one.
CAPABILITY_AUDIO_TRANSCRIPTION = "audio_transcription"
CAPABILITY_COMPLETION_CHAT = "completion_chat"
CAPABILITY_AUDIO_SPEECH = "audio_speech"

# There is no image_generation capability. The endpoint reports audio, ocr,
# vision, function_calling and a dozen others, and none of them says whether a
# model can produce an image.
#
# Image generation is a built-in connector, passed as a tool, so calling tools
# is a precondition for it: a model without this certainly cannot generate an
# image. It is not proof that it can, which is why the AI task entity still
# handles an empty result rather than trusting the flag.
CAPABILITY_FUNCTION_CALLING = "function_calling"

# Asked of the speech endpoint, and handed to Home Assistant as the file
# extension. mp3 because it is what the media player pipeline handles with the
# least ceremony; the API also offers pcm, wav, flac and opus.
TTS_AUDIO_FORMAT = "mp3"

# Offered to the assist pipeline as the languages the speech platforms accept.
# Home Assistant matches a pipeline's language against this list before it will
# use an entity, so a missing entry means the entity is simply never offered.
#
# Deliberately broad rather than exhaustive. The transcription models detect
# the language themselves and the code is passed only as a hint, so listing one
# that is handled poorly costs a bad result, while omitting one that is handled
# well means nobody can select it at all.
SPEECH_LANGUAGES = (
    "ar",
    "bn",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "he",
    "hi",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sv",
    "ta",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh",
)

# Image generation runs as a connector on the ordinary chat models rather than
# needing a separate endpoint or a model option -- unlike openai_conversation,
# which carries a RECOMMENDED_IMAGE_MODEL because its image generation is a
# different endpoint entirely.
IMAGE_GENERATION_TOOL = "image_generation"

# The built-in connectors Mistral runs for us. Sent as tools and executed on
# their side, so nothing local answers to these names.
#
# The two tiers bill differently and Mistral charges per search, so which one
# is used is named rather than toggled -- the expensive one should never be
# picked on someone's behalf.
WEB_SEARCH_TOOLS = ("web_search", "web_search_premium")

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

# How long to wait on the API before giving up, in seconds
TIMEOUT = 30
