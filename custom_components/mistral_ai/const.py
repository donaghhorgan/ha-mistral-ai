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

DEFAULT_AI_TASK_NAME = "Mistral AI task"
DEFAULT_CONVERSATION_NAME = "Mistral AI conversation"
DEFAULT_STT_NAME = "Mistral AI speech-to-text"
DEFAULT_TTS_NAME = "Mistral AI text-to-speech"

# Model capability flags, as reported by the models endpoint. The config flow
# filters the model list by these rather than hard-coding model names, which
# go stale every time Mistral ships or retires one.
CAPABILITY_AUDIO_TRANSCRIPTION = "audio_transcription"
CAPABILITY_AUDIO_SPEECH = "audio_speech"

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

# The built-in connectors the API can run for us. Sent as a tool alongside the
# Home Assistant ones, but executed by Mistral rather than by us -- there is no
# local handler for these names, which is why the stream filters them out.
#
# The premium tier bills differently, so which one is used is a choice the user
# makes rather than something decided here.
WEB_SEARCH_TOOLS = ("web_search", "web_search_premium")

# Image generation is a built-in connector too, so it runs on the ordinary
# chat models rather than needing a separate endpoint or a model option --
# unlike openai_conversation, which carries a RECOMMENDED_IMAGE_MODEL because
# its image generation is a different endpoint entirely.
IMAGE_GENERATION_TOOL = "image_generation"

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

# How long to wait on the API before giving up, in seconds
TIMEOUT = 30
