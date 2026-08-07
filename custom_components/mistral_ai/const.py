"""Constants for the Mistral AI integration."""

DOMAIN = "mistral_ai"

# Configuration keys
CONF_API_KEY = "api_key"
CONF_MAX_TOKENS = "max_tokens"
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
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

# 1.0 leaves sampling alone, which is what every core LLM integration defaults
# to for this. openai_conversation uses 1.0 and
# google_generative_ai_conversation 0.95, and neither is a tuned figure -- it
# is "do not restrict the distribution unless asked".
DEFAULT_TOP_P = 1.0

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

# The highest temperature each subentry type can actually send. Asked of the
# API rather than taken from the OpenAPI schema, because the schema is wrong
# about one of them and silent about another:
#
#   /v1/chat/completions          <= 1.5   schema agrees
#   conversations completion_args <= 1.0   schema agrees
#   /v1/audio/transcriptions      <= 1.5   schema declares no bound at all
#
# Anything above returns 422, and the slider used to go to 2.0, so the top
# quarter of it broke every request that used it.
#
# Conversation agents get the lowest of the three. Web search moves them to
# the conversations endpoint, and it is a checkbox on the same form -- a
# temperature that works until an unrelated setting is switched on is worse
# than one that is merely lower than it could be.
# top_p is bounded at 1.0 by both endpoints, unlike temperature, whose limit
# differs between them. Checked with real requests rather than read off the
# schema, since the schema was wrong about one temperature bound and silent
# about another.
MAX_TOP_P = 1.0

MAX_TEMPERATURE = {
    SUBENTRY_TYPE_AI_TASK_DATA: 1.5,
    SUBENTRY_TYPE_CONVERSATION: 1.0,
    SUBENTRY_TYPE_STT: 1.5,
    SUBENTRY_TYPE_TTS: 1.5,
}

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

# Whether a model accepts reasoning_effort at all.
#
# Not a filter on the dropdown -- a model that cannot reason is a perfectly
# good conversation agent -- but a gate on whether the field is offered, and on
# whether a stored value is sent. A model without this rejects *every* value,
# including "none":
#
#   400 reasoning_effort is not enabled for this model
#
# so sending it unconditionally would break every non-reasoning model. The flag
# predicted acceptance exactly across a sample of twelve chat models, and it
# has to be read per model id: mistral-medium-latest reasons and the pinned
# mistral-medium-2505 does not, so a name heuristic gets the same family wrong.
CAPABILITY_REASONING = "reasoning"

# The values the API actually accepts, which is not what it says it accepts.
#
# Two validation layers disagree. The schema layer rejects an unknown string
# with 422 and names seven values -- none, minimal, low, medium, high, xhigh,
# max -- one more than the published OpenAPI enum, so the spec is behind. The
# model layer then rejects five of those seven:
#
#   400 reasoning_effort='minimal' is not supported for this model.
#       Must be one of (<ReasoningEffort.none: 'none'>,
#                       <ReasoningEffort.high: 'high'>)
#
# Building this list from either the spec or the 422 message would offer five
# options that fail. Measured against both /v1/chat/completions and
# completion_args, which agree -- so unlike temperature there is no
# per-endpoint ceiling to pick between.
REASONING_EFFORTS = ("none", "high")

# Asked of the speech endpoint, and handed to Home Assistant as the file
# extension. mp3 because it is what the media player pipeline handles with the
# least ceremony; the API also offers pcm, wav, flac and opus.
TTS_AUDIO_FORMAT = "mp3"

# What an attachment may be, and how big.
#
# Documents are not gated on a model capability. The obvious candidate is the
# `vision` flag, and it is the wrong one -- a chat model without it read a PDF
# back correctly, so extraction happens server side and any chat model can do
# it. Checked rather than assumed, because gating on `vision` would have hidden
# the feature from most of the model list for no reason.
ATTACHMENT_DOCUMENT_TYPE = "application/pdf"

# Not the API's limit. It accepted a 30 MB PDF without complaint, so this is
# about the Home Assistant process rather than the endpoint: an attachment is
# read whole and base64 encoded, so the bytes and a string a third larger are
# both resident at once, and Home Assistant frequently runs on a machine where
# that matters. Far above any real photo or scanned bill.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

# Voices are paged, and the page defaults to 10 -- both in the API and in the
# SDK signature. Listing without asking for a page size showed the first ten
# of an account's voices and silently dropped the rest, which on the account
# this was found on meant showing 10 of 31.
#
# 100 is the documented maximum. The listing still pages, because an account
# can have more than that.
VOICE_PAGE_SIZE = 100

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
