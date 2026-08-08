# Mistral AI for Home Assistant

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/release/donaghhorgan/ha-mistral-ai.svg)](https://github.com/donaghhorgan/ha-mistral-ai/releases)
[![License](https://img.shields.io/github/license/donaghhorgan/ha-mistral-ai.svg)](LICENSE)

A Home Assistant integration for Mistral AI, providing four platforms: a
conversation agent for voice assistants and chatbots, AI tasks for automations
and dashboards, and speech-to-text and text-to-speech for the assist pipeline.

## Features

- 🤖 **Any available model**: The model list is fetched from the API using your
  own key, so it reflects what you can actually use
- 🏠 **Device control**: Optionally expose a Home Assistant LLM API so the agent
  can control your devices via tool calling
- ⚡ **Streaming responses**: Replies are streamed back as they are generated
- 📋 **AI tasks**: Generate data for automations and dashboards, with native
  structured output, and generate images
- 🎤 **Speech-to-text**: Transcribe assist pipeline audio, with the model list
  filtered to those that report the capability
- 🔊 **Text-to-speech**: Speak replies using the voices on your account,
  including custom ones, streamed so playback starts while the reply is still
  being written
- 🎛️ **Multiple agents**: Run several agents and tasks off one API key, each
  with its own model, prompt and parameters
- 🌐 **Multilingual**: Supports multiple languages through Mistral AI's models,
  and the setup screens themselves are translated into fourteen more
- ⚙️ **Easy setup**: Configured entirely through the Home Assistant UI

## Requirements

- Home Assistant 2025.8.0 or newer
- A Mistral AI API key (get one at [console.mistral.ai](https://console.mistral.ai/))

## Installation

### HACS (Recommended)

1. Ensure that [HACS](https://hacs.xyz/) is installed
2. Go to HACS → Integrations
3. Click the three dots in the top right corner and select "Custom repositories"
4. Add this repository URL: `https://github.com/donaghhorgan/ha-mistral-ai`
5. Select "Integration" as the category
6. Click "Add"
7. Search for "Mistral AI" and install it
8. Restart Home Assistant

### Manual Installation

1. Download the latest release from the [releases page](https://github.com/donaghhorgan/ha-mistral-ai/releases)
2. Copy the whole `custom_components/mistral_ai` directory into the
   `custom_components` directory of your Home Assistant configuration, so that
   it ends up here:

   ```text
   config/
   └── custom_components/
       └── mistral_ai/
   ```

   Copy the directory rather than picking files out of it. This step used to
   list every module, and the list went stale twice as platforms were added —
   following it left out `stt.py` and `tts.py` and produced an integration
   that would not load.

3. Restart Home Assistant

## Configuration

### Initial Setup

1. Go to Settings → Devices & Services
2. Click "Add Integration"
3. Search for "Mistral AI"
4. Enter your Mistral AI API key
5. Click "Submit"

### Getting a Mistral AI API Key

1. Visit [console.mistral.ai](https://console.mistral.ai/)
2. Sign up for an account or log in
3. Go to the API section
4. Create a new API key
5. Copy the key and use it during the integration setup

Setting up the integration creates one conversation agent to start with. The
API key is held by the integration itself, and each agent or task is
configured separately underneath it, so you can run several with different
models and prompts against the same key.

### Adding and configuring agents

1. Go to Settings → Devices & Services → Mistral AI
2. Click "Add conversation agent" or "Add AI task"
3. Configure it, then click "Submit"

To change an existing one, click "Configure" next to it.

### Changing the API key

1. Go to Settings → Devices & Services → Mistral AI
2. Open the three-dot menu on the Mistral AI entry itself and choose
   "Reconfigure"
3. Enter the new key, then click "Submit"

The key is checked against the API before it is saved, and everything set up
underneath it is kept — deleting the entry to change the key would take every
agent, task and speech entity with it.

### Configuration options

- **Model**: Chosen from the models your API key can actually reach — the
  list is fetched from the API rather than hard-coded, so it stays correct as
  Mistral adds and retires models. You can also type a model name directly.

  If an agent you already set up is using a model that is being retired, Home
  Assistant raises a repair for it under Settings → System → Repairs, before
  the model stops working rather than after. Where Mistral names a successor,
  the repair offers to switch that agent over in one click; where it does not,
  the repair says so and leaves the choice to you.

  A model Mistral has scheduled for retirement is shown with its end date and
  its replacement, and sorted below the ones that are staying. It still works
  until that date, so it stays selectable rather than disappearing — but you
  will not pick one by accident. A model that has already been withdrawn also
  stays in the list while it is the one configured, so reconfiguring an entity
  that needs moving does not lose track of where it was.

- **Temperature**: Controls randomness in responses
  - Lower values (0.1-0.3): More focused and deterministic
  - Higher values (0.7-1.0): More creative and varied
  - The slider stops where the API does, which is not the same everywhere:
    1.0 for conversation agents, 1.5 for AI tasks and speech-to-text. A
    conversation agent gets the lower limit because web search moves its
    requests to an endpoint that caps at 1.0, and that is a setting on the
    same page.

- **Maximum tokens**: Maximum length of responses. If a reply runs out of room
  before the model has said anything — which happens with reasoning models,
  since thinking is spent from the same budget — the request reports that
  rather than returning nothing. A reply cut off part way through is kept, so
  you see what arrived. This applies with web search on as well as off,
  although the two are detected differently — that endpoint reports no finish
  reason, so a response that spends the entire budget is what gives it away.

  The default is 3000,
  matching Home Assistant's own OpenAI and Google integrations.

- **Top-p** (conversation agents and AI tasks): An alternative to
  temperature. The model considers only the most likely words whose
  probabilities add up to this value, so lowering it narrows what it will
  choose from. 1.0 leaves sampling alone, which is the default.

  Lower this or the temperature, not both — they are two ways of doing the
  same thing and reducing both at once tends to make replies repetitive. It
  is not offered for speech-to-text or text-to-speech, because neither
  endpoint takes it.

- **Reasoning** (conversation agents and AI tasks): Lets the model think
  before it answers. Off leaves it answering directly; on gives it room to
  work a problem through first, which costs tokens and time but helps on
  anything multi-step.

  Read this together with the maximum tokens setting above, because the two
  compete. Thinking is spent from the same budget as the answer, so a tight
  limit can be used up before the answer begins — that is the case the
  maximum tokens note describes, and turning reasoning on is what makes it
  easy to hit. If replies start coming back empty, or AI tasks start failing
  with no structured data, raise the maximum tokens before changing anything
  else.

  The field only appears for models that support it, because a model that
  does not rejects the setting outright rather than ignoring it — so an
  agent configured with it would fail on every request. If you switch a
  subentry to a model that cannot reason, the setting is dropped when you
  save. Leave it unset to use whatever the model does by default, which is
  what every existing agent does today.

- **Instructions** (conversation agents only): Custom instructions for the
  assistant. You can use Home Assistant template variables like
  `{{ ha_name }}`.

- **Control Home Assistant** (conversation agents only): Selects a Home
  Assistant LLM API, which is what lets the agent control your devices. New
  conversation agents get the Assist API by default, matching Home
  Assistant's own OpenAI, Anthropic and Google integrations.

  Clearing it gives an agent that only answers questions. Note that it then
  stops advertising the `CONTROL` feature, so Home Assistant will not offer it
  anywhere control is required — the agent is not broken, it is scoped.

  A model that keeps asking to use tools is stopped after ten rounds, and says
  so. That limit is what Home Assistant's own LLM integrations use. Reaching it
  usually means the request needs breaking into smaller steps; the message
  names the cause rather than leaving you with an empty reply.

### Web search (beta)

Conversation agents answer from training data, so anything time-sensitive comes
back stale or invented. Turning on "Web search" lets the agent look things up.
It is off unless a tier is chosen, because Mistral bills per search and the
premium tier costs more — the expensive one is never picked for you.

It works alongside device control: Mistral runs the search itself and hands
back any Home Assistant tool calls for the integration to run, so one turn can
both check the forecast and turn on a light.

Marked beta for one reason, worth knowing before turning it on:

- Web search is a built-in connector, and connectors only run on Mistral's
  conversations API, which is in public preview. Its shapes can change.

As with image generation, the request sets `store: false`, so Mistral does not
retain the conversation.

### Speech-to-text

Add a speech-to-text entity with "Add speech-to-text" and select it as the
speech-to-text engine of an assist pipeline under
Settings → Voice assistants.

The model dropdown lists only models that report the `audio_transcription`
capability, which is asked of the API rather than hard-coded, so it stays
correct as Mistral ships and retires models.

Two things worth knowing:

- Home Assistant streams raw 16-bit 16 kHz mono audio. The integration wraps
  it in a WAV container before sending, because the transcription endpoint
  infers the format from the file it is given.
- The pipeline's language is passed to the API as a hint only — these models
  detect the language themselves.

### Text-to-speech

Add a text-to-speech entity with "Add text-to-speech" and select it as the
text-to-speech engine of an assist pipeline, or call `tts.speak` with its
entity ID.

Speech is streamed, in both directions. Audio starts playing as Mistral
produces it rather than once the whole file has arrived, and the reply is
spoken sentence by sentence as the conversation agent writes it — so the
first words are heard without waiting for the last ones to be composed.

Nothing to configure, and it applies to `tts.speak` as well as to a pipeline.
A short phrase still arrives in one piece; the difference shows on long
replies, which is where the wait was worth removing.

The model dropdown lists only models reporting the `audio_speech` capability.
The voice dropdown is populated from your account, because custom voices are
created against it, and a voice is required — the API will not choose one,
and refuses a request without it. Callers can override the configured voice
per request with the `voice` option.

If the voices cannot be listed, adding the entity is refused rather than
offered without the field. Preset voices exist on every account, so an empty
list means the request failed, and an entity saved in that state could never
speak.

The language is not sent to the API: the voice carries its own language.
Selecting a French voice and an English pipeline gets you a French voice
reading English, which is a voice problem rather than a language one.

### AI tasks

As well as conversation agents, this integration provides
[AI Task](https://www.home-assistant.io/integrations/ai_task/) entities, which
generate data for automations and dashboards rather than holding a
conversation. Add one with "Add AI task", then call
`ai_task.generate_data` with its entity ID. Passing a structure returns
validated JSON, using Mistral's native structured output.

Conversation agents and AI tasks both accept attachments: images, and PDFs.
A PDF is sent as a document rather than a picture, and the model answers from
its contents — asked what a bill was for, it reads the amount and the date off
the page. No model capability gates this; extraction happens on Mistral's side
and any chat model can do it.

Attachments are inlined into the request, so they are capped at 20 MB. That is
not the API's limit — it accepts considerably more — it is to avoid reading an
unbounded file into memory on a machine that may not have it to spare.

The same entity also generates images via `ai_task.generate_image`, on the
model already configured on the subentry — there is no second model to pick.
Home Assistant saves the result to the media source and returns a reference
to it.

Image generation is the one thing here that does not go through the chat
completions endpoint. It is a built-in connector, and connectors only run on
Mistral's conversations API, so those requests go there instead. Two
consequences worth knowing:

- The request sets `store: false`, so the conversation is not retained by
  Mistral. That endpoint would otherwise keep it and list it afterwards,
  where chat completions stores nothing.
- `store: false` covers the conversation and not the image. The connector
  writes the image as a file on your account and hands back a reference to
  it, so the integration downloads the bytes and then deletes the remote
  copy. Home Assistant has its own copy in the media source by that point.

  Earlier versions did not delete it, so an account used for image generation
  before this may have images still sitting on it. They are listed under
  Files in the Mistral console.
- The conversations API is in public preview, so image generation rests on a
  less stable footing than the rest of the integration.

## Usage

### Voice Assistants

Once configured, you can use the Mistral AI conversation agent with any
Home Assistant voice assistant:

1. Go to Settings → Voice assistants
2. Select your voice assistant (Assist, Alexa, Google Assistant, etc.)
3. Set the conversation agent to your Mistral AI agent

### Automation and Scripts

You can also use the conversation agent in automations and scripts:

```yaml
action: conversation.process
data:
  agent_id: conversation.mistral_ai_conversation
  text: "What's the weather like today?"
```

And to generate data with an AI task:

```yaml
action: ai_task.generate_data
data:
  entity_id: ai_task.mistral_ai_task
  task_name: summarise
  instructions: "Summarise today's weather in one sentence."
```

### Custom Prompts

You can customize the system prompt to make the AI assistant more helpful
for your specific use case:

**Example - Smart Home Assistant:**

```text
You are a helpful smart home assistant for {{ ha_name }}.
You help users control their smart home devices and answer questions about home automation.
Be concise and focus on actionable responses.
Current location: {{ ha_name }}
```

**Example - Family Assistant:**

```text
You are a friendly family assistant named Alex for the {{ ha_name }} household.
You help with daily tasks, reminders, and general questions.
Always be warm and family-friendly in your responses.
```

## API Usage and Costs

This integration uses the Mistral AI API, which is a paid service. Costs depend on:

- **Model used**: Larger models cost more per token
- **Usage volume**: You pay per token (input + output)
- **Token limits**: Set an appropriate maximum to control costs

### Cost Optimization Tips

1. **Choose the right model**: Use smaller models for simpler tasks
2. **Set reasonable token limits**: Don't set the maximum higher than needed
3. **Use concise prompts**: Shorter system prompts reduce input costs
4. **Monitor usage**: Check your Mistral AI console for usage statistics

## Troubleshooting

### Common Issues

**Integration won't load:**

- Check that you have the correct API key
- Verify your internet connection
- Check Home Assistant logs for error details

**API errors:**

- Verify your Mistral AI account has sufficient credits
- Check if you've exceeded rate limits
- Ensure your API key has the necessary permissions

**Slow responses:**

- Try a smaller/faster model like `mistral-small-latest`
- Reduce the maximum tokens setting
- Check your internet connection speed

### Debug Logging

To enable debug logging for troubleshooting:

```yaml
logger:
  default: info
  logs:
    custom_components.mistral_ai: debug
```

## Development

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

#### Translations

The setup screens are available in Czech, Danish, Dutch, French, German,
Italian, Japanese, Norwegian Bokmål, Polish, Portuguese, Russian, Simplified
Chinese, Spanish and Swedish. Only the English is written by a speaker of the
language; the rest are machine-translated and have not been reviewed, so
expect the occasional wrong word or stiff phrasing.

Corrections are welcome and need no ceremony — open an issue quoting the text
you saw, or edit the file under `custom_components/mistral_ai/translations/`
directly. English is the source: every other file mirrors its keys exactly,
and `scripts/check_translation_consistency.py` will tell you if an edit breaks
that.

### Local Development

```bash
# Install dependencies (dev tools plus the current Home Assistant version)
uv sync

# Run the linters, type checker and consistency checks
uv run pre-commit run --all-files

# Run the tests
uv run pytest

# Run the tests against the oldest supported Home Assistant version.
# Resolved on the fly rather than locked -- see CLAUDE.md for why, and
# .github/workflows/ci.yml for the pins this mirrors.
uv run --isolated --no-project \
  --with pytest-homeassistant-custom-component==0.13.269 \
  --with hassil==2.2.3 --with home-assistant-intents==2025.7.30 \
  --with pycares==4.9.0 --with ha-ffmpeg --with mutagen \
  --with pymicro-vad --with pyspeex-noise \
  --with "mistralai>=2.9.0" --with PyTurboJPEG \
  pytest tests/ -q --no-cov
```

See [`AGENTS.md`](AGENTS.md) for conventions and
[`scripts/README.md`](scripts/README.md) for the consistency checks.

## Support

- **Issues**: [GitHub Issues](https://github.com/donaghhorgan/ha-mistral-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/donaghhorgan/ha-mistral-ai/discussions)
- **Home Assistant Community**: [Community Forum](https://community.home-assistant.io/)

## License

This project is licensed under the GNU General Public License v3.0 - see the
[LICENSE](LICENSE) file for details.

## Disclaimer

This integration is not officially associated with Mistral AI. It's a
community-developed integration that uses the Mistral AI API.

## Acknowledgments

- [Mistral AI](https://mistral.ai/) for providing the language models
- [Home Assistant](https://www.home-assistant.io/) for the awesome home
  automation platform
- The Home Assistant community for inspiration and support
