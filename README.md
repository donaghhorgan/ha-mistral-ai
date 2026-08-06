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
  including custom ones
- 🎛️ **Multiple agents**: Run several agents and tasks off one API key, each
  with its own model, prompt and parameters
- 🌐 **Multilingual**: Supports multiple languages through Mistral AI's models
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

### Configuration options

- **Model**: Chosen from the models your API key can actually reach — the
  list is fetched from the API rather than hard-coded, so it stays correct as
  Mistral adds and retires models. You can also type a model name directly.

  A model Mistral has scheduled for retirement is shown with its end date and
  its replacement, and sorted below the ones that are staying. It still works
  until that date, so it stays selectable rather than disappearing — but you
  will not pick one by accident. A model that has already been withdrawn also
  stays in the list while it is the one configured, so reconfiguring an entity
  that needs moving does not lose track of where it was.

- **Temperature**: Controls randomness in responses (0.0 - 2.0)
  - Lower values (0.1-0.3): More focused and deterministic
  - Higher values (0.7-1.0): More creative and varied

- **Maximum tokens**: Maximum length of responses. The default is 1000. A
  structured AI task that needs more than this comes back as truncated JSON,
  which surfaces as `Error with Mistral AI structured response` — a parse
  failure that reads like a model problem but is a length one.

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

The model dropdown lists only models reporting the `audio_speech` capability.
The voice dropdown is populated from your account, because custom voices are
created against it — if the account has none, the field is omitted and the
API picks a voice itself. Callers can override the configured voice per
request with the `voice` option.

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

### Local Development

```bash
# Install dependencies (dev tools plus the current Home Assistant version)
uv sync

# Run the linters, type checker and consistency checks
uv run pre-commit run --all-files

# Run the tests
uv run pytest

# Run the tests against the oldest supported Home Assistant version
uv sync --no-default-groups --group dev --group ha-minimum
uv run --no-sync pytest
uv sync  # switch back
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
