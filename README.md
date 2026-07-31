# Mistral AI Conversation for Home Assistant

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub Release](https://img.shields.io/github/release/donaghhorgan/ha-mistral-conversation.svg)](https://github.com/donaghhorgan/ha-mistral-conversation/releases)
[![License](https://img.shields.io/github/license/donaghhorgan/ha-mistral-conversation.svg)](LICENSE)

A Home Assistant integration that provides a conversation agent powered by
Mistral AI. This integration allows you to use Mistral AI's language models as
a conversation agent for voice assistants, chatbots, and other conversational
interfaces in Home Assistant.

## Features

- 🤖 **Any available model**: The model list is fetched from the API using your
  own key, so it reflects what you can actually use
- 🏠 **Device control**: Optionally expose a Home Assistant LLM API so the agent
  can control your devices via tool calling
- ⚡ **Streaming responses**: Replies are streamed back as they are generated
- 📋 **AI tasks**: Generate data for automations and dashboards, with native
  structured output
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
4. Add this repository URL: `https://github.com/donaghhorgan/ha-mistral-conversation`
5. Select "Integration" as the category
6. Click "Add"
7. Search for "Mistral AI Conversation" and install it
8. Restart Home Assistant

### Manual Installation

1. Download the latest release from the [releases page](https://github.com/donaghhorgan/ha-mistral-conversation/releases)
2. Extract the contents to your `custom_components` directory
3. The folder structure should look like:

   ```text
   custom_components/
   └── mistral_conversation/
       ├── __init__.py
       ├── ai_task.py
       ├── config_flow.py
       ├── const.py
       ├── conversation.py
       ├── entity.py
       ├── manifest.json
       └── translations/
           └── en.json
   ```

4. Restart Home Assistant

## Configuration

### Initial Setup

1. Go to Settings → Devices & Services
2. Click "Add Integration"
3. Search for "Mistral AI Conversation"
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

1. Go to Settings → Devices & Services → Mistral AI Conversation
2. Click "Add conversation agent" or "Add AI task"
3. Configure it, then click "Submit"

To change an existing one, click "Configure" next to it.

### Configuration options

- **Model**: Chosen from the models your API key can actually reach — the
  list is fetched from the API rather than hard-coded, so it stays correct as
  Mistral adds and retires models. You can also type a model name directly.

- **Temperature**: Controls randomness in responses (0.0 - 2.0)
  - Lower values (0.1-0.3): More focused and deterministic
  - Higher values (0.7-1.0): More creative and varied

- **Maximum tokens**: Maximum length of responses

- **Instructions** (conversation agents only): Custom instructions for the
  assistant. You can use Home Assistant template variables like
  `{{ ha_name }}`.

- **Control Home Assistant** (conversation agents only): Select a Home
  Assistant LLM API to let the agent control your devices. Leave it unset for
  an agent that only answers questions.

### AI tasks

As well as conversation agents, this integration provides
[AI Task](https://www.home-assistant.io/integrations/ai_task/) entities, which
generate data for automations and dashboards rather than holding a
conversation. Add one with "Add AI task", then call
`ai_task.generate_data` with its entity ID. Passing a structure returns
validated JSON, using Mistral's native structured output.

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
    custom_components.mistral_conversation: debug
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
# Install all dependencies, including the dev group
uv sync --all-groups

# Run the linters, type checker and consistency checks
uv run pre-commit run --all-files

# Run the tests
uv run pytest
```

See [`AGENTS.md`](AGENTS.md) for conventions and
[`scripts/README.md`](scripts/README.md) for the consistency checks.

## Support

- **Issues**: [GitHub Issues](https://github.com/donaghhorgan/ha-mistral-conversation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/donaghhorgan/ha-mistral-conversation/discussions)
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
