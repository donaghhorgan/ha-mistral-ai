"""Base entity for the Mistral AI integration."""

from __future__ import annotations

import base64
import json
import logging
from mimetypes import guess_file_type
from typing import TYPE_CHECKING, Any

import httpx
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from homeassistant.helpers.entity import Entity
from homeassistant.util import slugify
from mistralai.client.errors import SDKError
from voluptuous_openapi import convert

from .const import (
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    MAX_TOOL_ITERATIONS,
    TIMEOUT,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable, Callable
    from pathlib import Path

    from . import MistralConfigEntry

_LOGGER = logging.getLogger(__name__)


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format a Home Assistant tool for the Mistral AI API."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
        },
    }


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    """Coerce tool call arguments to a dict.

    Mistral types these as ``Union[dict, str]``: complete responses may return
    either, and streamed responses always deliver a JSON string.
    """
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as err:
        raise HomeAssistantError(
            f"Mistral AI returned malformed tool arguments: {arguments!r}"
        ) from err
    if not isinstance(parsed, dict):
        raise HomeAssistantError(
            f"Mistral AI returned non-object tool arguments: {arguments!r}"
        )
    return parsed


def _convert_content(content: conversation.Content) -> dict[str, Any]:
    """Convert Home Assistant chat log content to a Mistral AI message."""
    if isinstance(content, conversation.UserContent):
        return {"role": "user", "content": content.content}

    if isinstance(content, conversation.SystemContent):
        return {"role": "system", "content": content.content}

    if isinstance(content, conversation.AssistantContent):
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content.content or "",
        }
        if content.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.tool_name,
                        "arguments": json.dumps(tool_call.tool_args),
                    },
                }
                for tool_call in content.tool_calls
            ]
        return message

    if isinstance(content, conversation.ToolResultContent):
        return {
            "role": "tool",
            "name": content.tool_name,
            "tool_call_id": content.tool_call_id,
            "content": json.dumps(content.tool_result),
        }

    raise HomeAssistantError(f"Unsupported content type: {type(content)}")


async def _async_attachment_chunks(
    hass: HomeAssistant, attachments: list[conversation.Attachment]
) -> list[dict[str, Any]]:
    """Read attachments and convert them to Mistral content chunks.

    Mistral takes images inline as data URIs, so the files have to be read and
    base64 encoded. That is blocking, hence the executor.
    """

    def _read() -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for attachment in attachments:
            path: Path = attachment.path
            if not path.exists():
                raise HomeAssistantError(f"`{path}` does not exist")

            mime_type = attachment.mime_type or guess_file_type(path)[0]
            if not mime_type or not mime_type.startswith("image/"):
                raise HomeAssistantError(
                    "Only images are supported by the Mistral AI API, "
                    f"`{path}` is not an image"
                )

            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            chunks.append(
                {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{encoded}",
                }
            )
        return chunks

    return await hass.async_add_executor_job(_read)


async def _async_convert_messages(
    hass: HomeAssistant, contents: list[conversation.Content]
) -> list[dict[str, Any]]:
    """Convert a chat log into Mistral messages, including any attachments."""
    messages: list[dict[str, Any]] = []

    for content in contents:
        message = _convert_content(content)

        # A user turn carrying images becomes a list of chunks rather than a
        # plain string. Only user turns can carry attachments.
        if isinstance(content, conversation.UserContent) and content.attachments:
            chunks = await _async_attachment_chunks(hass, content.attachments)
            message["content"] = [
                {"type": "text", "text": content.content},
                *chunks,
            ]

        messages.append(message)

    return messages


async def _transform_stream(
    stream: AsyncIterable[Any],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform a Mistral AI stream into Home Assistant chat log deltas.

    Tool call arguments arrive as fragments of a JSON string spread across
    chunks, so they are buffered and only emitted once the stream completes.
    Home Assistant dispatches a tool call the moment it is yielded, so
    yielding a partial one would invoke the tool with broken arguments.
    """
    started = False
    # Keyed by the index reported by the API so that parallel tool calls in
    # one response do not overwrite each other.
    tool_calls: dict[int, dict[str, Any]] = {}

    async for chunk in stream:
        data = getattr(chunk, "data", None)
        if data is None or not data.choices:
            continue

        delta = data.choices[0].delta

        if not started:
            yield {"role": "assistant"}
            started = True

        if delta.content:
            # Content is usually a plain string, but reasoning models return a
            # list of chunks, mixing text with the model's thinking.
            if isinstance(delta.content, str):
                yield {"content": delta.content}
            else:
                for part in delta.content:
                    if getattr(part, "type", None) == "thinking":
                        for thought in getattr(part, "thinking", None) or []:
                            if text := getattr(thought, "text", None):
                                yield {"thinking_content": text}
                    elif text := getattr(part, "text", None):
                        yield {"content": text}

        for index, tool_call in enumerate(delta.tool_calls or []):
            key = tool_call.index if tool_call.index is not None else index
            buffered = tool_calls.setdefault(
                key, {"id": None, "name": None, "args": ""}
            )
            if tool_call.id:
                buffered["id"] = tool_call.id
            if tool_call.function is None:
                continue
            if tool_call.function.name:
                buffered["name"] = tool_call.function.name
            arguments = tool_call.function.arguments
            if isinstance(arguments, str):
                buffered["args"] += arguments
            elif arguments:
                buffered["args"] = json.dumps(arguments)

    if complete := [call for call in tool_calls.values() if call["name"]]:
        yield {
            "tool_calls": [
                llm.ToolInput(
                    id=call["id"],
                    tool_name=call["name"],
                    tool_args=_parse_arguments(call["args"]),
                )
                for call in complete
            ]
        }


class MistralBaseEntity(Entity):
    """Everything a Mistral AI entity needs regardless of what it does.

    Split out from MistralBaseLLMEntity so that the speech platforms, which
    hold no chat log and call no tools, do not inherit machinery they never
    use.
    """

    def __init__(self, entry: MistralConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        # Named explicitly rather than inheriting the device name via
        # _attr_has_entity_name. The two look identical in the UI, but the
        # text-to-speech component rejects an entity whose name resolves to
        # UNDEFINED, which is what the inherited form gives it:
        #
        #   if engine_instance.name is None or engine_instance.name is UNDEFINED:
        #       raise HomeAssistantError("TTS engine name is not set.")
        #
        # It fails before any request is made, so nothing appears in this
        # integration's log at all. google_generative_ai_conversation names
        # its entities the same way, for what is presumably the same reason.
        self._attr_name = subentry.title

        model = subentry.data.get(CONF_MODEL, DEFAULT_MODEL)
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Mistral AI",
            model=model,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def attribution(self) -> str:
        """Return the attribution."""
        return "Powered by Mistral AI"


class MistralBaseLLMEntity(MistralBaseEntity):
    """Mistral AI base LLM entity."""

    async def _async_chat_log_messages(
        self, chat_log: conversation.ChatLog
    ) -> list[dict[str, Any]]:
        """Convert a chat log into Mistral messages.

        Exists so the image path can build its request from the log too,
        without reaching for a private module function.
        """
        return await _async_convert_messages(self.hass, chat_log.content)

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
    ) -> None:
        """Generate an answer for the chat log."""
        options = self.subentry.data
        client = self.entry.runtime_data

        model_args: dict[str, Any] = {
            "model": options.get(CONF_MODEL, DEFAULT_MODEL),
            "temperature": options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            "max_tokens": options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
        }

        # llm.selector_serializer rather than None when there is no LLM API.
        # An AI task never has one, and voluptuous_openapi cannot convert a
        # Home Assistant selector without it -- every structured generation
        # died on `cannot use 'TextSelector' as a dict key`. This is what
        # openai_conversation does for the same reason.
        custom_serializer = (
            chat_log.llm_api.custom_serializer
            if chat_log.llm_api
            else llm.selector_serializer
        )

        if chat_log.llm_api and (
            tools := [
                _format_tool(tool, custom_serializer) for tool in chat_log.llm_api.tools
            ]
        ):
            # An empty tools list is rejected by the API, so only send it when
            # the selected LLM API actually exposes something.
            model_args["tools"] = tools

        if structure and structure_name:
            # Mistral supports structured output natively, so there is no need
            # to smuggle the schema through as a synthetic tool.
            model_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": slugify(structure_name),
                    "schema": convert(structure, custom_serializer=custom_serializer),
                    "strict": True,
                },
            }

        for _iteration in range(MAX_TOOL_ITERATIONS):
            # Rebuild the messages each pass so that tool results added by the
            # previous iteration are included.
            messages = await _async_convert_messages(self.hass, chat_log.content)

            try:
                # timeout_ms is applied by the SDK per read, so a stalled
                # stream fails rather than hanging, while a long but steady
                # response is left alone. Wrapping the loop in a deadline
                # instead would also cut off legitimate tool execution.
                stream = await client.chat.stream_async(
                    messages=messages, timeout_ms=TIMEOUT * 1000, **model_args
                )

                async for _content in chat_log.async_add_delta_content_stream(
                    self.entity_id, _transform_stream(stream)
                ):
                    pass
            except SDKError as err:
                raise self._convert_error(err) from err
            except (TimeoutError, httpx.HTTPError) as err:
                _LOGGER.error("Error talking to Mistral AI: %s", err)
                raise HomeAssistantError(f"Error talking to Mistral AI: {err}") from err
            except HomeAssistantError:
                # Raised by tool execution, or by our own conversion above.
                # Already meaningful, so let it through untouched.
                raise
            except Exception as err:
                # The SDK keeps adding exception types that inherit from
                # neither SDKError nor httpx.HTTPError -- 2.8.0 added
                # StreamDisconnectedError for mid-stream SSE errors -- and
                # they are only reachable from private modules, so catching
                # them by name would tie us to SDK internals. Anything
                # unexpected becomes a clean error rather than a traceback in
                # the user's face.
                _LOGGER.exception("Unexpected error from Mistral AI")
                raise HomeAssistantError(
                    f"Unexpected error from Mistral AI: {err}"
                ) from err

            if not chat_log.unresponded_tool_results:
                break

    def _convert_error(self, err: SDKError) -> HomeAssistantError:
        """Map a Mistral AI SDK error onto a Home Assistant error."""
        if err.status_code in (401, 403):
            # Prompt for a new key rather than failing on every future
            # sentence.
            self.entry.async_start_reauth(self.hass)
            return HomeAssistantError(
                "Invalid Mistral AI API key, please reconfigure the integration"
            )
        if err.status_code == 429:
            return HomeAssistantError("Rate limited by Mistral AI, please try again")
        _LOGGER.error("Mistral AI API error: %s", err)
        return HomeAssistantError(f"Error talking to Mistral AI: {err}")
