"""Base entity for the Mistral AI integration."""

from __future__ import annotations

import base64
import json
import logging
from contextlib import asynccontextmanager
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
    ATTACHMENT_DOCUMENT_TYPE,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_WEB_SEARCH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    MAX_ATTACHMENT_BYTES,
    MAX_TEMPERATURE,
    MAX_TOOL_ITERATIONS,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
    TIMEOUT,
    WEB_SEARCH_TOOLS,
)
from .helpers import clamped_temperature

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable, Callable, Mapping
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


def _validation_detail(err: SDKError) -> str:
    """Return the readable part of a 422, or an empty string.

    The conversations endpoint validates a request against a union of two
    shapes -- one keyed by `model`, one by `agent_id` -- and reports failures
    against **both**. This integration only ever sends the first, so one wrong
    field arrives as four errors, three of them about a request that was never
    made:

        completion_args.temperature  Input should be less than or equal to 1
        agent_id                     Field required
        model                        Extra inputs are not permitted
        completion_args              Extra inputs are not permitted

    Roughly 2.5 kB of it, led by `Field required: agent_id` -- which sends the
    reader looking for an agent ID setting this integration does not have, and
    buries the one line that says what is actually wrong.

    So the entries for the branch we did not use are dropped. If none match --
    a different endpoint, or a shape that has changed -- everything is kept
    rather than nothing, because a wall of text still beats an empty message.

    Returns an empty string when the body cannot be parsed at all, which the
    caller treats as "fall back to the generic error".
    """
    try:
        body = json.loads(err.body or err.raw_response.text)
    except (ValueError, AttributeError):
        return ""

    entries = body.get("detail")
    if not isinstance(entries, list):
        return ""

    def _is_ours(entry: Any) -> bool:
        return "AgentConversationRequest" not in str(entry.get("loc", ""))

    ours = [entry for entry in entries if isinstance(entry, dict) and _is_ours(entry)]
    chosen = ours or [entry for entry in entries if isinstance(entry, dict)]

    messages = []
    for entry in chosen:
        # The field name, without the union machinery wrapped around it.
        parts = [
            str(part)
            for part in entry.get("loc", [])
            if isinstance(part, str) and "ConversationRequest" not in part
        ]
        field = ".".join(part for part in parts if part != "body")
        message = str(entry.get("msg", "")).strip()
        if message:
            messages.append(f"{field}: {message}" if field else message)

    return "; ".join(dict.fromkeys(messages))


def _reasoning_args(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return the reasoning effort to send, or nothing at all.

    Absent by default, and that is the whole point. Every other generation
    parameter here has a default sent on every request; this one must not,
    because a model that does not advertise the capability rejects the field
    outright rather than ignoring it:

        400 reasoning_effort is not enabled for this model

    So sending "none" to mean "off" would break every non-reasoning model --
    codestral, ministral, devstral, voxtral, and the pinned mistral-medium
    builds among them. Omitting the key leaves each model on its own default,
    which is the behaviour every subentry has today.

    The config flow is what guarantees a stored value only ever belongs to a
    model that accepts it: the field is offered only for a reasoning-capable
    model, and _prune_unsupported drops it when the model is changed to one
    that is not. Checking here instead would mean a model lookup on every
    request, on the path where the user is waiting for a reply.
    """
    if effort := options.get(CONF_REASONING_EFFORT):
        return {"reasoning_effort": effort}
    return {}


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
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="attachment_not_found",
                    translation_placeholders={"path": str(path)},
                )

            mime_type = attachment.mime_type or guess_file_type(path)[0]

            # Images go as image_url, PDFs as document_url. Anything else is
            # refused -- the message names what is accepted rather than saying
            # "only images", which read as a limit of the API and was not one.
            if mime_type and mime_type.startswith("image/"):
                chunk_type = "image_url"
            elif mime_type == ATTACHMENT_DOCUMENT_TYPE:
                chunk_type = "document_url"
            else:
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="unsupported_attachment_type",
                    translation_placeholders={"path": str(path)},
                )

            # Checked before reading, so an oversized file is never loaded at
            # all. The endpoint is not what this protects: it accepted a 30 MB
            # PDF happily. Attachments are inlined as base64, so the bytes and
            # a string a third larger are resident together, and Home Assistant
            # often runs somewhere that cannot spare it.
            if (size := path.stat().st_size) > MAX_ATTACHMENT_BYTES:
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="attachment_too_large",
                    translation_placeholders={
                        "path": str(path),
                        "size": f"{size / 1024 / 1024:.1f}",
                        "limit": f"{MAX_ATTACHMENT_BYTES // 1024 // 1024}",
                    },
                )

            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            chunks.append(
                {
                    "type": chunk_type,
                    chunk_type: f"data:{mime_type};base64,{encoded}",
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

    # Whether the model ever said anything, as opposed to only thinking, and
    # whether the response was cut off. Together they identify a reply that
    # ran out of room before it began -- see the check after the loop.
    said_something = False
    truncated = False

    async for chunk in stream:
        data = getattr(chunk, "data", None)
        if data is None or not data.choices:
            continue

        if data.choices[0].finish_reason == "length":
            truncated = True

        delta = data.choices[0].delta

        if not started:
            yield {"role": "assistant"}
            started = True

        if delta.content:
            # Content is usually a plain string, but reasoning models return a
            # list of chunks, mixing text with the model's thinking.
            if isinstance(delta.content, str):
                said_something = True
                yield {"content": delta.content}
            else:
                for part in delta.content:
                    if getattr(part, "type", None) == "thinking":
                        for thought in getattr(part, "thinking", None) or []:
                            if text := getattr(thought, "text", None):
                                yield {"thinking_content": text}
                    elif text := getattr(part, "text", None):
                        said_something = True
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
        return

    # Cut off before the model said anything. A reasoning model spends tokens
    # thinking before it answers, and maximum tokens caps the whole response,
    # so a low enough limit produces a successful reply containing only
    # thinking -- no error anywhere, and nothing for the user.
    #
    # Left alone when there is partial text: half an answer is worth more than
    # an error, and the user can see it was cut short. Only the case with
    # nothing at all is worth raising over, because it is indistinguishable
    # from the model having ignored the question.
    #
    # Also catches ordinary truncation with no reasoning involved, which is
    # the same failure with a different cause: an AI task whose structured
    # output was cut off used to arrive as "did not return valid structured
    # data", blaming the model for a length problem.
    if truncated and not said_something:
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="truncated_before_answering",
        )


async def _transform_conversation_stream(
    stream: AsyncIterable[Any],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform a conversations event stream into chat log deltas.

    A parallel to _transform_stream rather than a branch in it, because the
    conversations endpoint reports a different vocabulary of events. Four
    differences matter:

    - A failure arrives as an event carrying a message and a code, where chat
      completions raises. It is turned back into an exception here so callers
      see one shape of failure.
    - tool.execution events describe the connector's own work. Home Assistant
      has no concept for "the server is searching right now", so they are
      skipped -- but they arrive interleaved and must not be read as content.
    - A delta carries either a string or a single chunk, and a chunk may be
      text, thinking, or a citation. Citations are dropped for the same reason
      as in the non-streamed path: this reply is also spoken aloud.
    - Tool call arguments are fragmented, as in chat completions, but keyed by
      tool_call_id rather than by index. They are buffered and emitted at the
      end, because Home Assistant dispatches a tool call the moment it is
      yielded and a partial one would run with broken arguments.
    """
    started = False
    tool_calls: dict[str, dict[str, Any]] = {}

    async for event in stream:
        data = getattr(event, "data", None)
        if data is None:
            continue

        event_type = getattr(data, "type", None)

        if event_type == "conversation.response.error":
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="api_error",
                translation_placeholders={
                    "error": f"{getattr(data, 'message', 'unknown')} "
                    f"({getattr(data, 'code', 'no code')})"
                },
            )

        if event_type not in ("message.output.delta", "function.call.delta"):
            # Lifecycle, connector progress, and agent handoff. Nothing in the
            # chat log corresponds to any of them.
            continue

        if not started:
            yield {"role": "assistant"}
            started = True

        if event_type == "message.output.delta":
            for delta in _content_deltas(getattr(data, "content", None)):
                yield delta
            continue

        if not (tool_call_id := getattr(data, "tool_call_id", None)):
            continue
        buffered = tool_calls.setdefault(
            tool_call_id, {"id": tool_call_id, "name": None, "args": ""}
        )
        if name := getattr(data, "name", None):
            buffered["name"] = name
        if arguments := getattr(data, "arguments", None):
            buffered["args"] += arguments

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


def _content_deltas(content: Any) -> list[conversation.AssistantContentDeltaDict]:
    """Return the deltas a message delta's content is worth."""
    if content is None:
        return []

    if isinstance(content, str):
        return [{"content": content}] if content else []

    if getattr(content, "type", None) == "thinking":
        return [
            {"thinking_content": text}
            for thought in getattr(content, "thinking", None) or []
            if isinstance(text := getattr(thought, "text", None), str)
        ]

    # Anything else carrying text is text. A citation or a file reference
    # carries none, and is dropped.
    if isinstance(text := getattr(content, "text", None), str) and text:
        return [{"content": text}]

    return []


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

    @asynccontextmanager
    async def _translating_errors(
        self, *, transport_key: str = "api_error"
    ) -> AsyncGenerator[None]:
        """Turn anything the API raises into a Home Assistant error.

        `transport_key` names the message for a timeout or a connection
        failure, which is the only part that differs between callers --
        downloading a generated image says so rather than blaming the chat
        request.

        A HomeAssistantError passes through untouched. It comes from tool
        execution, or from our own conversion, and is already meaningful.

        The bare `Exception` arm is deliberate. The SDK keeps adding exception
        types that inherit from neither SDKError nor httpx.HTTPError -- 2.8.0
        added StreamDisconnectedError for mid-stream SSE errors -- and they are
        only reachable from private modules, so catching them by name would tie
        us to SDK internals. Anything unexpected becomes a clean error rather
        than a traceback in the user's face.
        """
        try:
            yield
        except SDKError as err:
            raise self._convert_error(err) from err
        except (TimeoutError, httpx.HTTPError) as err:
            _LOGGER.error("Error talking to Mistral AI: %s", err)
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key=transport_key,
                translation_placeholders={"error": str(err)},
            ) from err
        except HomeAssistantError:
            raise
        except Exception as err:
            _LOGGER.exception("Unexpected error from Mistral AI")
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="unexpected_error",
                translation_placeholders={"error": str(err)},
            ) from err

    async def _async_handle_chat_log_with_search(
        self, chat_log: conversation.ChatLog
    ) -> None:
        """Generate an answer through the conversations endpoint.

        Web search is a built-in connector, and connectors do not run on chat
        completions: that endpoint accepts the tool and then either rejects it
        outright or never runs it, because its responses have nowhere to carry
        the references a connector returns.

        Streamed through a transform of its own, because this endpoint reports
        a different vocabulary of events -- see
        _transform_conversation_stream.
        """
        options = self.subentry.data
        client = self.entry.runtime_data.client

        tools: list[dict[str, Any]] = []
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]
        tools.append({"type": options[CONF_WEB_SEARCH]})

        for _iteration in range(MAX_TOOL_ITERATIONS):
            # Rebuilt each pass so tool results from the previous one are
            # included. The endpoint would keep the conversation itself, but
            # store is off, so the history is re-sent instead -- exactly as
            # the chat completions path re-sends its messages.
            instructions, inputs = await self._async_conversation_inputs(chat_log)

            async with self._translating_errors():
                stream = await client.beta.conversations.start_stream_async(
                    model=options.get(CONF_MODEL, DEFAULT_MODEL),
                    instructions=instructions,
                    inputs=inputs,
                    tools=tools,
                    # No handoff_execution. The SDK accepts it and this used to
                    # send "client", which the endpoint rejects outright when
                    # the request carries a model rather than an agent_id:
                    #
                    #   422 Conversation with a 'model' can't contain the
                    #       following fields handoff_execution
                    #
                    # Every web search request failed on it. The field belongs
                    # to agent-based conversations, and the OpenAPI schema does
                    # not show that -- it is declared on the shared request
                    # base that both variants inherit, and the restriction is a
                    # cross-field validator rather than anything in the schema.
                    #
                    # Nothing is lost by dropping it: a model-based
                    # conversation hands function tools back to the caller
                    # anyway. Checked with a real request -- a turn carrying
                    # both a web_search connector and a Home Assistant function
                    # returns a function.call entry for us to run, exactly as
                    # the explicit "client" was meant to ask for.
                    #
                    # Explicit: this endpoint retains conversations and lists
                    # them afterwards, where chat completions stores nothing.
                    store=False,
                    completion_args={
                        "temperature": clamped_temperature(
                            options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                            MAX_TEMPERATURE[SUBENTRY_TYPE_CONVERSATION],
                        ),
                        "top_p": options.get(CONF_TOP_P, DEFAULT_TOP_P),
                        "max_tokens": options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                        # Only when set, and only ever set on a model that
                        # advertises reasoning -- see _reasoning_args.
                        **_reasoning_args(options),
                    },
                    timeout_ms=TIMEOUT * 1000,
                )

                async for _content in chat_log.async_add_delta_content_stream(
                    self.entity_id, _transform_conversation_stream(stream)
                ):
                    pass

            if not chat_log.unresponded_tool_results:
                break

    async def _async_conversation_inputs(
        self, chat_log: conversation.ChatLog
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Split a chat log into conversation instructions and input entries.

        The endpoint has no system role -- a message entry is user or
        assistant only -- and takes the system prompt as `instructions`
        instead. Tool calls and their results are entries of their own rather
        than fields on a message, which is the other shape difference from
        chat completions.
        """
        messages = await self._async_chat_log_messages(chat_log)

        instructions: list[str] = []
        inputs: list[dict[str, Any]] = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                if isinstance(content, str):
                    instructions.append(content)
            elif role == "tool":
                inputs.append(
                    {
                        "type": "function.result",
                        "tool_call_id": message["tool_call_id"],
                        "result": content,
                    }
                )
            else:
                if content or not message.get("tool_calls"):
                    inputs.append({"role": role, "content": content})
                inputs.extend(
                    {
                        "type": "function.call",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"],
                    }
                    for tool_call in message.get("tool_calls") or []
                )

        return "\n\n".join(instructions) or None, inputs

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
        client = self.entry.runtime_data.client

        # Connectors only run on the conversations endpoint, so an agent with
        # web search on takes an entirely different path -- see
        # _async_handle_chat_log_with_search for what that costs.
        if options.get(CONF_WEB_SEARCH) in WEB_SEARCH_TOOLS:
            await self._async_handle_chat_log_with_search(chat_log)
            return

        model_args: dict[str, Any] = {
            "model": options.get(CONF_MODEL, DEFAULT_MODEL),
            # The chat completions ceiling, not the conversations one. This
            # path is only reached with web search off; with it on the request
            # goes to the other endpoint above, which is stricter.
            "temperature": clamped_temperature(
                options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                MAX_TEMPERATURE[SUBENTRY_TYPE_AI_TASK_DATA],
            ),
            # Bounded at 1.0 by both endpoints, so unlike temperature there is
            # no per-endpoint ceiling to pick between.
            "top_p": options.get(CONF_TOP_P, DEFAULT_TOP_P),
            "max_tokens": options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            # Accepted top level here and inside completion_args there, with
            # the same values in both, so unlike temperature there is no
            # per-endpoint ceiling to pick between.
            **_reasoning_args(options),
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

            async with self._translating_errors():
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

            if not chat_log.unresponded_tool_results:
                break

    def _convert_error(self, err: SDKError) -> HomeAssistantError:
        """Map a Mistral AI SDK error onto a Home Assistant error."""
        if err.status_code == 401:
            # Prompt for a new key rather than failing on every future
            # sentence.
            self.entry.async_start_reauth(self.hass)
            return HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="invalid_api_key",
            )
        if err.status_code == 403:
            # Not an authentication failure, despite the neighbouring status
            # code, and this used to be handled as one.
            #
            # Every way of getting the key wrong -- a wrong key, no
            # Authorization header, a header without the Bearer prefix --
            # answers 401 with {"detail": "Invalid API Key"}. 403 is what the
            # API says when the key is fine and the account is not allowed to
            # do the thing, which a new key cannot fix. Treating it as auth
            # meant a user with, say, no access to the model they had typed in
            # was told their key had been rejected and handed a dialog asking
            # for another one.
            _LOGGER.error("Mistral AI refused the request: %s", err)
            return HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="forbidden",
            )
        if err.status_code == 429:
            return HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="rate_limited",
            )
        if err.status_code == 422 and (detail := _validation_detail(err)):
            _LOGGER.error("Mistral AI rejected the request: %s", err)
            return HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="invalid_request",
                translation_placeholders={"detail": detail},
            )
        _LOGGER.error("Mistral AI API error: %s", err)
        return HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="api_error",
            translation_placeholders={"error": str(err)},
        )
