"""Base entity for the Mistral AI integration."""

from __future__ import annotations

import base64
import json
import logging
from contextlib import asynccontextmanager
from mimetypes import guess_file_type
from typing import TYPE_CHECKING, Any, NoReturn

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
    CONF_WEB_SEARCH_CITATIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_WEB_SEARCH_CITATIONS,
    DOMAIN,
    MAX_ATTACHMENT_BYTES,
    MAX_TEMPERATURE,
    MAX_TOOL_ITERATIONS,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
    TIMEOUT,
    WEB_SEARCH_CITATIONS_PROMPT,
    WEB_SEARCH_TOOLS,
)
from .helpers import clamped_temperature

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable, Callable, Mapping
    from pathlib import Path

    from mistralai.client import Mistral

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
        raw = err.body or err.raw_response.text
    except AttributeError:
        return ""

    return _validation_detail_from_body(raw)


def _validation_detail_from_body(raw: str) -> str:
    """Return the readable part of a 422 body, or an empty string.

    Split from _validation_detail so the streamed speech path can use it: it
    builds its own request and has a response body rather than an SDKError.
    """
    try:
        body = json.loads(raw)
    except (ValueError, TypeError):
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


def _raise_tool_loop_exhausted() -> NoReturn:
    """Report a model that never stopped calling tools.

    Both tool loops run a fixed number of passes and break when the model
    stops asking for tools. Falling off the end used to be silent: no log
    line, no error, and a chat log still holding tool results that were never
    sent back.

    What the user saw did not name the cause on either platform. A
    conversation returned whatever fragmentary text came with the last tool
    call, usually nothing, indistinguishable from the model having had nothing
    to say. An AI task hit `chat_log.content[-1]` not being an
    AssistantContent and raised "Last content in chat log is not an
    AssistantContent" -- untranslated, internal-sounding, and about a symptom
    rather than the cause.

    Raising rather than returning the empty reply follows what #131 settled
    for truncation: a successful-looking response with no answer in it is
    reported, not passed up as if it were an answer. For AI tasks this only
    replaces a worse message with a better one, since that path already
    failed.
    """
    _LOGGER.warning(
        "Mistral AI kept calling tools for all %s rounds without finishing; "
        "giving up on this turn",
        MAX_TOOL_ITERATIONS,
    )
    raise HomeAssistantError(
        translation_domain=DOMAIN,
        translation_key="tool_loop_exhausted",
        translation_placeholders={"limit": str(MAX_TOOL_ITERATIONS)},
    )


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
            "content": json.dumps(content.tool_result, default=str),
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


# A stand-in for the real connector, attached to chat completions instead of
# always paying for the conversations endpoint -- see
# _async_detect_search_escalation. Calling it never actually searches
# anything: the model calling it at all is the only signal used, so the
# description only has to get the model to reach for it at the right moments,
# not describe a real contract.
#
# Verified against the real API (mistral-small-2603, 24 probes across eight
# prompts): called on every prompt that needed a search (weather, news, an
# explicit search request) and zero of the ones that did not (small talk,
# arithmetic, general knowledge, a plain Home Assistant tool call) -- and
# correctly called alongside a real Home Assistant tool in the same turn when
# a prompt asked for both, in either order.
LAZY_SEARCH_TOOL_NAME = "web_search"

LAZY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": LAZY_SEARCH_TOOL_NAME,
        "description": (
            "Search the web for information you do not already know, such as "
            "news, weather, sports scores, or anything time-sensitive."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


async def _async_detect_search_escalation(
    stream: AsyncIterable[Any],
) -> tuple[bool, list[Any]]:
    """Peek a chat-completions stream for a call to the lazy search stand-in.

    Verified against the real API: a turn's tool calls arrive complete, in
    the same chunk that carries finish_reason "tool_calls" -- never split
    across chunks, never alongside content. So the decision is safe to make
    as soon as either signal appears: real content means the model is
    answering directly, and finish_reason "tool_calls" means every tool call
    for this turn is already in the delta that carries it.

    Returns (escalate, consumed). `consumed` holds every chunk already read
    off `stream` -- an async generator cannot be un-consumed, so the caller
    replays them ahead of whatever remains of `stream` when escalate is
    False, through _prefixed_stream, and _transform_stream then sees the same
    sequence it would have seen without the detour.
    """
    consumed: list[Any] = []
    async for chunk in stream:
        consumed.append(chunk)
        data = getattr(chunk, "data", None)
        if data is None or not data.choices:
            continue

        delta = data.choices[0].delta
        if delta.content:
            return False, consumed

        if data.choices[0].finish_reason == "tool_calls":
            names = {
                tool_call.function.name
                for tool_call in delta.tool_calls or []
                if tool_call.function is not None
            }
            return LAZY_SEARCH_TOOL_NAME in names, consumed

    return False, consumed


async def _prefixed_stream(
    prefix: list[Any], rest: AsyncIterable[Any]
) -> AsyncGenerator[Any]:
    """Replay chunks already read off a stream, then continue reading it."""
    for chunk in prefix:
        yield chunk
    async for chunk in rest:
        yield chunk


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
    stream: AsyncIterable[Any], max_tokens: int, valid_tool_names: frozenset[str]
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform a conversations event stream into chat log deltas.

    A parallel to _transform_stream rather than a branch in it, because the
    conversations endpoint reports a different vocabulary of events. Five
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
    - A function.call.delta can carry a name that belongs to neither Home
      Assistant nor the connector -- observed once as a comma-joined run of
      other tool-call IDs and digits, on the web-search path specifically.
      Dispatching it can only fail ("Tool ... not found"), costing a full
      round trip while the model recovers, so it is dropped against
      valid_tool_names instead: the names chat_log.llm_api actually exposes,
      which the connector itself is never a member of since it runs
      server-side and never reaches Home Assistant as a tool call.

    Truncation is the sixth difference, and the reason max_tokens is passed
    in. No conversation event carries a finish reason -- searching the SDK's
    models for one finds it only on the two chat-completions shapes -- so the
    test #131 applies on the other path is simply unavailable here, and a
    reply cut off before it said anything arrived as silence.

    What the endpoint does report is usage on the closing event, and a
    truncated response spends the cap exactly. Measured: max_tokens 12 with
    reasoning on returns completion_tokens 12 and a content list holding only
    thinking, while an answer that finishes uses 13 of a 500 or 3000 cap. So
    reaching the ceiling stands in for finish_reason "length" here, and the
    rest of the rule matches _transform_stream -- raise only when nothing was
    said and no tool call was made.
    """
    started = False
    said_something = False
    completion_tokens = 0
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

        if event_type == "conversation.response.done":
            usage = getattr(data, "usage", None)
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            continue

        if event_type not in ("message.output.delta", "function.call.delta"):
            # Lifecycle, connector progress, and agent handoff. Nothing in the
            # chat log corresponds to any of them.
            continue

        if not started:
            yield {"role": "assistant"}
            started = True

        if event_type == "message.output.delta":
            for delta in _content_deltas(getattr(data, "content", None)):
                # Thinking is not an answer. _content_deltas emits it so the
                # reasoning is visible, but a reply made only of it has told
                # the user nothing.
                said_something = said_something or "content" in delta
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

    if phantom := [
        call["name"]
        for call in tool_calls.values()
        if call["name"] and call["name"] not in valid_tool_names
    ]:
        _LOGGER.debug(
            "Dropping tool call(s) for unknown tool(s) %s from the "
            "conversations endpoint's event stream",
            phantom,
        )

    if complete := [
        call
        for call in tool_calls.values()
        if call["name"] and call["name"] in valid_tool_names
    ]:
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

    # A truncated turn that produced a tool call still dispatches it, handled
    # by the return above: the tool call is the answer in that turn.
    if completion_tokens >= max_tokens and not said_something:
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="truncated_before_answering",
        )


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

    def _error_for_status(
        self, status_code: int | None, detail: str, validation: str = ""
    ) -> HomeAssistantError:
        """Map an HTTP status onto a Home Assistant error.

        Split out of _convert_error so the streamed speech path shares it.
        That path builds its own request, because the SDK has no streaming
        speech method, and without this it was the one user-facing error in
        the integration with no translation key -- an English f-string with
        the raw response body in it.
        """
        if status_code == 401:
            # Prompt for a new key rather than failing on every future
            # sentence.
            self.entry.async_start_reauth(self.hass)
            return HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="invalid_api_key",
            )
        if status_code == 403:
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
            _LOGGER.error("Mistral AI refused the request: %s", detail)
            return HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="forbidden",
            )
        if status_code == 429:
            return HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="rate_limited",
            )
        if (
            status_code == 400
            and "does not support" in detail.lower()
            and "connector" in detail.lower()
        ):
            # The model picker offers models the conversations endpoint will
            # then refuse, because nothing filters on connector support --
            # the API reports no capability flag for it, unlike
            # audio_transcription and the others model_choices filters on.
            # Every turn on this pair failed with the raw body read aloud:
            #
            #   API error occurred: Status 400. Body: {"message":"Model
            #   ministral-8b-2512 currently does not support builtin
            #   connectors.", ...}
            #
            # So this names the actual conflict instead -- the model
            # configured for the agent, not the one the API happened to echo
            # back, since the two are the same thing here but the local value
            # does not depend on the API's wording holding still.
            _LOGGER.error("Mistral AI rejected the request: %s", detail)
            return HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="web_search_unsupported_model",
                translation_placeholders={
                    "model": self.subentry.data.get(CONF_MODEL, DEFAULT_MODEL)
                },
            )
        if status_code == 422 and validation:
            _LOGGER.error("Mistral AI rejected the request: %s", detail)
            return HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="invalid_request",
                translation_placeholders={"detail": validation},
            )
        _LOGGER.error("Mistral AI API error: %s", detail)
        return HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="api_error",
            translation_placeholders={"error": detail},
        )


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
        max_tokens = options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        valid_tool_names = frozenset(
            tool.name for tool in (chat_log.llm_api.tools if chat_log.llm_api else [])
        )

        for _iteration in range(MAX_TOOL_ITERATIONS):
            # Rebuilt each pass so tool results from the previous one are
            # included. The endpoint would keep the conversation itself, but
            # store is off, so the history is re-sent instead -- exactly as
            # the chat completions path re-sends its messages.
            instructions, inputs = await self._async_conversation_inputs(chat_log)

            # Read here rather than baked into the stored prompt, so that
            # turning it on reaches every agent rather than only ones created
            # afterwards -- see WEB_SEARCH_CITATIONS_PROMPT.
            if options.get(CONF_WEB_SEARCH_CITATIONS, DEFAULT_WEB_SEARCH_CITATIONS):
                instructions = "\n\n".join(
                    part for part in (instructions, WEB_SEARCH_CITATIONS_PROMPT) if part
                )

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
                        "max_tokens": max_tokens,
                        # Only when set, and only ever set on a model that
                        # advertises reasoning -- see _reasoning_args.
                        **_reasoning_args(options),
                    },
                    timeout_ms=TIMEOUT * 1000,
                )

                async for _content in chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    _transform_conversation_stream(
                        stream, max_tokens, valid_tool_names
                    ),
                ):
                    pass

            if not chat_log.unresponded_tool_results:
                break
        else:
            _raise_tool_loop_exhausted()

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

        # Connectors only run on the conversations endpoint, and attaching
        # one there unconditionally cost every turn ~0.7s of time-to-first-
        # token to serve a feature roughly 1 turn in 40 actually needed --
        # #176. So an agent with web search on still starts here, on chat
        # completions, carrying a lightweight stand-in tool alongside its
        # real ones; only a turn that calls it escalates to the conversations
        # endpoint, which is where the cost belongs.
        wants_search = options.get(CONF_WEB_SEARCH) in WEB_SEARCH_TOOLS

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

        tools: list[dict[str, Any]] = []
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, custom_serializer) for tool in chat_log.llm_api.tools
            ]

        if wants_search and any(
            tool["function"]["name"] == LAZY_SEARCH_TOOL_NAME for tool in tools
        ):
            # A real tool already claims the stand-in's name -- an edge case
            # worth not colliding with rather than one worth solving cleverly.
            # Falls back to what every web-search turn did before lazy
            # escalation existed: the connector attached unconditionally.
            await self._async_handle_chat_log_with_search(chat_log)
            return

        if tools or wants_search:
            # An empty tools list is rejected by the API, so only send it when
            # there is something to send -- either real tools, or (with
            # search wanted) at least the stand-in.
            model_args["tools"] = [*tools, LAZY_SEARCH_TOOL] if wants_search else tools

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

            if await self._async_stream_chat_completions(
                chat_log, client, messages, model_args, wants_search=wants_search
            ):
                # The stand-in was called: nothing from this attempt was kept
                # -- see _async_stream_chat_completions -- and the rest of
                # this turn, including any further tool-calling rounds, is
                # handed to the endpoint that can actually search.
                await self._async_handle_chat_log_with_search(chat_log)
                return

            if not chat_log.unresponded_tool_results:
                break
        else:
            _raise_tool_loop_exhausted()

    async def _async_stream_chat_completions(
        self,
        chat_log: conversation.ChatLog,
        client: Mistral,
        messages: list[dict[str, Any]],
        model_args: dict[str, Any],
        *,
        wants_search: bool,
    ) -> bool:
        """Stream one chat-completions round; return whether it escalated.

        Split out of _async_handle_chat_log because escalating has to return
        True from inside the try/except _translating_errors wraps, and
        looping back around to try the conversations endpoint from in there
        would nest one tool loop inside another for no reason -- the caller
        already has one.

        When wants_search is set, model_args carries the lazy stand-in
        alongside the real tools, and a stream that calls it is abandoned
        without dispatching anything: chat_log never learns the stand-in was
        called, so nothing needs to be undone before the same turn is retried
        on the endpoint that can actually search.
        """
        async with self._translating_errors():
            # timeout_ms is applied by the SDK per read, so a stalled stream
            # fails rather than hanging, while a long but steady response is
            # left alone. Wrapping the loop in a deadline instead would also
            # cut off legitimate tool execution.
            stream = await client.chat.stream_async(
                messages=messages, timeout_ms=TIMEOUT * 1000, **model_args
            )

            if wants_search:
                escalate, consumed = await _async_detect_search_escalation(stream)
                if escalate:
                    # Abandoned before the SSE sentinel, or even EOF -- ending
                    # the async for loop early leaves the underlying response
                    # open otherwise, since EventStreamAsync only closes it
                    # once the stream is read to completion. A stream built
                    # from a test double has no `response` to close.
                    if (response := getattr(stream, "response", None)) is not None:
                        await response.aclose()
                    return True
                stream = _prefixed_stream(consumed, stream)

            async for _content in chat_log.async_add_delta_content_stream(
                self.entity_id, _transform_stream(stream)
            ):
                pass

        return False

    def _convert_error(self, err: SDKError) -> HomeAssistantError:
        """Map a Mistral AI SDK error onto a Home Assistant error."""
        return self._error_for_status(
            err.status_code,
            str(err),
            _validation_detail(err) if err.status_code == 422 else "",
        )
