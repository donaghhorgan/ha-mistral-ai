"""AI Task platform for the Mistral AI integration."""

from __future__ import annotations

import logging
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any

import httpx
from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads
from mistralai.client.errors import SDKError

from .const import (
    CONF_MODEL,
    DEFAULT_MODEL,
    IMAGE_GENERATION_TOOL,
    SUBENTRY_TYPE_AI_TASK_DATA,
    TIMEOUT,
)
from .entity import MistralBaseLLMEntity

if TYPE_CHECKING:
    from . import MistralConfigEntry

_LOGGER = logging.getLogger(__name__)

# GENERATE_IMAGE arrived after the Home Assistant version in hacs.json --
# 2025.8.0 has only GENERATE_DATA and SUPPORT_ATTACHMENTS, and no GenImageTask
# either. Naming it unconditionally is an AttributeError at import, which does
# not degrade the feature, it stops the whole integration loading.
#
# So it is added when the running Home Assistant has it, and left out when it
# does not. The annotations on _async_generate_image are safe regardless:
# `from __future__ import annotations` keeps them unevaluated, and the method
# is never called by a Home Assistant that does not know the feature.
SUPPORTED_FEATURES = (
    ai_task.AITaskEntityFeature.GENERATE_DATA
    | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
)
if hasattr(ai_task.AITaskEntityFeature, "GENERATE_IMAGE"):
    SUPPORTED_FEATURES |= ai_task.AITaskEntityFeature.GENERATE_IMAGE


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: MistralConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_AI_TASK_DATA:
            continue

        async_add_entities(
            [MistralTaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class MistralTaskEntity(ai_task.AITaskEntity, MistralBaseLLMEntity):
    """Mistral AI task entity."""

    _attr_supported_features = SUPPORTED_FEATURES

    def __init__(self, entry: MistralConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)

    async def _async_generate_image(
        self,
        task: ai_task.GenImageTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenImageTaskResult:
        """Handle a generate image task."""
        client = self.entry.runtime_data
        model = self.subentry.data.get(CONF_MODEL, DEFAULT_MODEL)

        # Built from the chat log rather than from task.instructions alone.
        # Home Assistant has already put the instructions into the log as a
        # user turn, carrying task.attachments with them, so converting the log
        # is what makes reference images reach the API -- naming the
        # instructions directly dropped them, while the entity went on
        # advertising SUPPORT_ATTACHMENTS.
        messages = await self._async_chat_log_messages(chat_log)

        # Not streamed. There is nothing to stream for an image, and going
        # through the ordinary completion call keeps this away from
        # _transform_stream, which is built around text deltas.
        try:
            response = await client.chat.complete_async(
                model=model,
                messages=messages,
                tools=[{"type": IMAGE_GENERATION_TOOL}],
                timeout_ms=TIMEOUT * 1000,
            )
        except SDKError as err:
            raise self._convert_error(err) from err
        except (TimeoutError, httpx.HTTPError) as err:
            raise HomeAssistantError(f"Error talking to Mistral AI: {err}") from err

        choices = getattr(response, "choices", None)
        message = choices[0].message if choices else None
        content = getattr(message, "content", None)

        # Record the assistant turn. Home Assistant adds the user side before
        # calling us, so leaving this out traced an image task as a question
        # with no answer, and the conversation debug view showed nothing at
        # all. Done before the checks below so a refusal is still recorded --
        # what the model said is the only clue as to why there is no image.
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id, content=_message_text(content)
            )
        )

        chunk = _find_generated_file(content)
        if chunk is None:
            raise HomeAssistantError("Mistral AI returned no image")

        try:
            downloaded = await client.files.download_async(
                file_id=chunk.file_id, timeout_ms=TIMEOUT * 1000
            )
        except SDKError as err:
            raise self._convert_error(err) from err
        except (TimeoutError, httpx.HTTPError) as err:
            raise HomeAssistantError(f"Error downloading image: {err}") from err

        image = downloaded.content
        if not image:
            raise HomeAssistantError("Mistral AI returned an empty image")

        # The download reports what it actually sent. Trusting file_type from
        # the chunk instead would mean guessing at a mime type for whatever
        # the model chose to produce.
        mime_type = downloaded.headers.get("content-type") or "image/png"

        return ai_task.GenImageTaskResult(
            conversation_id=chat_log.conversation_id,
            image_data=image,
            mime_type=mime_type.split(";")[0].strip(),
            model=model,
        )

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        await self._async_handle_chat_log(chat_log, task.name, task.structure)

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

        text = chat_log.content[-1].content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            _LOGGER.error("Failed to parse JSON response: %s. Response: %s", err, text)
            raise HomeAssistantError(
                "Error with Mistral AI structured response"
            ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )


def _message_text(content: Any) -> str | None:
    """Return the text a message carries, ignoring any file chunks.

    Content is a plain string for an ordinary reply, but image generation
    returns a list mixing whatever the model said with the file reference.
    """
    if content is None or isinstance(content, str):
        return content

    # isinstance rather than a truth test: a chunk carrying a non-string in
    # `text` would otherwise fail the join with a TypeError, which is a poor
    # way to report that an image came back in an unexpected shape.
    text = "".join(
        chunk_text
        for chunk in content
        if isinstance(chunk_text := getattr(chunk, "text", None), str)
    )
    return text or None


def _find_generated_file(content: Any) -> Any | None:
    """Return the tool file chunk in a message, if there is one.

    Image generation does not return the image. It returns a reference to a
    file the API is holding, as a chunk carrying a file_id, alongside whatever
    text the model produced. The bytes need a second call.
    """
    if isinstance(content, str) or content is None:
        return None

    for chunk in content:
        if getattr(chunk, "file_id", None):
            return chunk

    return None
