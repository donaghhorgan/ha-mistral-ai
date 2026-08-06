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
    CAPABILITY_FUNCTION_CALLING,
    CONF_MODEL,
    DEFAULT_MODEL,
    DOMAIN,
    IMAGE_GENERATION_TOOL,
    SUBENTRY_TYPE_AI_TASK_DATA,
    TIMEOUT,
)
from .entity import MistralBaseLLMEntity
from .helpers import entry_chunks, outputs_text

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
GENERATE_IMAGE_FEATURE = getattr(ai_task.AITaskEntityFeature, "GENERATE_IMAGE", None)

SUPPORTED_FEATURES = (
    ai_task.AITaskEntityFeature.GENERATE_DATA
    | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
)
if GENERATE_IMAGE_FEATURE is not None:
    SUPPORTED_FEATURES |= GENERATE_IMAGE_FEATURE


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

    async def async_added_to_hass(self) -> None:
        """Withdraw image generation if the model cannot run the connector.

        Image generation is not a separate endpoint here, it is a built-in
        connector passed as a tool, so a model that cannot call tools cannot
        produce an image no matter what is asked of it. Advertising the
        feature anyway meant the failure landed at automation run time as
        "Mistral AI returned no image".
        """
        await super().async_added_to_hass()

        if GENERATE_IMAGE_FEATURE is None:
            return

        if await self._async_model_calls_tools() is False:
            self._attr_supported_features = SUPPORTED_FEATURES & ~GENERATE_IMAGE_FEATURE

    async def _async_model_calls_tools(self) -> bool | None:
        """Return whether the configured model can call tools.

        None means the question could not be answered -- the lookup failed, or
        the API reported no capabilities at all. Callers treat that as "leave
        the feature alone": withdrawing a feature that works, because a
        listing call did not, is worse than the error it would prevent.
        """
        model = self.subentry.data.get(CONF_MODEL, DEFAULT_MODEL)
        try:
            card = await self.entry.runtime_data.models.retrieve_async(
                model_id=model, timeout_ms=TIMEOUT * 1000
            )
        except (SDKError, TimeoutError, httpx.HTTPError) as err:
            _LOGGER.debug("Could not read the capabilities of %s: %s", model, err)
            return None

        if (capabilities := getattr(card, "capabilities", None)) is None:
            return None

        return bool(getattr(capabilities, CAPABILITY_FUNCTION_CALLING, False))

    async def _async_conversation_inputs(
        self, chat_log: conversation.ChatLog
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Split a chat log into conversation instructions and input entries.

        The conversations endpoint has no system role -- MessageInputEntry is
        user or assistant only -- and takes the system prompt as `instructions`
        instead. Everything else carries over as it is, because the input chunk
        types are the same ones the chat log already converts to: text and
        image_url, which is what an attachment becomes.
        """
        messages = await self._async_chat_log_messages(chat_log)

        instructions = "\n\n".join(
            content
            for message in messages
            if message["role"] == "system"
            and isinstance(content := message["content"], str)
        )
        inputs = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message["role"] in ("user", "assistant")
        ]

        return instructions or None, inputs

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
        instructions, inputs = await self._async_conversation_inputs(chat_log)

        # The conversations endpoint rather than chat completions, because
        # connectors only run there. A chat completion response has nowhere to
        # put what they return: its content union has no ToolFileChunk, so the
        # file reference an image comes back as could never appear in one. The
        # endpoint accepted the request, never ran the connector, and returned
        # ordinary text -- which is what "Mistral AI returned no image" was.
        #
        # Still not streamed. There is nothing to stream for an image, and this
        # way the conversation event shape stays out of _transform_stream,
        # which is built around chat completion deltas.
        try:
            response = await client.beta.conversations.start_async(
                model=model,
                instructions=instructions,
                inputs=inputs,
                tools=[{"type": IMAGE_GENERATION_TOOL}],
                # Explicit because the endpoint defaults it to true, unlike
                # chat completions, which stores nothing. Left alone, every
                # image task would be retained on Mistral's servers and
                # listable afterwards -- a change in what leaves the house,
                # not an implementation detail.
                store=False,
                timeout_ms=TIMEOUT * 1000,
            )
        except SDKError as err:
            raise self._convert_error(err) from err
        except (TimeoutError, httpx.HTTPError) as err:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="api_error",
                translation_placeholders={"error": str(err)},
            ) from err

        outputs = getattr(response, "outputs", None) or []

        # Record the assistant turn. Home Assistant adds the user side before
        # calling us, so leaving this out traced an image task as a question
        # with no answer, and the conversation debug view showed nothing at
        # all. Done before the checks below so a refusal is still recorded --
        # what the model said is the only clue as to why there is no image.
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id, content=outputs_text(outputs)
            )
        )

        chunk = _find_generated_file(outputs)
        if chunk is None:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="no_image_returned",
                translation_placeholders={"model": model},
            )

        try:
            downloaded = await client.files.download_async(
                file_id=chunk.file_id, timeout_ms=TIMEOUT * 1000
            )
        except SDKError as err:
            raise self._convert_error(err) from err
        except (TimeoutError, httpx.HTTPError) as err:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="image_download_error",
                translation_placeholders={"error": str(err)},
            ) from err

        image = downloaded.content

        # The bytes are ours now, and Home Assistant saves its own copy to the
        # media source, so the remote file has no further purpose. Nothing
        # else deletes it: `store: false` governs the conversation, not the
        # file the connector wrote, so without this every image an automation
        # ever generates stays on the account. An hourly dashboard would leave
        # 8,760 of them a year, readable and deletable by anyone holding the
        # API key.
        #
        # Deleted before the empty check below, so a zero-byte result is still
        # cleaned up rather than left behind by the error path.
        await self._async_delete_file(chunk.file_id)

        if not image:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="empty_image",
            )

        return ai_task.GenImageTaskResult(
            conversation_id=chat_log.conversation_id,
            image_data=image,
            mime_type=_image_mime_type(image, getattr(chunk, "file_type", None)),
            model=model,
        )

    async def _async_delete_file(self, file_id: str) -> None:
        """Remove a generated file from the account, best effort.

        Never raises. By the time this runs the caller holds the image and the
        task has effectively succeeded, so turning a failed cleanup into a
        failed generation would be the wrong trade -- the user would see an
        error for something they actually got.

        Logged at debug rather than warning for the same reason: a file left
        behind costs a little storage on an account the user controls, and it
        is not a thing they can act on from a notification.
        """
        try:
            await self.entry.runtime_data.files.delete_async(
                file_id=file_id, timeout_ms=TIMEOUT * 1000
            )
        except (SDKError, TimeoutError, httpx.HTTPError) as err:
            _LOGGER.debug("Could not delete generated file %s: %s", file_id, err)
        except Exception:  # noqa: BLE001
            # Deliberately broad. The SDK has exception types that inherit from
            # none of the above, and nothing here is worth failing a completed
            # image generation over.
            _LOGGER.debug("Could not delete generated file %s", file_id)

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
                translation_domain=DOMAIN,
                translation_key="invalid_structured_response",
            ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )


# Enough of each header to identify it. Home Assistant serves the image with
# whatever mime type we report, so getting it wrong is a picture that will not
# display rather than an error anyone can trace.
_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _image_mime_type(image: bytes, file_type: Any) -> str:
    """Return the mime type of downloaded image bytes.

    Read from the bytes rather than from either thing that claims to describe
    them, because both were checked against the live API and neither survived:

    - the download responds `application/octet-stream`, so its content-type
      header says nothing at all
    - the chunk's `file_type` said `png` for a file the connector itself
      reported at a URL ending `.jpg`

    So `file_type` is a fallback for a format not listed here, not the answer.
    """
    for signature, mime_type in _IMAGE_SIGNATURES:
        if image.startswith(signature):
            return mime_type

    # WebP carries its marker after the RIFF length, not at the start.
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"

    if isinstance(file_type, str) and file_type:
        return f"image/{file_type.lower()}"

    return "image/png"


def _find_generated_file(outputs: Any) -> Any | None:
    """Return the tool file chunk in a response, if there is one.

    Image generation does not return the image. It returns a reference to a
    file the API is holding, as a chunk carrying a file_id, alongside whatever
    text the model produced. The bytes need a second call.
    """
    for chunk in entry_chunks(outputs):
        if getattr(chunk, "file_id", None):
            return chunk

    return None
