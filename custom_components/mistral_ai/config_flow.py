"""Config flow for the Mistral AI integration."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx
import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from mistralai.client.errors import SDKError

from .client import async_create_client
from .const import (
    CAPABILITY_AUDIO_SPEECH,
    CAPABILITY_AUDIO_TRANSCRIPTION,
    CAPABILITY_COMPLETION_CHAT,
    CONF_API_KEY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VOICE,
    CONF_WEB_SEARCH,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_STT_NAME,
    DEFAULT_STT_TEMPERATURE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TTS_NAME,
    DOMAIN,
    MAX_TEMPERATURE,
    MAX_TOP_P,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_STT,
    SUBENTRY_TYPE_TTS,
    TIMEOUT,
    WEB_SEARCH_TOOLS,
)
from .helpers import async_list_voices

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from mistralai.client import Mistral

_LOGGER = logging.getLogger(__name__)

# Applied to conversation subentries when they are created, matching the
# official Home Assistant LLM integrations -- openai_conversation, anthropic
# and google_generative_ai_conversation all enable the Assist API on a new
# conversation agent. Ollama is the exception, and local models being poor at
# tool calling is the reason, which does not apply here.
#
# Leaving it unset produces an agent that answers questions and cannot control
# anything. Worse, it does not advertise ConversationEntityFeature.CONTROL, so
# Home Assistant will not offer it where control is required -- which reads as
# the integration being broken rather than as a setting being off.
# Which model capability each subentry type needs. Asked of the API rather
# than written down, so the lists stay right as Mistral ships and retires
# models -- a key can reach transcription, OCR, embedding and coding models
# that have no business in a conversation dropdown.
SUBENTRY_CAPABILITIES = {
    SUBENTRY_TYPE_AI_TASK_DATA: CAPABILITY_COMPLETION_CHAT,
    SUBENTRY_TYPE_CONVERSATION: CAPABILITY_COMPLETION_CHAT,
    SUBENTRY_TYPE_STT: CAPABILITY_AUDIO_TRANSCRIPTION,
    SUBENTRY_TYPE_TTS: CAPABILITY_AUDIO_SPEECH,
}

DEFAULT_SUBENTRY_NAMES = {
    SUBENTRY_TYPE_AI_TASK_DATA: DEFAULT_AI_TASK_NAME,
    SUBENTRY_TYPE_CONVERSATION: DEFAULT_CONVERSATION_NAME,
    SUBENTRY_TYPE_STT: DEFAULT_STT_NAME,
    SUBENTRY_TYPE_TTS: DEFAULT_TTS_NAME,
}

RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD, autocomplete="off")
        ),
    }
)


class ModelChoice(NamedTuple):
    """A model the key can reach, and what the API says about its future."""

    id: str
    deprecation: datetime | None
    replacement: str | None

    @property
    def label(self) -> str:
        """Return the text to show in a dropdown.

        A deprecated model still works until its date, so it stays selectable
        and says so rather than disappearing. Someone with a reason to pick one
        can; someone choosing blind is told what they are choosing.
        """
        if self.deprecation is None:
            return self.id

        retires = self.deprecation.date().isoformat()
        if self.replacement:
            return f"{self.id} — retires {retires}, replaced by {self.replacement}"
        return f"{self.id} — retires {retires}"


async def async_list_models(
    client: Mistral, *, capability: str | None = None
) -> list[ModelChoice]:
    """Return the models available to the given client.

    `capability` filters to models advertising it, e.g. audio_transcription.
    Filtering on what the API reports rather than on model names keeps the
    speech platforms working across a Mistral release: names come and go, the
    capability flags do not.

    Deprecation is carried through rather than discarded. The API reports a
    retirement date and a replacement for every model on its way out, and the
    dropdown used to show those exactly like any other -- so a model could be
    chosen weeks before it stopped working, with nothing said at any point.

    Raises InvalidAuth if the key is rejected, Forbidden if it is accepted but
    the account is not permitted, and CannotConnect for any other failure, so
    callers can map each onto a form error.
    """
    try:
        async with asyncio.timeout(TIMEOUT):
            response = await client.models.list_async()
    except SDKError as err:
        if err.status_code == 401:
            raise InvalidAuth from err
        if err.status_code == 403:
            raise Forbidden(str(err)) from err
        raise CannotConnect(str(err)) from err
    except (TimeoutError, httpx.HTTPError) as err:
        raise CannotConnect(str(err)) from err

    models = [model for model in (response.data or []) if getattr(model, "id", None)]
    if capability is not None:
        models = [
            model
            for model in models
            if getattr(getattr(model, "capabilities", None), capability, False)
        ]

    choices = [
        ModelChoice(
            id=model.id,
            deprecation=getattr(model, "deprecation", None),
            replacement=getattr(model, "deprecation_replacement_model", None),
        )
        for model in models
    ]

    # Live models first, then the ones on their way out, each group by name.
    # Sorting purely by name buries a deprecation warning among fifty entries
    # that do not have one.
    return sorted(
        choices, key=lambda choice: (choice.deprecation is not None, choice.id)
    )


class MistralConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Mistral AI."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                client = await async_create_client(self.hass, user_input[CONF_API_KEY])
                await async_list_models(client)
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except Forbidden:
                _LOGGER.exception("Mistral AI refused the request")
                errors["base"] = "forbidden"
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(
                    title="Mistral AI",
                    data=user_input,
                    subentries=[
                        {
                            "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                            "data": dict(RECOMMENDED_CONVERSATION_OPTIONS),
                            "title": DEFAULT_CONVERSATION_NAME,
                            "unique_id": None,
                        }
                    ],
                )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Handle re-authentication when the API key stops working."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm re-authentication with a new API key."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                client = await async_create_client(self.hass, user_input[CONF_API_KEY])
                await async_list_models(client)
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except Forbidden:
                _LOGGER.exception("Mistral AI refused the request")
                errors["base"] = "forbidden"
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_update_reload_and_abort(
                    self._get_reauth_entry(), data_updates=user_input
                )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return the subentry types supported by this integration."""
        return {
            SUBENTRY_TYPE_CONVERSATION: MistralSubentryFlowHandler,
            SUBENTRY_TYPE_AI_TASK_DATA: MistralSubentryFlowHandler,
            SUBENTRY_TYPE_STT: MistralSubentryFlowHandler,
            SUBENTRY_TYPE_TTS: MistralSubentryFlowHandler,
        }


class MistralSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing Mistral AI subentries."""

    @property
    def _is_new(self) -> bool:
        """Return whether this is a new subentry rather than a reconfigure."""
        return self.source == "user"

    async def async_step_set_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure a conversation agent or AI task."""
        entry = self._get_entry()
        if entry.state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            if self._is_new:
                title = user_input.pop(CONF_NAME)
                return self.async_create_entry(title=title, data=user_input)
            return self.async_update_and_abort(
                entry, self._get_reconfigure_subentry(), data=user_input
            )

        # The entry is loaded, so it already holds a client. Building another
        # one here would mean a fresh connection pool for every render of this
        # form, and this form is rendered often.
        client = entry.runtime_data

        try:
            models = await async_list_models(
                client,
                capability=SUBENTRY_CAPABILITIES.get(self._subentry_type),
            )
        except InvalidAuth:
            return self.async_abort(reason="invalid_auth")
        except Forbidden:
            _LOGGER.exception("Mistral AI refused the request")
            return self.async_abort(reason="forbidden")
        except CannotConnect:
            _LOGGER.exception("Failed to list Mistral AI models")
            return self.async_abort(reason="cannot_connect")

        if not self._is_new:
            options = dict(self._get_reconfigure_subentry().data)
        elif self._subentry_type == SUBENTRY_TYPE_AI_TASK_DATA:
            # AI tasks generate data; they are never given Home Assistant
            # control, so there is nothing to seed.
            options = {}
        else:
            options = dict(RECOMMENDED_CONVERSATION_OPTIONS)

        # A selected API can disappear -- whatever provided it gets removed --
        # and the stale id would otherwise be shown as selected, saved back,
        # and then fail every single message, because async_provide_llm_data
        # resolves it and raises on an id it does not know. Both
        # openai_conversation and ollama filter here for the same reason.
        if selected := options.get(CONF_LLM_HASS_API):
            if isinstance(selected, str):
                selected = [selected]
            valid = {api.id for api in llm.async_get_apis(self.hass)}
            options[CONF_LLM_HASS_API] = [api for api in selected if api in valid]

        voices: list[SelectOptionDict] = []
        if self._subentry_type == SUBENTRY_TYPE_TTS:
            voices = [
                SelectOptionDict(label=voice.name, value=voice.voice_id)
                for voice in await async_list_voices(client)
            ]

            # An empty list means the listing failed, not that there is
            # nothing to choose from: preset voices exist on every workspace,
            # including a brand new empty one, and async_list_voices returns
            # an empty list for any error rather than raising.
            #
            # Aborting is the point. The form used to be shown anyway, minus
            # the voice field, and produced an entity that could never speak --
            # the endpoint refuses a request with no voice, and the 400 comes
            # back as silence. Better to say so now than to hand someone a
            # working-looking entity that is not.
            if not voices:
                return self.async_abort(reason="no_voices")

        return self.async_show_form(
            step_id="set_options",
            data_schema=vol.Schema(
                _subentry_schema(
                    self.hass,
                    is_new=self._is_new,
                    subentry_type=self._subentry_type,
                    options=options,
                    models=models,
                    voices=voices,
                )
            ),
        )

    async_step_user = async_step_set_options
    async_step_reconfigure = async_step_set_options


def _subentry_schema(
    hass: HomeAssistant,
    *,
    is_new: bool,
    subentry_type: str,
    options: Mapping[str, Any],
    models: list[ModelChoice],
    voices: list[SelectOptionDict] | None = None,
) -> dict[vol.Marker, Any]:
    """Build the schema for a subentry options form."""
    schema: dict[vol.Marker, Any] = {}

    if is_new:
        default_name = DEFAULT_SUBENTRY_NAMES.get(
            subentry_type, DEFAULT_CONVERSATION_NAME
        )
        schema[vol.Required(CONF_NAME, default=default_name)] = TextSelector()

    # Prefer the configured model, then DEFAULT_MODEL where the API still
    # offers it, then whatever the API did offer. Naming a model per platform
    # here would go stale the next time Mistral retires one; this only names
    # the one general-purpose default and treats it as a preference rather
    # than a guarantee.
    listed = {choice.id: choice for choice in models}

    if (fallback_model := DEFAULT_MODEL) not in listed:
        fallback_model = models[0].id if models else DEFAULT_MODEL
    default_model = options.get(CONF_MODEL, fallback_model)

    # Keep the configured model selectable even if the API stops listing it,
    # so reconfiguring does not silently drop the user's choice. A model that
    # has been retired outright is exactly when someone needs to open this
    # form, so it must not vanish from it.
    model_options = [
        SelectOptionDict(label=choice.label, value=choice.id) for choice in models
    ]
    if default_model not in listed:
        model_options.insert(
            0, SelectOptionDict(label=default_model, value=default_model)
        )

    schema[vol.Required(CONF_MODEL, default=default_model)] = SelectSelector(
        SelectSelectorConfig(
            options=model_options,
            mode=SelectSelectorMode.DROPDOWN,
            custom_value=True,
        )
    )

    # Not offered for text-to-speech, because the speech endpoint takes no such
    # argument -- `audio.speech.complete_async` has no temperature parameter.
    # Offering it stored a setting that could never do anything.
    #
    # Transcription does take one, and gets a lower default: see
    # DEFAULT_STT_TEMPERATURE.
    if subentry_type != SUBENTRY_TYPE_TTS:
        default_temperature = (
            DEFAULT_STT_TEMPERATURE
            if subentry_type == SUBENTRY_TYPE_STT
            else DEFAULT_TEMPERATURE
        )
        schema[
            vol.Optional(
                CONF_TEMPERATURE,
                default=options.get(CONF_TEMPERATURE, default_temperature),
            )
        ] = NumberSelector(
            NumberSelectorConfig(
                min=0.0,
                max=MAX_TEMPERATURE[subentry_type],
                step=0.1,
                mode=NumberSelectorMode.SLIDER,
            )
        )

    # Not offered for the speech platforms: neither the speech endpoint nor
    # the transcription one takes it, so a stored value could do nothing.
    if subentry_type in (SUBENTRY_TYPE_AI_TASK_DATA, SUBENTRY_TYPE_CONVERSATION):
        schema[
            vol.Optional(
                CONF_TOP_P,
                default=options.get(CONF_TOP_P, DEFAULT_TOP_P),
            )
        ] = NumberSelector(
            NumberSelectorConfig(
                min=0.0, max=MAX_TOP_P, step=0.05, mode=NumberSelectorMode.SLIDER
            )
        )

    # Required, because the endpoint will not choose one:
    #
    #   400 Either ref_audio or voice must be provided.
    #
    # This used to be optional and was left out entirely when no voices could
    # be listed, on the belief that the API would pick one. It does not. An
    # entity configured that way stored no voice, sent no voice, and answered
    # every request with a 400 that async_get_tts_audio turns into silence.
    #
    # `is not None` rather than a truthiness test, and the distinction is the
    # bug. `and voices` silently dropped the field for an empty list, which is
    # how an entity with no voice came to exist. An empty list cannot reach
    # here -- the flow aborts on it before building the form -- so this only
    # narrows the argument's type, and an empty dropdown would be a visible
    # fault rather than a missing field.
    if subentry_type == SUBENTRY_TYPE_TTS and voices is not None:
        schema[
            vol.Required(
                CONF_VOICE,
                description={"suggested_value": options.get(CONF_VOICE)},
            )
        ] = SelectSelector(
            SelectSelectorConfig(options=voices, mode=SelectSelectorMode.DROPDOWN)
        )

    # A response length makes no sense for speech: the length is whatever was
    # said, or whatever was asked to be read out.
    if subentry_type not in (SUBENTRY_TYPE_STT, SUBENTRY_TYPE_TTS):
        schema[
            vol.Optional(
                CONF_MAX_TOKENS,
                default=options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            )
        ] = NumberSelector(
            NumberSelectorConfig(min=1, max=32000, step=1, mode=NumberSelectorMode.BOX)
        )

    # Prompts and Home Assistant control only apply to conversation agents.
    if subentry_type == SUBENTRY_TYPE_CONVERSATION:
        apis: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(hass)
        ]
        schema.update(
            {
                vol.Optional(
                    CONF_PROMPT,
                    description={
                        "suggested_value": options.get(
                            CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                        )
                    },
                ): TemplateSelector(),
                vol.Optional(
                    CONF_LLM_HASS_API,
                    description={"suggested_value": options.get(CONF_LLM_HASS_API)},
                ): SelectSelector(SelectSelectorConfig(options=apis, multiple=True)),
                # Left unset means off. The tier is named rather than toggled
                # because the two bill differently and Mistral charges per
                # search, so the expensive one should never be picked on
                # someone's behalf.
                vol.Optional(
                    CONF_WEB_SEARCH,
                    description={"suggested_value": options.get(CONF_WEB_SEARCH)},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=list(WEB_SEARCH_TOOLS),
                        mode=SelectSelectorMode.DROPDOWN,
                        translation_key=CONF_WEB_SEARCH,
                    )
                ),
            }
        )

    return schema


class CannotConnect(Exception):
    """Error to indicate we cannot connect."""


class InvalidAuth(Exception):
    """Error to indicate the API key was rejected."""


class Forbidden(Exception):
    """Error to indicate the key is valid but the account is not permitted.

    Kept apart from InvalidAuth because the remedy is different: there is no
    key the user can type that resolves it, so a form that says "invalid key"
    and offers the field again sends them in a circle.
    """
