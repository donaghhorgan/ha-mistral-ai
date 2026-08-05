"""Config flow for the Mistral AI Conversation integration."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING, Any

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
from mistralai.client import Mistral
from mistralai.client.errors import SDKError

from .const import (
    CONF_API_KEY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
    TIMEOUT,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

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


async def async_list_models(hass: HomeAssistant, api_key: str) -> list[str]:
    """Return the models available to the given API key.

    Raises InvalidAuth if the key is rejected and CannotConnect for any other
    failure, so callers can map both onto a form error.
    """
    client = await hass.async_add_executor_job(partial(Mistral, api_key=api_key))
    try:
        async with asyncio.timeout(TIMEOUT):
            response = await client.models.list_async()
    except SDKError as err:
        if err.status_code in (401, 403):
            raise InvalidAuth from err
        raise CannotConnect(str(err)) from err
    except (TimeoutError, httpx.HTTPError) as err:
        raise CannotConnect(str(err)) from err

    return sorted(
        model.id for model in (response.data or []) if getattr(model, "id", None)
    )


class MistralConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Mistral AI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                await async_list_models(self.hass, user_input[CONF_API_KEY])
            except InvalidAuth:
                errors["base"] = "invalid_auth"
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

    async def async_step_reauth(self, entry_data: dict[str, Any]) -> ConfigFlowResult:
        """Handle re-authentication when the API key stops working."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm re-authentication with a new API key."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                await async_list_models(self.hass, user_input[CONF_API_KEY])
            except InvalidAuth:
                errors["base"] = "invalid_auth"
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

        try:
            models = await async_list_models(self.hass, entry.data[CONF_API_KEY])
        except InvalidAuth:
            return self.async_abort(reason="invalid_auth")
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

        return self.async_show_form(
            step_id="set_options",
            data_schema=vol.Schema(
                _subentry_schema(
                    self.hass,
                    is_new=self._is_new,
                    subentry_type=self._subentry_type,
                    options=options,
                    models=models,
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
    models: list[str],
) -> dict[vol.Marker, Any]:
    """Build the schema for a subentry options form."""
    schema: dict[vol.Marker, Any] = {}

    if is_new:
        default_name = (
            DEFAULT_AI_TASK_NAME
            if subentry_type == SUBENTRY_TYPE_AI_TASK_DATA
            else DEFAULT_CONVERSATION_NAME
        )
        schema[vol.Required(CONF_NAME, default=default_name)] = TextSelector()

    # Keep the configured model selectable even if the API stops listing it,
    # so reconfiguring does not silently drop the user's choice.
    model_options = sorted({*models, options.get(CONF_MODEL, DEFAULT_MODEL)})

    schema.update(
        {
            vol.Required(
                CONF_MODEL, default=options.get(CONF_MODEL, DEFAULT_MODEL)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=model_options,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            ),
            vol.Optional(
                CONF_TEMPERATURE,
                default=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.0, max=2.0, step=0.1, mode=NumberSelectorMode.SLIDER
                )
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                default=options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=1, max=32000, step=1, mode=NumberSelectorMode.BOX
                )
            ),
        }
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
            }
        )

    return schema


class CannotConnect(Exception):
    """Error to indicate we cannot connect."""


class InvalidAuth(Exception):
    """Error to indicate the API key was rejected."""
