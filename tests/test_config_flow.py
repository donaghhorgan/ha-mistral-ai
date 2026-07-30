"""Tests for the Mistral AI Conversation config and subentry flows."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_conversation.const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_TEMPERATURE,
    DEFAULT_MODEL,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
)

from .helpers import make_sdk_error


async def test_user_flow_creates_entry_and_subentry(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    """A valid key creates the entry plus a default conversation agent."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "test-api-key"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_API_KEY: "test-api-key"}

    subentries = list(result["result"].subentries.values())
    assert len(subentries) == 1
    assert subentries[0].subentry_type == SUBENTRY_TYPE_CONVERSATION


@pytest.mark.parametrize(
    ("side_effect", "expected_error"),
    [
        (make_sdk_error(401), "invalid_auth"),
        (make_sdk_error(403), "invalid_auth"),
        (make_sdk_error(500), "cannot_connect"),
        (httpx.ConnectError("no route to host"), "cannot_connect"),
        (TimeoutError(), "cannot_connect"),
        (ValueError("something odd"), "unknown"),
    ],
)
async def test_user_flow_errors(
    hass: HomeAssistant,
    mock_client: MagicMock,
    side_effect: Exception,
    expected_error: str,
) -> None:
    """Each failure mode maps onto its own distinct form error."""
    mock_client.models.list_async = AsyncMock(side_effect=side_effect)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "bad-key"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": expected_error}


async def test_user_flow_recovers_after_error(
    hass: HomeAssistant, mock_client: MagicMock, mock_models_response: MagicMock
) -> None:
    """A corrected key after a failure still creates the entry."""
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(401))

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "bad-key"}
    )
    assert result["errors"] == {"base": "invalid_auth"}

    mock_client.models.list_async = AsyncMock(return_value=mock_models_response)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "good-key"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_API_KEY: "good-key"}


async def test_reauth_flow(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Reauth replaces the stored API key."""
    result = await init_integration.start_reauth_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "new-api-key"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert init_integration.data[CONF_API_KEY] == "new-api-key"


async def test_reauth_flow_rejects_bad_key(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Reauth keeps the old key when the new one is also rejected."""
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(401))

    result = await init_integration.start_reauth_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "still-bad"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_auth"}
    assert init_integration.data[CONF_API_KEY] == "test-api-key"


@pytest.mark.parametrize(
    "subentry_type", [SUBENTRY_TYPE_CONVERSATION, SUBENTRY_TYPE_AI_TASK_DATA]
)
async def test_add_subentry(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    subentry_type: str,
) -> None:
    """Both subentry types can be added, and their options are stored."""
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, subentry_type),
        context={"source": SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "set_options"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "My agent",
            CONF_MODEL: "mistral-large-latest",
            CONF_TEMPERATURE: 0.3,
        },
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "My agent"
    # The name is consumed as the title and must not leak into the options.
    assert CONF_NAME not in result["data"]
    assert result["data"][CONF_MODEL] == "mistral-large-latest"
    assert result["data"][CONF_TEMPERATURE] == 0.3


async def test_subentry_offers_models_from_api(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The model dropdown is populated from the API, not a hard-coded list."""
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )

    model_selector = result["data_schema"].schema[CONF_MODEL]
    options = model_selector.config["options"]

    assert DEFAULT_MODEL in options
    assert "mistral-large-latest" in options
    mock_client.models.list_async.assert_awaited()


async def test_reconfigure_subentry(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An existing subentry can be reconfigured in place."""
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": "reconfigure", "subentry_id": subentry.subentry_id},
    )
    assert result["type"] is FlowResultType.FORM

    # Reconfiguring must not ask for a name again.
    assert CONF_NAME not in result["data_schema"].schema

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {CONF_MODEL: "mistral-large-latest", CONF_TEMPERATURE: 1.5},
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"

    updated = init_integration.subentries[subentry.subentry_id]
    assert updated.data[CONF_MODEL] == "mistral-large-latest"
    assert updated.data[CONF_TEMPERATURE] == 1.5


async def test_subentry_aborts_when_entry_not_loaded(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Subentries cannot be added while the entry is unloaded."""
    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_not_loaded"


async def test_subentry_aborts_when_api_unreachable(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A failure listing models aborts the subentry flow with a reason."""
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(500))

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "cannot_connect"
