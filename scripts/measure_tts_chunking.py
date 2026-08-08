#!/usr/bin/env python3
"""Measure what one speech request per sentence costs, and what it buys.

`_sentences` in `custom_components/mistral_ai/tts.py` issues a speech request
per sentence, so audio starts while the conversation agent is still writing.
The shape was chosen because it was the obvious one and was never measured
against the alternatives, which is [#160].

Worth stating plainly, because the code is easy to misread as helping the
speech model: it does the opposite. The model handles sentence structure,
abbreviations and prosody better than any regex, and given the whole reply it
would get all of them right. Splitting is what denies it that context --
request two arrives as a standalone "Seuss. His work includes..." with nothing
saying that "Dr." came before it. The split buys one thing, the first audio
arriving early, so every boundary is a cost paid for latency and the question
is how many boundaries are worth it.

Two kinds of number come out of this, and they are not equally trustworthy.

*Deterministic*, and needing no API key (`--offline`): how many requests each
strategy issues, where it puts the boundaries, and which boundaries the
abbreviation lookbehind actually moves. Same answer every run, so one run is
the whole answer.

*Measured*, at one billed request per chunk per trial: time to first audio
byte, wall clock to the last byte, and how much audio came back. Network and
server load are not controllable from here, so each figure is a median over
`--trials` runs with the spread printed beside it, and a difference inside
that spread is not a finding.

Three measurement habits are built in rather than left to the caller, because
this project has already published two numbers that were wrong for want of
them:

- **A fresh baseline per cell.** The clock starts immediately before the first
  request of each (case, strategy, trial), never once for the whole run. A
  global baseline is how one case's result was previously attributed to all of
  them.
- **The streamed path.** Requests go out with `stream: true` and the audio is
  read out of the server-sent events, which is the code path `tts.py` uses.
  Timing the buffered endpoint and generalising to the streamed one is the
  other mistake already made here; the two differ by roughly the whole benefit
  being measured.
- **Rotated cell order.** Every trial visits the cells in a different rotation,
  so a network that gets slower during the run spreads that across strategies
  instead of landing on whichever one always went last.

What it still cannot control: other traffic on the account, server-side load,
and the route to the API. Run it more than once before believing a small
difference.

**The live half of this script has never been executed.** It was written
against a rotated key, and every request it made returned 401, so only the
`--offline` half has ever produced output. The request-building code mirrors
`tts.py` and the event parsing mirrors `_speech_events`, but mirroring is a
claim and not a result -- the first person to run this with a working key
should expect to fix something, and should treat its first numbers with the
suspicion any unrun code deserves. Delete this paragraph once it has run.

Usage:

    # Deterministic half only. No key, no requests, no cost.
    python scripts/measure_tts_chunking.py --offline

    # The real thing. Needs MISTRAL_API_KEY and a voice from the account.
    python scripts/measure_tts_chunking.py --voice VOICE_ID

    # One case, more trials, for a difference that looks marginal.
    python scripts/measure_tts_chunking.py --voice VOICE_ID --case long --trials 9

[#160]: https://github.com/donaghhorgan/ha-mistral-ai/issues/160
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

# The integration is imported rather than copied, so that "the current split"
# measures what actually ships. A copy would drift, and a measurement of a
# stale copy reads exactly like a measurement of the real thing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from custom_components.mistral_ai.const import (  # noqa: E402
    TTS_AUDIO_FORMAT,
)
from custom_components.mistral_ai.tts import (  # noqa: E402
    MIN_SPEECH_CHARS,
    _sentences,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator, Sequence

SPEECH_URL = "https://api.mistral.ai/v1/audio/speech"

# The speech model the contract tests use. Named here rather than read from a
# config entry because this script has no Home Assistant to read one from.
TTS_MODEL = "voxtral-mini-tts-latest"

# How the incoming text is handed to the splitters. Home Assistant streams the
# reply as the model writes it, which is by token and therefore mid-word, and
# a splitter given the whole string at once would not be exercised the way it
# is in production. Twelve characters is roughly a token.
STREAM_PIECE = 12

# Boundary patterns to compare against the shipped one.
#
# The shipped split is no longer a pattern: sentence boundaries come from
# `sentence_stream`, the library `elevenlabs` uses, so there is nothing here
# to point a regex at. SENTENCE_END stands in for it -- a terminator followed
# by whitespace, which is what this integration used before -- and its rows
# measure what the old rule would have done. NAIVE_END drops the abbreviation
# lookbehind that rule carried, so the gap between those two rows is what the
# lookbehind bought when it existed.
#
# Neither can split Chinese or Japanese, where "。" is not followed by
# whitespace. That is the gap the library closed, and it does not show up in
# these cases because they are all English.
SENTENCE_END = re.compile(r"(?<=[.!?])(?<!\.\w\.)\s+")
NAIVE_END = re.compile(r"(?<=[.!?])\s+")

# Paragraphs only: a blank line. The coarsest split that still starts audio
# before the reply is finished.
PARAGRAPH_END = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Strategy:
    """One way of deciding where speech requests begin and end."""

    name: str
    # None means no split at all -- one request for the whole reply, which is
    # the latency baseline everything else is paid for.
    pattern: re.Pattern[str] | None
    floor: int
    note: str


# `shipped` is the real thing: its chunks come from calling `_sentences`, not
# from re-deriving them here. Everything else is a comparison built from a
# pattern, which is why only this one can be trusted to describe what runs in
# production.
SHIPPED = "shipped"

STRATEGIES: tuple[Strategy, ...] = (
    Strategy(SHIPPED, None, MIN_SPEECH_CHARS, "sentence_stream, what ships today"),
    Strategy("whole", None, 0, "one request for the whole reply"),
    Strategy("sentences-0", SENTENCE_END, 0, "every boundary, no minimum length"),
    Strategy("sentences-40", SENTENCE_END, MIN_SPEECH_CHARS, "the pre-library split"),
    Strategy(
        "naive-40",
        NAIVE_END,
        MIN_SPEECH_CHARS,
        "the shipped split without the abbreviation lookbehind",
    ),
    Strategy("sentences-120", SENTENCE_END, 120, "sentences grouped to ~a clause set"),
    Strategy("sentences-300", SENTENCE_END, 300, "sentences grouped to ~a paragraph"),
    Strategy("paragraphs", PARAGRAPH_END, 0, "split on blank lines only"),
)


# Realistic replies, not lorem ipsum. Three lengths, because the trade changes
# with length -- a two-sentence answer has nothing to overlap, and a long one
# has the most to gain -- plus one carrying the abbreviations the lookbehind
# exists for.
CASES: dict[str, str] = {
    "short": (
        "The kitchen light is on and the hall light is off. "
        "The thermostat is set to 20.5 degrees."
    ),
    "medium": (
        "There are four lights on downstairs at the moment. The kitchen and "
        "the hall are both at full brightness, the porch light is dimmed to "
        "thirty percent, and the lamp in the living room is on its warm "
        "setting. Upstairs everything is off except the landing light. The "
        "back door has been unlocked since about four o'clock, and the "
        "washing machine finished its cycle twenty minutes ago."
    ),
    "abbreviations": (
        "Theodor Seuss Geisel was an American children's author who published "
        "under the pen name Dr. Seuss. His work includes more than sixty "
        "books, several of which, e.g. The Cat in the Hat and Green Eggs and "
        "Ham, are among the best selling children's books ever printed. He "
        "worked as an illustrator for advertising campaigns in the U.S. "
        "before turning to books full time, and he continued to draw "
        "political cartoons during the war. His books have since been "
        "translated into dozens of languages, i.e. far beyond the audience he "
        "originally wrote them for."
    ),
    "long": (
        "Here is what the house looks like right now. Downstairs, the kitchen "
        "light and the hall light are both on, the porch light is dimmed to "
        "thirty percent, and the living room lamp is on its warm setting. The "
        "back door has been unlocked since about four o'clock this afternoon, "
        "which is longer than usual, and the washing machine finished its "
        "cycle twenty minutes ago.\n\n"
        "Upstairs is quieter. Everything is off except the landing light, "
        "which is on its motion schedule until eleven. The bedroom window "
        "sensor has been reporting open since lunchtime, and the temperature "
        "in that room has fallen to 17.5 degrees as a result. The bathroom "
        "extractor fan ran for eight minutes this morning and has not run "
        "since.\n\n"
        "Nothing needs your attention urgently. If you want, I can close the "
        "bedroom window schedule, lock the back door, and turn the downstairs "
        "lights down to their evening scene. I can also set a reminder for "
        "the washing, since it has been sitting in the drum for a while now "
        "and the forecast for tomorrow is wet."
    ),
}


def regroup(text: str, strategy: Strategy, piece: int = STREAM_PIECE) -> list[str]:
    """Return the chunks a strategy would issue requests for.

    Comparisons only. The shipped split is `sentence_stream` and is measured
    by calling `_sentences` directly -- see SHIPPED above -- because a
    reimplementation of a library is a reimplementation that will drift.
    What this builds is the alternatives: a pattern plus a floor.

    Fed in `piece`-sized fragments for the same reason the real one is: the
    stream arrives mid-word, and a splitter handed the whole string never sees
    a boundary arrive in two halves.
    """
    if strategy.pattern is None:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    buffer = ""

    for start in range(0, len(text), piece):
        buffer += text[start : start + piece]

        while True:
            boundary = next(
                (
                    match
                    for match in strategy.pattern.finditer(buffer)
                    if match.start() >= strategy.floor
                ),
                None,
            )
            if boundary is None:
                break

            chunks.append(buffer[: boundary.start()].strip())
            buffer = buffer[boundary.end() :]

    if final := buffer.strip():
        chunks.append(final)

    return chunks


async def _pieces(text: str, piece: int = STREAM_PIECE) -> AsyncGenerator[str]:
    """Yield text the way Home Assistant streams a reply: in fragments."""
    for start in range(0, len(text), piece):
        yield text[start : start + piece]


async def shipped_chunks(text: str) -> list[str]:
    """Return the chunks the integration itself would request, for real.

    Calls `_sentences`, so the "shipped" row cannot drift from `tts.py` the
    way a reimplementation would. The regex strategies beside it are
    deliberately *not* the shipped split -- they are what it is being
    compared against.
    """
    return [chunk async for chunk in _sentences(_pieces(text))]


@dataclass
class Trial:
    """One timed run of one strategy over one case."""

    ttfb: float
    total: float
    audio_bytes: int
    audio_seconds: float | None


@dataclass
class Cell:
    """Every trial for one (case, strategy) pair."""

    case: str
    strategy: Strategy
    chunks: list[str]
    trials: list[Trial] = field(default_factory=list)

    @property
    def requests(self) -> int:
        """Return the number of billed speech calls one reply costs."""
        return len(self.chunks)


def _speech_payload(text: str, voice: str, model: str) -> dict[str, object]:
    """Return the request body, matching what `tts.py` sends."""
    return {
        "input": text,
        "model": model,
        "response_format": TTS_AUDIO_FORMAT,
        "voice_id": voice,
        "stream": True,
    }


async def _audio_from_stream(response: httpx.Response) -> AsyncGenerator[bytes]:
    """Yield decoded audio from a server-sent event stream.

    Deliberately the same shape as `_speech_events` in `tts.py`: events are
    framed by blank lines and recognised by carrying `audio_data`, rather than
    by their event name. A parser that read the stream differently from the
    integration would be timing a different thing.
    """
    data: list[str] = []

    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
            continue
        if line.strip():
            continue
        if data:
            payload, data = "\n".join(data), []
            if chunk := _audio_from_payload(payload):
                yield chunk

    if data and (chunk := _audio_from_payload("\n".join(data))):
        yield chunk


def _audio_from_payload(payload: str) -> bytes:
    """Return the audio one event carries, or empty bytes if it carries none."""
    try:
        event = json.loads(payload)
    except ValueError:
        return b""
    if not isinstance(event, dict):
        return b""
    encoded = event.get("audio_data")
    return base64.b64decode(encoded, validate=True) if encoded else b""


def _mp3_seconds(replies: Sequence[bytes]) -> float | None:
    """Return the total duration of the audio, or None if it cannot be read.

    One entry per *request*, each already joined from its stream deltas. The
    distinction matters: a delta is not a file. Only the first of a reply's
    deltas carries the header, so handing this the deltas individually makes
    every reply past a second or so unreadable, and the whole measurement
    comes back empty.

    Each reply is then measured separately and the durations summed, rather
    than concatenating first and measuring once. Concatenated MP3s carry one
    Xing header per part and a reader trusts the first, so measuring the join
    would report the first reply's length for the whole thing -- which would
    look like the split losing audio when nothing had been lost.

    Summing is also the right comparison for the question being asked: whether
    more boundaries change how much audio a reply produces.
    """
    try:
        from mutagen.mp3 import MP3
    except ImportError:
        return None

    total = 0.0
    for reply in replies:
        try:
            info = MP3(io.BytesIO(reply)).info
        except Exception:  # noqa: BLE001  # mutagen raises its own error tree
            return None
        # `info` is typed as optional, and a file mutagen cannot parse is
        # exactly the case worth reporting as unknown rather than as zero
        # seconds -- zero would read as audio having gone missing.
        length = getattr(info, "length", None)
        if not isinstance(length, (int, float)):
            return None
        total += float(length)
    return total


async def run_trial(
    http: httpx.AsyncClient, cell: Cell, voice: str, model: str
) -> Trial:
    """Time one strategy over one case, issuing its requests in order.

    Sequential, because that is what `async_stream_tts_audio` does: it consumes
    one speech stream to exhaustion before asking for the next. Issuing them
    concurrently would measure a design nothing implements.

    The clock starts here, immediately before the first request of *this* cell.

    Audio is collected one list per request rather than one flat list, because
    that is the unit `_mp3_seconds` can measure. A request's stream arrives as
    several `speech.audio.delta` events, and only the first carries the file
    header -- the rest are frame fragments that no reader can open alone. A
    flat list therefore made every multi-delta reply unmeasurable, which is
    every reply long enough to be interesting.
    """
    replies: list[list[bytes]] = []
    ttfb: float | None = None

    start = time.perf_counter()

    for chunk in cell.chunks:
        deltas: list[bytes] = []
        async with http.stream(
            "POST", SPEECH_URL, json=_speech_payload(chunk, voice, model)
        ) as response:
            if response.status_code != httpx.codes.OK:
                await response.aread()
                raise SystemExit(
                    f"speech request failed with {response.status_code}: "
                    f"{response.text[:300]}"
                )

            async for audio in _audio_from_stream(response):
                if ttfb is None:
                    ttfb = time.perf_counter() - start
                deltas.append(audio)
        replies.append(deltas)

    total = time.perf_counter() - start

    if ttfb is None:
        raise SystemExit("the stream produced no audio at all")

    return Trial(
        ttfb=ttfb,
        total=total,
        audio_bytes=sum(len(delta) for deltas in replies for delta in deltas),
        audio_seconds=_mp3_seconds([b"".join(deltas) for deltas in replies]),
    )


def _rotations(cells: list[Cell], trials: int) -> Iterator[list[Cell]]:
    """Yield the cell order for each trial, rotated one place each time.

    Network conditions drift over a run that makes hundreds of requests. Fixed
    ordering hands that drift to whichever strategy always goes last, and the
    result looks like a property of the strategy.
    """
    for trial in range(trials):
        offset = trial % len(cells) if cells else 0
        yield cells[offset:] + cells[:offset]


def _summarise(values: list[float]) -> str:
    """Return a median with the observed spread beside it."""
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.median(values):.2f} ({min(values):.2f}-{max(values):.2f})"


def report_offline(cells: list[Cell]) -> None:
    """Print what each strategy does to the text, before any request is made."""
    print("\n== Chunking (deterministic, no requests made) ==\n")

    for strategy in STRATEGIES:
        print(f"  {strategy.name:<16}{strategy.note}")
    print()

    for case in CASES:
        case_cells = [cell for cell in cells if cell.case == case]
        if not case_cells:
            continue
        text = CASES[case]
        paragraphs = len(PARAGRAPH_END.split(text))
        print(f"{case} -- {len(text)} chars, {paragraphs} paragraph(s)")
        print(f"  {'strategy':<16}{'requests':>8}  chunk sizes")
        for cell in case_cells:
            sizes = " ".join(str(len(chunk)) for chunk in cell.chunks)
            print(f"  {cell.strategy.name:<16}{cell.requests:>8}  {sizes}")
        print()

    _report_lookbehind(cells)
    _report_floor(cells)


def _report_lookbehind(cells: list[Cell]) -> None:
    """Print whether the abbreviation lookbehind changes anything.

    This is the cheapest question in the whole exercise and the one most worth
    answering: the lookbehind is complexity carried on every reply, and if it
    never moves a boundary on realistic text it is complexity for nothing.
    """
    print("== What the abbreviation lookbehind changes ==\n")

    for case in CASES:
        shipped = next(
            (c for c in cells if c.case == case and c.strategy.name == "sentences-40"),
            None,
        )
        naive = next(
            (c for c in cells if c.case == case and c.strategy.name == "naive-40"),
            None,
        )
        if shipped is None or naive is None:
            continue

        if shipped.chunks == naive.chunks:
            print(f"  {case:<16}no difference -- same {shipped.requests} chunk(s)")
            continue

        print(
            f"  {case:<16}{naive.requests} chunks without it, {shipped.requests} with"
        )
        for chunk in naive.chunks:
            if chunk not in shipped.chunks:
                print(f'    only without: "{chunk[:70]}"')
    print()


def _report_floor(cells: list[Cell]) -> None:
    """Print whether MIN_SPEECH_CHARS changes anything.

    The other piece of complexity [#160] puts in question. The floor exists so
    that "Yes." does not become its own billed request; whether real replies
    ever contain a fragment short enough for it to catch is a different
    question, and this answers it.
    """
    print("== What the minimum length changes ==\n")

    for case in CASES:
        floored = next(
            (c for c in cells if c.case == case and c.strategy.name == "sentences-40"),
            None,
        )
        unfloored = next(
            (c for c in cells if c.case == case and c.strategy.name == "sentences-0"),
            None,
        )
        if floored is None or unfloored is None:
            continue

        if floored.chunks == unfloored.chunks:
            print(f"  {case:<16}no difference -- same {floored.requests} chunk(s)")
            continue

        print(
            f"  {case:<16}{unfloored.requests} chunks without it, "
            f"{floored.requests} with"
        )
        for chunk in unfloored.chunks:
            if chunk not in floored.chunks:
                print(f'    only without: "{chunk[:70]}"')
    print()


def report_timings(cells: list[Cell], trials: int) -> None:
    """Print the measured half, medians first and spread beside them."""
    print(f"\n== Timing (median of {trials} trials, min-max in brackets) ==\n")

    for case in CASES:
        case_cells = [cell for cell in cells if cell.case == case and cell.trials]
        if not case_cells:
            continue

        print(f"{case}")
        print(
            f"  {'strategy':<16}{'reqs':>5}  {'TTFB s':<22}"
            f"{'total s':<22}{'audio s':<16}"
        )
        for cell in case_cells:
            ttfbs = [trial.ttfb for trial in cell.trials]
            totals = [trial.total for trial in cell.trials]
            seconds = [
                trial.audio_seconds
                for trial in cell.trials
                if trial.audio_seconds is not None
            ]
            print(
                f"  {cell.strategy.name:<16}{cell.requests:>5}  "
                f"{_summarise(ttfbs):<22}{_summarise(totals):<22}"
                f"{_summarise(seconds):<16}"
            )
        print()


async def main() -> None:
    """Build the cells, run whichever half was asked for, and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="only the deterministic half: no key, no requests, no cost",
    )
    parser.add_argument("--voice", help="voice id to speak with (required when live)")
    parser.add_argument("--model", default=TTS_MODEL, help="speech model")
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="timed runs per cell; the median is what gets reported",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="limit to one case, repeatable (default: all)",
    )
    args = parser.parse_args()

    wanted = args.case or list(CASES)
    cells = [
        Cell(
            case=case,
            strategy=strategy,
            chunks=(
                await shipped_chunks(CASES[case])
                if strategy.name == SHIPPED
                else regroup(CASES[case], strategy)
            ),
        )
        for case in wanted
        for strategy in STRATEGIES
    ]

    report_offline(cells)

    if args.offline:
        billed = sum(cell.requests for cell in cells) * args.trials
        print(f"Offline only. A live run would have made {billed} billed requests.\n")
        return

    if not args.voice:
        raise SystemExit("--voice is required for a live run; see --offline")

    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit("MISTRAL_API_KEY is not set")

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {key}"}, timeout=120.0
    ) as http:
        # One discarded request before anything is timed. The first request of
        # a process pays for DNS, the TLS handshake and connection setup, and
        # charging that to whichever cell happens to run first is exactly the
        # kind of missing per-case baseline this script exists to avoid.
        await run_trial(
            http,
            Cell(case="warmup", strategy=STRATEGIES[0], chunks=["Warming up."]),
            args.voice,
            args.model,
        )

        for order in _rotations(cells, args.trials):
            for cell in order:
                cell.trials.append(await run_trial(http, cell, args.voice, args.model))

    report_timings(cells, args.trials)


if __name__ == "__main__":
    asyncio.run(main())
