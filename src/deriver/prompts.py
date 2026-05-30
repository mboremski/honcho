"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from datetime import datetime
from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens

# Fixed reference time used only for static token estimation. The exact value is
# irrelevant; it just needs to render a representative date string.
_ESTIMATE_REFERENCE_TIME = datetime(2025, 1, 1)


def _normalized_custom_instructions(custom_instructions: str | None) -> str | None:
    """Return stripped custom instructions, if any."""
    if custom_instructions is None:
        return None

    normalized = custom_instructions.strip()
    return normalized or None


def _custom_instructions_section(custom_instructions: str | None) -> str:
    """Render optional custom instructions for the deriver prompt."""
    normalized_custom_instructions = _normalized_custom_instructions(
        custom_instructions
    )
    if normalized_custom_instructions is None:
        return ""

    return c(
        f"""
        CUSTOM INSTRUCTIONS:
        {normalized_custom_instructions}
        """
    )


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
    current_time: datetime,
    custom_instructions: str | None = None,
) -> str:
    """
    Generate minimal prompt for fast observation extraction.

    Args:
        peer_id: The ID of the user being analyzed.
        messages: All messages in the range (interleaving messages and new turns combined).
        current_time: The wall-clock time at which this batch is being processed.
            Used to anchor extracted facts to "now" so the model can tell how old
            a message is (e.g. when ingesting historically-dated imports).

    Returns:
        Formatted prompt string for observation extraction.
    """
    custom_instructions_section = _custom_instructions_section(custom_instructions)
    current_date_str = current_time.strftime("%Y-%m-%d")
    return c(
        f"""
Analyze messages from {peer_id} to extract **explicit atomic facts** about them.

TEMPORAL CONTEXT:
- Today's date is {current_date_str}.
- Each message below is prefixed with the timestamp at which it was sent (format: YYYY-MM-DD HH:MM:SS). These messages may be recent, or they may be imported from the distant past, so always anchor a fact to WHEN it was stated using that timestamp.
- For facts that can change over time (e.g. job, employer, city of residence, age, relationship status, current preferences), qualify them with their time of validity instead of stating them as timeless present facts. Example: if a 2017 message says "I work in Erkelenz", extract "{peer_id} worked in Erkelenz as of 2017" — NOT "{peer_id} works in Erkelenz".
- Facts that are stable over time (e.g. place of birth, date of birth, nationality, past events that already happened) do not need this qualification.

[EXPLICIT] DEFINITION: Facts about {peer_id} that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- Properly attribute observations to the correct subject: if it is about {peer_id}, say so. If {peer_id} is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand {peer_id}.
- Extract ALL observations from {peer_id} messages, using others as context.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

EXAMPLES:
- EXPLICIT: "I just had my 25th birthday last Saturday" → "{peer_id} is 25 years old", "{peer_id}'s birthday is June 21st"
- EXPLICIT: "I took my dog for a walk in NYC" → "{peer_id} has a dog", "{peer_id} lives in NYC"
- EXPLICIT: "{peer_id} attended college" + general knowledge → "{peer_id} completed high school or equivalent"

{custom_instructions_section}

Messages to analyze:
<messages>
{messages}
</messages>
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate the static minimal deriver prompt without custom instructions."""
    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        current_time=_ESTIMATE_REFERENCE_TIME,
        custom_instructions=None,
    )
    return estimate_tokens(prompt)


def estimate_deriver_prompt_tokens(custom_instructions: str | None) -> int:
    """Estimate minimal deriver prompt tokens, including custom instructions if present."""
    normalized_custom_instructions = _normalized_custom_instructions(
        custom_instructions
    )
    if normalized_custom_instructions is None:
        return estimate_minimal_deriver_prompt_tokens()

    prompt = minimal_deriver_prompt(
        peer_id="",
        messages="",
        current_time=_ESTIMATE_REFERENCE_TIME,
        custom_instructions=normalized_custom_instructions,
    )
    return estimate_tokens(prompt)
