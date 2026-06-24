# =====================================================================
# OPENAI DROP-IN FOR THE ANTHROPIC CLIENT
# ---------------------------------------------------------------------
# The whole agent talks to one object via a single call shape:
#
#     msg = client.messages.create(
#         model=..., max_tokens=...,
#         system=[{"type": "text", "text": SYSTEM, "cache_control": {...}}],
#         messages=[{"role": "user", "content": context}],
#     )
#     text = "".join(b.text for b in msg.content if b.type == "text")
#
# This module exposes an object with that exact surface but routes the call to
# OpenAI's chat.completions, so we can A/B gpt-4.1-mini against Claude by flipping
# LLM_BACKEND — no changes at any call site (matching.reply / _llm_parse / ...).
# =====================================================================


class _TextBlock:
    """Mimics an Anthropic content block: has `.type` and `.text`."""
    __slots__ = ("type", "text")

    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    """Mimics an Anthropic message: `.content` is a list of content blocks."""

    def __init__(self, text):
        self.content = [_TextBlock(text)]


class _Messages:
    def __init__(self, client):
        self._client = client

    def create(self, model, max_tokens, system=None, messages=None, **_ignored):
        # Anthropic carries the system prompt in a top-level `system` list of text
        # blocks; OpenAI wants it as a leading {"role": "system"} message. Flatten the
        # blocks (dropping cache_control, which OpenAI handles transparently) and
        # translate. Our call sites only ever send plain-string user content, but we
        # handle the block form too so this stays a faithful stand-in.
        oai_messages = []
        if system:
            if isinstance(system, list):
                sys_text = "\n".join(
                    b["text"] for b in system if isinstance(b, dict) and b.get("text")
                )
            else:
                sys_text = str(system)
            if sys_text:
                oai_messages.append({"role": "system", "content": sys_text})

        for m in (messages or []):
            content = m["content"]
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            oai_messages.append({"role": m["role"], "content": content})

        # Newer OpenAI models (gpt-5.x, reasoning models) reject the legacy `max_tokens`
        # and require `max_completion_tokens`; gpt-4.1 accepts it too, so we use it
        # uniformly. Anthropic's `max_tokens` maps straight onto it.
        resp = self._client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=oai_messages,
        )
        return _Response(resp.choices[0].message.content or "")


class OpenAIClient:
    """anthropic.Anthropic look-alike backed by OpenAI's chat.completions."""

    def __init__(self, api_key):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.messages = _Messages(self._client)
