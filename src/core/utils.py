"""
Lab 11 — Helper Utilities
"""
from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


async def chat_with_agent(agent, runner, user_message: str, session_id=None, user_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message, user_id=user_id)
        return text, None

    from google.genai import types

    user_id = user_id or "student"
    app_name = runner.app_name

    import asyncio
    import uuid

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    for attempt in range(5):
        if session is None or attempt > 0:
            sid = f"sess_{uuid.uuid4().hex[:8]}"
            try:
                session = await runner.session_service.create_session(
                    app_name=app_name, user_id=user_id, session_id=sid
                )
            except Exception:
                session = await runner.session_service.create_session(
                    app_name=app_name, user_id=user_id
                )

        try:
            final_response = ""
            async for event in runner.run_async(
                user_id=user_id, session_id=session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text
            return final_response, session
        except Exception as e:
            err_str = str(e)
            if (
                "503" in err_str
                or "429" in err_str
                or "UNAVAILABLE" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
            ) and attempt < 4:
                wait_sec = 15.0 * (attempt + 1)
                print(
                    f"[chat_with_agent] API rate-limit/temporary spike. Waiting {wait_sec}s before retry ({attempt + 1}/4)...",
                    flush=True,
                )
                await asyncio.sleep(wait_sec)
                continue
            raise
