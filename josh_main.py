from __future__ import annotations

import time
import sys

from core.settings import load_settings
from core.brain import Brain
from core.memory import LongTermMemory
from wakeword.porcupine_listener import wait_for_wake_word
from audio.recorder import record_wav
from speech.stt_whisper import WhisperSTT
from audio.tts_piper import say, stream_to_piper

from tools.planner import decide_plan
from tools.time_tool import get_time_string
from tools.web_tool import web_search


TEXT_ONLY = "--text" in sys.argv

OPEN_EXTENDED_PHRASES = (
    "open extended dialogue",
    "open extended dialog",
)

CLOSE_EXTENDED_PHRASES = (
    "that'll be all",
    "that will be all",
)


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    t = (text or "").lower().strip()
    return any(p in t for p in phrases)


def get_user_text_textmode():
    try:
        return input("You: ").strip()
    except EOFError:
        return ""


def maybe_store_memory(user_text: str, long_mem: LongTermMemory):
    triggers = (
        "remember that",
        "store that",
        "save this",
    )
    lower = (user_text or "").lower()
    if any(t in lower for t in triggers):
        long_mem.add_fact(user_text)


def _answer_from_web(brain: Brain, user_text: str, raw_snippets: str, short_history, long_mem) -> str:
    prompt = (
        "You are J.O.S.H. Answer in ONE short sentence.\n"
        "Use the web snippets below to answer the user's question.\n"
        "You MAY do simple arithmetic if the needed numbers appear in the snippets "
        "(example: '$20/month' implies '$240/year').\n"
        "If a price is requested, return ONLY the price and the billing period (e.g., '$20/month' or '$240/year').\n"
        "If the snippets don't contain the needed facts AND you can't derive it from snippet values, say you couldn't find it.\n\n"
        f"User question: {user_text}\n\n"
        f"Web snippets:\n{raw_snippets}\n\n"
        "Answer:"
    )
    # Stream this too, so web answers are fast.
    print("J.O.S.H.: ", end="", flush=True)
    reply = stream_to_piper(
        brain,
        prompt,
        short_history,
        long_mem,
        piper_exe=brain_settings.piper_exe,   # set below
        piper_voice=brain_settings.piper_voice,
    )
    print()
    return reply.strip()


def dialogue_loop(
    settings,
    brain: Brain,
    stt: WhisperSTT,
    short_history,
    long_mem: LongTermMemory,
    first_pre_roll=None,
    *,
    announce_listening: bool = True,
):
    global brain_settings
    brain_settings = settings

    if announce_listening:
        print("J.O.S.H.: I'm listening, sir.")
        say("I'm listening, sir.", settings)

    last_activity = time.time()
    pre = first_pre_roll

    while True:
        if time.time() - last_activity > settings.idle_timeout_seconds:
            print("J.O.S.H.: Idle timeout reached. Shutting down, sir.")
            say("Idle. Shutting down, sir.", settings)
            return "exit"

        # ---- INPUT ----
        if TEXT_ONLY:
            user_text = get_user_text_textmode()
        else:
            record_wav(
                "input.wav",
                in_rate=settings.in_rate,
                channels=settings.channels,
                block_dur=settings.block_dur,
                silence_threshold=settings.silence_threshold,
                silence_seconds=settings.silence_seconds,
                max_record_seconds=settings.max_record_seconds,
                min_record_seconds=settings.min_record_seconds,
                start_grace_seconds=settings.start_grace_seconds,
                speech_start_threshold=settings.speech_start_threshold,
                pre_roll_frames=pre,
            )
            pre = None
            user_text = stt.transcribe("input.wav")

        if not user_text:
            print("J.O.S.H.: Nothing detected, sir.")
            continue

        print("You said:", user_text)
        lower = user_text.lower().strip()

        # ---- EXIT EXTENDED MODE ----
        if _contains_any_phrase(lower, CLOSE_EXTENDED_PHRASES) or "go to sleep" in lower:
            print("J.O.S.H.: Understood, sir.")
            say("Understood, sir.", settings)
            return

        if "shut down" in lower or "exit program" in lower:
            print("J.O.S.H.: Shutting down. Goodbye, sir.")
            say("Shutting down. Goodbye, sir.", settings)
            return "exit"

        # ---- PLAN ----
        plan = decide_plan(settings.openai_api_key, user_text)

        # ---- TOOLS (only if needed) ----
        if plan.needs_tools and plan.tools == ["time"]:
            reply = get_time_string(plan.timezone)
            print(f"J.O.S.H.: {reply}")
            say(reply, settings)

            short_history.append((user_text, reply))
            last_activity = time.time()
            maybe_store_memory(user_text, long_mem)
            long_mem.save(settings.memory_file)
            continue

        if plan.needs_tools and "web" in plan.tools:
            raw = web_search(plan.web_query or user_text)
            reply = _answer_from_web(brain, user_text, raw, short_history, long_mem)

            short_history.append((user_text, reply))
            last_activity = time.time()
            maybe_store_memory(user_text, long_mem)
            long_mem.save(settings.memory_file)
            continue

        if plan.needs_tools and plan.tools == ["weather"]:
            reply = "Weather tool not installed yet, sir."
            print(f"J.O.S.H.: {reply}")
            say(reply, settings)

            short_history.append((user_text, reply))
            last_activity = time.time()
            maybe_store_memory(user_text, long_mem)
            long_mem.save(settings.memory_file)
            continue

        # ---- NORMAL LLM (streaming) ----
        print("J.O.S.H.: ", end="", flush=True)
        reply = stream_to_piper(
            brain,
            user_text,
            short_history,
            long_mem,
            piper_exe=settings.piper_exe,
            piper_voice=settings.piper_voice,
        )
        print()

        short_history.append((user_text, reply))
        last_activity = time.time()
        maybe_store_memory(user_text, long_mem)
        long_mem.save(settings.memory_file)


def alexa_turn(settings, brain: Brain, stt: WhisperSTT, short_history, long_mem: LongTermMemory, first_pre_roll=None):
    global brain_settings
    brain_settings = settings

    # ---- INPUT ONCE ----
    if TEXT_ONLY:
        user_text = get_user_text_textmode()
    else:
        record_wav(
            "input.wav",
            in_rate=settings.in_rate,
            channels=settings.channels,
            block_dur=settings.block_dur,
            silence_threshold=settings.silence_threshold,
            silence_seconds=settings.silence_seconds,
            max_record_seconds=settings.max_record_seconds,
            min_record_seconds=settings.min_record_seconds,
            start_grace_seconds=settings.start_grace_seconds,
            speech_start_threshold=settings.speech_start_threshold,
            pre_roll_frames=first_pre_roll,
        )
        user_text = stt.transcribe("input.wav")

    if not user_text:
        print("J.O.S.H.: Nothing detected, sir.")
        say("Nothing detected, sir.", settings)
        return None

    print("You said:", user_text)
    lower = user_text.lower().strip()

    # ---- SHUTDOWN ----
    if "shut down" in lower or "exit program" in lower:
        print("J.O.S.H.: Shutting down. Goodbye, sir.")
        say("Shutting down. Goodbye, sir.", settings)
        return "exit"

    # ---- ENTER EXTENDED MODE ----
    if _contains_any_phrase(lower, OPEN_EXTENDED_PHRASES):
        print("J.O.S.H.: Extended dialogue activated, sir.")
        say("Extended dialogue activated, sir.", settings)
        return dialogue_loop(
            settings,
            brain,
            stt,
            short_history,
            long_mem,
            first_pre_roll=None,
            announce_listening=False,
        )

    # ---- GO TO SLEEP ----
    if "go to sleep" in lower:
        print("J.O.S.H.: Understood, sir.")
        say("Understood, sir.", settings)
        return None

    # ---- PLAN ----
    plan = decide_plan(settings.openai_api_key, user_text)

    # ---- TOOLS (only if needed) ----
    if plan.needs_tools and plan.tools == ["time"]:
        reply = get_time_string(plan.timezone)
        print(f"J.O.S.H.: {reply}")
        say(reply, settings)
        short_history.append((user_text, reply))
        maybe_store_memory(user_text, long_mem)
        long_mem.save(settings.memory_file)
        return None

    if plan.needs_tools and "web" in plan.tools:
        raw = web_search(plan.web_query or user_text)
        reply = _answer_from_web(brain, user_text, raw, short_history, long_mem)

        short_history.append((user_text, reply))
        maybe_store_memory(user_text, long_mem)
        long_mem.save(settings.memory_file)
        return None

    if plan.needs_tools and plan.tools == ["weather"]:
        reply = "Weather tool not installed yet, sir."
        print(f"J.O.S.H.: {reply}")
        say(reply, settings)
        short_history.append((user_text, reply))
        maybe_store_memory(user_text, long_mem)
        long_mem.save(settings.memory_file)
        return None

    # ---- NORMAL LLM (streaming) ----
    print("J.O.S.H.: ", end="", flush=True)
    reply = stream_to_piper(
        brain,
        user_text,
        short_history,
        long_mem,
        piper_exe=settings.piper_exe,
        piper_voice=settings.piper_voice,
    )
    print()

    short_history.append((user_text, reply))
    maybe_store_memory(user_text, long_mem)
    long_mem.save(settings.memory_file)

    return None


def main():
    settings = load_settings()

    long_mem = LongTermMemory.load(settings.memory_file)
    short_history = []

    stt = WhisperSTT(model_size="base")
    brain = Brain(
        api_key=settings.openai_api_key,
        system_persona=settings.system_persona,
        max_turns_in_context=settings.max_turns_in_context,
    )

    while True:
        if TEXT_ONLY:
            result = alexa_turn(settings, brain, stt, short_history, long_mem, first_pre_roll=None)
        else:
            pre_roll = wait_for_wake_word(
                access_key=settings.picovoice_access_key,
                keyword_path=settings.keyword_path,
                pre_roll_seconds=settings.pre_roll_seconds,
            )
            result = alexa_turn(settings, brain, stt, short_history, long_mem, first_pre_roll=pre_roll)

        if result == "exit":
            break


if __name__ == "__main__":
    main()