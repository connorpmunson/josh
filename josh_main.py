from __future__ import annotations

import time
import threading
from core.settings import load_settings
from core.brain import Brain
from core.memory import LongTermMemory
from wakeword.porcupine_listener import wait_for_wake_word
from audio.recorder import record_wav
from speech.stt_whisper import WhisperSTT
from tools.router import route
from tools.llm_tool_router import decide_tool
from tools.time_tool import get_time_string
from tools.web_tool import web_search
from audio.tts_piper import say, stream_to_piper, stop_speaking
from wakeword.interrupt_listener import wait_for_wake_word_interrupt

# ---- FLAGS ----
import sys
TEXT_ONLY = "--text" in sys.argv


# ---- MODES ----
MODE_ALEXA = "alexa"
MODE_EXTENDED = "extended"

OPEN_EXTENDED_PHRASES = (
    "open extended dialogue",
    "open extended dialog",
)

CLOSE_EXTENDED_PHRASES = (
    "that'll be all",
    "that will be all",
)


# ---------------- HELPER ----------------

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
    lower = user_text.lower()
    if any(t in lower for t in triggers):
        long_mem.add_fact(user_text)

def _start_barge_in_listener(settings):
    if TEXT_ONLY:
        stop_evt = threading.Event()
        return stop_evt, threading.Thread()
    """
    Starts a background wake-word listener that, if triggered,
    calls stop_speaking() to immediately cut TTS.
    Returns (stop_event, listener_thread).
    """
    stop_evt = threading.Event()

    listener = threading.Thread(
        target=wait_for_wake_word_interrupt,
        args=(settings.picovoice_access_key, settings.keyword_path, stop_evt),
        daemon=True,
    )
    listener.start()

    def _watch():
        stop_evt.wait()
        stop_speaking()

    threading.Thread(target=_watch, daemon=True).start()

    return stop_evt, listener


def _say_with_barge_in(text: str, settings):
    stop_evt, listener = _start_barge_in_listener(settings)
    try:
        say(text, settings)
    finally:
        stop_evt.set()
        if listener.is_alive():
            listener.join(timeout=0.2)

# ---------------- EXTENDED DIALOGUE ----------------

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
        if _contains_any_phrase(lower, CLOSE_EXTENDED_PHRASES):
            print("J.O.S.H.: Understood, sir.")
            say("Understood, sir.", settings)
            return

        # ---- HARD COMMANDS ----
        if "go to sleep" in lower:
            print("J.O.S.H.: Understood, sir.")
            say("Understood, sir.", settings)
            return

        if "shut down" in lower or "exit program" in lower:
            print("J.O.S.H.: Shutting down. Goodbye, sir.")
            say("Shutting down. Goodbye, sir.", settings)
            return "exit"

        # ---- TOOL ROUTING ----
        decision = decide_tool(settings.openai_api_key, user_text)

        if decision.tool == "time":
            reply = get_time_string(decision.timezone)
            print(f"J.O.S.H.: {reply}")
            _say_with_barge_in(reply, settings)
            short_history.append((user_text, reply))
            last_activity = time.time()
            maybe_store_memory(user_text, long_mem)
            long_mem.save(settings.memory_file)
            continue

        if decision.tool == "web" and decision.query:
            raw = web_search(decision.query)

            # Ask the LLM to extract the one useful answer from the snippets.
            prompt = (
                "You are J.O.S.H. Answer in ONE short sentence.\n"
                "Use the web snippets below to answer the user's question.\n"
                "If a price is requested, return ONLY the price and the billing period (e.g., '$20/month').\n"
                "If the snippets don't contain the answer, say you couldn't find it.\n\n"
                f"User question: {user_text}\n\n"
                f"Web snippets:\n{raw}\n\n"
                "Answer:"
            )

            reply = brain.think(prompt, short_history, long_mem)
            print(f"J.O.S.H.: {reply}")
            _say_with_barge_in(reply, settings)
            short_history.append((user_text, reply))
            last_activity = time.time()
            maybe_store_memory(user_text, long_mem)
            long_mem.save(settings.memory_file)
            continue

        # fallback: your existing regex router
        tool_result = route(user_text)
        if tool_result.handled:
            reply = tool_result.text.strip()

            # If web tool returned snippets, summarize them before speaking.
            if tool_result.tool_name == "web_tool":
                prompt = (
                    "You are J.O.S.H. Answer in ONE short sentence.\n"
                    "Use the web snippets below to answer the user's question.\n"
                    "If a price is requested, return ONLY the price and the billing period (e.g., '$20/month').\n"
                    "If the snippets don't contain the answer, say you couldn't find it.\n\n"
                    f"User question: {user_text}\n\n"
                    f"Web snippets:\n{reply}\n\n"
                    "Answer:"
                )

                reply = brain.think(prompt, short_history, long_mem).strip()

            print(f"J.O.S.H.: {reply}")
            _say_with_barge_in(reply, settings)
            short_history.append((user_text, reply))
            last_activity = time.time()
            maybe_store_memory(user_text, long_mem)
            long_mem.save(settings.memory_file)
            continue

        # ---- LLM RESPONSE ----
        print("J.O.S.H.: ", end="", flush=True)
        stop_evt, listener = _start_barge_in_listener(settings)
        try:
            reply = stream_to_piper(
                brain,
                user_text,
                short_history,
                long_mem,
                piper_exe=settings.piper_exe,
                piper_voice=settings.piper_voice,
            )
        finally:
            stop_evt.set()
            if listener.is_alive():
                listener.join(timeout=0.2)
        print()

        short_history.append((user_text, reply))
        last_activity = time.time()

        maybe_store_memory(user_text, long_mem)
        long_mem.save(settings.memory_file)


# ---------------- ALEXA MODE ----------------

def alexa_turn(settings, brain: Brain, stt: WhisperSTT, short_history, long_mem: LongTermMemory, first_pre_roll=None):

    #print("J.O.S.H.: I'm listening, sir.")
    #say("I'm listening, sir.", settings)

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

    # ---- TOOL ROUTING ----
    decision = decide_tool(settings.openai_api_key, user_text)

    if decision.tool == "time":
        reply = get_time_string(decision.timezone)
        print(f"J.O.S.H.: {reply}")
        _say_with_barge_in(reply, settings)
        short_history.append((user_text, reply))
        maybe_store_memory(user_text, long_mem)
        long_mem.save(settings.memory_file)
        return None

    if decision.tool == "web" and decision.query:
        raw = web_search(decision.query)

        # Ask the LLM to extract the one useful answer from the snippets.
        prompt = (
            "You are J.O.S.H. Answer in ONE short sentence.\n"
            "Use the web snippets below to answer the user's question.\n"
            "If a price is requested, return ONLY the price and the billing period (e.g., '$20/month').\n"
            "If the snippets don't contain the answer, say you couldn't find it.\n\n"
            f"User question: {user_text}\n\n"
            f"Web snippets:\n{raw}\n\n"
            "Answer:"
        )

        reply = brain.think(prompt, short_history, long_mem)

        print(f"J.O.S.H.: {reply}")
        _say_with_barge_in(reply, settings)
        short_history.append((user_text, reply))
        maybe_store_memory(user_text, long_mem)
        long_mem.save(settings.memory_file)
        return None

    # fallback: existing regex router
    tool_result = route(user_text)
    if tool_result.handled:
        reply = tool_result.text.strip()

        if tool_result.tool_name == "web_tool":
            prompt = (
                "You are J.O.S.H. Answer in ONE short sentence.\n"
                "Use the web snippets below to answer the user's question.\n"
                "If a price is requested, return ONLY the price and the billing period (e.g., '$20/month').\n"
                "If the snippets don't contain the answer, say you couldn't find it.\n\n"
                f"User question: {user_text}\n\n"
                f"Web snippets:\n{reply}\n\n"
                "Answer:"
            )

            reply = brain.think(prompt, short_history, long_mem).strip()

        print(f"J.O.S.H.: {reply}")
        _say_with_barge_in(reply, settings)
        short_history.append((user_text, reply))
        maybe_store_memory(user_text, long_mem)
        long_mem.save(settings.memory_file)
        return None

    # ---- LLM RESPONSE (ONE TURN) ----
    print("J.O.S.H.: ", end="", flush=True)
    stop_evt, listener = _start_barge_in_listener(settings)
    try:
        reply = stream_to_piper(
            brain,
            user_text,
            short_history,
            long_mem,
            piper_exe=settings.piper_exe,
            piper_voice=settings.piper_voice,
        )
    finally:
        stop_evt.set()
        if listener.is_alive():
            listener.join(timeout=0.2)
    print()

    short_history.append((user_text, reply))
    maybe_store_memory(user_text, long_mem)
    long_mem.save(settings.memory_file)

    return None


# ---------------- MAIN ----------------

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