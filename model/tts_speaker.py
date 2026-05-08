# ══════════════════════════════════════════════════════════
# tts_speaker.py
# Non-blocking TTS engine using a background thread.
# Supports cooldowns so the same object isn't repeated
# every frame, and priority interrupts for close danger.
# ══════════════════════════════════════════════════════════
import pyttsx3
import threading
import queue
import time


class TTSSpeaker:
    """
    Runs TTS in a daemon background thread so it never
    blocks the video/detection loop.

    Usage:
        speaker = TTSSpeaker()
        speaker.speak("There is a car to your left", key="car")
        speaker.stop()   # call on exit
    """

    def __init__(self, rate=145, volume=1.0,
                 cooldown_sec=5.0):
        """
        rate        : words per minute (145 = clear & not too fast)
        volume      : 0.0 – 1.0
        cooldown_sec: seconds before the same object is announced again
        """
        self.rate         = rate
        self.volume       = volume
        self.cooldown_sec = cooldown_sec

        self._queue     = queue.Queue()
        self._cooldowns = {}          # key → last spoken timestamp
        self._lock      = threading.Lock()

        # Daemon thread dies automatically when main program exits
        self._thread = threading.Thread(
            target=self._worker, daemon=True
        )
        self._thread.start()

    # ── Public API ─────────────────────────────────────────
    def speak(self, message, key=None, priority=False):
        """
        Queue a message to be spoken.

        key      : cooldown identifier (e.g. 'car_right')
                   pass None to always speak
        priority : True = danger alert, clears the queue
                   and speaks immediately
        """
        now = time.time()

        # Cooldown check (skip non-priority duplicates)
        if key and not priority:
            with self._lock:
                last = self._cooldowns.get(key, 0)
                if now - last < self.cooldown_sec:
                    return

        # Update cooldown timestamp
        if key:
            with self._lock:
                self._cooldowns[key] = now

        # Priority: flush current queue, speak first
        if priority:
            self._flush_queue()

        self._queue.put(message)
    
    def stop(self):
        """Gracefully shut down the TTS thread."""
        self._queue.put(None)   # sentinel

    # ── Internal ───────────────────────────────────────────
    def _worker(self):
        """Runs in background thread — consumes the queue."""
        engine = pyttsx3.init()
        engine.setProperty('rate',   self.rate)
        engine.setProperty('volume', self.volume)

        # Prefer a clear female voice if available
        voices = engine.getProperty('voices')
        for v in voices:
            if 'female' in v.name.lower() or \
               'zira'   in v.name.lower() or \
               'hazel'  in v.name.lower():
                engine.setProperty('voice', v.id)
                break

        while True:
            msg = self._queue.get()
            if msg is None:       # stop sentinel
                self._queue.task_done()
                break

            batch = [msg]
            stop_after = False

            # Drain any messages already queued so we run one TTS loop.
            while True:
                try:
                    nxt = self._queue.get_nowait()
                    if nxt is None:
                        stop_after = True
                        self._queue.task_done()
                        break
                    batch.append(nxt)
                except queue.Empty:
                    break

            try:
                for text in batch:
                    engine.say(text)
                engine.runAndWait()
            finally:
                for _ in batch:
                    self._queue.task_done()

            if stop_after:
                break

    def _flush_queue(self):
        """Empty the queue (used before priority messages)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
    # Add this method to the TTSSpeaker class:

    def wait_until_done(self):
        """
        Block the calling thread until every message
        in the queue has been fully spoken.
        Uses queue.join() — reliable, no sleep guessing.
        """
        self._queue.join()