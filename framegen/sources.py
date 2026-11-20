"""Where frames come from: what each kind of source is like, and the pieces every source shares.

A source (framegen.io, framegen.chiaki_tap) yields Ticks: one frame for each refresh of the console's output,
new content or a repeat. It describes itself with `info` (a SourceInfo), so the pipeline and run.py read what it
is like instead of checking which class it is.

Shared pieces:
  TestMode       "Simulate a 30 fps game": a 60 fps game played as if it were 30 fps, with the frames it leaves
                 out kept to score the generated ones against. Works with any 60 Hz source.
  Clock60        puts a source whose frames come at uneven times (a Moonlight stream through the frame tap) onto a
                 steady 60 Hz beat, the way a console's HDMI output delivers them.
  Steady60       the same for an app's window, one frame a beat, with the beat kept just after the frames come
                 (BeatBuffer), so a frame a little late doesn't put a 30 fps game out of step.
  make_detector  the repeat detector that suits a source.

`python -m framegen.sources --json` (or `run.py --list-sources --json`) lists the sources for the launcher. It
imports nothing heavy, so it answers in well under a second.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace

# Sources that work but aren't offered in the launcher: video files are for testing. SMOOVPLAY_SOURCES=all shows
# them anyway.
HIDDEN = {"file"}
# Offered as Experimental: built from the best information available, without the hardware to test every setup. The
# launcher marks them and offers Send feedback (a GitHub issue form).
EXPERIMENTAL = {"capture", "window", "tap"}


@dataclass(frozen=True)
class SourceInfo:
    kind: str  # "chiaki" (PS5 Remote Play), "capture" (capture card), "window" (another app's window, framegen/window.py),
    # "tap" (another app's decoded frames through the frame tap, framegen/tap.py), "file" (a recording as-is),
    # "sim" (simulated PS5)
    name: str  # shown to people
    live: bool = True  # frames come in real time; False: a file, read as fast as the pipeline takes them
    rate: str = "fixed"  # "fixed": a frame every refresh, repeats included (Remote Play, HDMI). "variable": frames
    # come when the sender chooses (window capture, Moonlight), so Clock60 puts them on a 60 Hz beat
    hz: float = 60.0  # refreshes per second, for a fixed rate
    repeats: str = "pixels"  # how a repeat is told from new content: "hint" (the source says, e.g. the frame tap's
    # flags; the picture check is only a fallback) or "pixels" (pictures compared: see make_detector)
    test_mode: bool = True  # "Simulate a 30 fps game" works with it (False: built in, as for "sim")
    full_range: bool = True  # frames come as full-range RGB (0-255); False: limited range (16-235), to be expanded
    audio: str = ""  # who plays the sound, in words ("" = nobody yet)
    controls: str = ""  # where the controller goes, in words


KINDS = {
    "chiaki": SourceInfo("chiaki", "PS5 Remote Play", audio="chiaki-ng plays the sound",
                         controls="Your controller connects to this PC; chiaki-ng sends it to the PS5"),
    "capture": SourceInfo("capture", "Capture card", audio="SmoovPlay plays the card's sound, in step with the picture",
                          controls="Your controller connects to the console as usual"),
    "window": SourceInfo("window", "App window", rate="variable", test_mode=False,
                         audio="The app plays its own sound, a little ahead of the picture",
                         controls="Your controller connects to the app as usual"),
    "tap": SourceInfo("tap", "Frame tap (for developers)", audio="The app plays its own sound",
                      controls="Your controller connects to the app as usual"),
    "file": SourceInfo("file", "Video file", live=False),
    "sim": SourceInfo("sim", "Video file as a PS5 30 fps game", live=False, test_mode=False),
}


def source_info(kind: str, **changes) -> SourceInfo:
    """A kind's description, with what only the running source knows (its frame rate, ...) filled in."""
    return replace(KINDS[kind], **changes)


# ----------------------------------------------------------------------------- test mode
class TestMode:
    """"Simulate a 30 fps game", for any 60 Hz source: frames 0, 2, 4... of a 60 fps game arrive as new content,
    and each odd frame is replaced by the one before it, so the stream looks like a 30 fps game (A A B B). The
    frames left out are kept, so the pipeline can score each generated frame against the real one it replaced:
    gt(content) is the frame left out at `content`.

    It's a filter, tick -> tick or None (drop it). A source that queues frames on a thread of its own (Remote Play,
    capture) applies it before the queue, so a frame dropped from a full queue never shifts which frames are kept;
    a file applies it as it reads. Switching it on or off (set_on) takes effect from the next frame.

    It needs a game running at a steady 60 fps. Each tick carries extra["game_new"], the source's own word on whether
    the game drew a new frame there (Remote Play's size check; None for a source that doesn't say), so the pipeline
    can tell when the game isn't at 60, and gt() won't offer a frame the game itself repeated.
    """

    __test__ = False  # not a pytest test class, despite the name

    KEEP = 8  # left-out frames kept for scoring; the pipeline asks within a frame or two

    def __init__(self, on: bool = False) -> None:
        self.on = on
        self._n = 0  # frames seen while on
        self._last = None  # the last kept frame
        self._truth: dict = {}
        self._game: dict = {}  # the source's repeat hint for each recent frame: True new, False repeat, None unknown

    def set_on(self, on: bool) -> None:
        if on and not self.on:
            self._last = None  # don't repeat a frame from before it was switched on
            self._truth.clear()
            self._game.clear()
        self.on = on

    def gt(self, content: int):
        """The real frame left out at `content`, or None. None too when the game repeated a frame there or at the
        kept frames either side (by the source's hint): below 60 fps the left-out frame isn't the real one halfway
        between them, so there's nothing fair to compare with."""
        if any(self._game.get(i) is False for i in (content - 1, content, content + 1)):
            return None
        return self._truth.get(content)

    def __call__(self, tick):
        if not self.on:
            return tick
        i = self._n
        self._n += 1
        # (Remote Play: its size check, which only says roughly; the pipeline's own call is the picture's)
        self._game[i] = tick.extra["game_new"] = tick.extra.get("size_new", tick.new_hint)
        self._game.pop(i - self.KEEP, None)
        if i % 2 == 0:  # kept: arrives as new content
            self._last = tick.frame
            tick.content, tick.truth_new, tick.new_hint = i, True, True
        elif self._last is not None:  # left out: the kept frame again, the real one kept as truth
            self._truth[i] = tick.frame
            self._truth.pop(i - self.KEEP, None)
            tick.frame = self._last
            tick.content, tick.truth_new, tick.new_hint = i - 1, False, False
        else:  # switched on at an odd frame: nothing to repeat yet
            return None
        return tick

    def wrap(self, ticks):
        from .io import Waiting
        for tick in ticks:
            if isinstance(tick, Waiting):  # (no frame yet: passed on, for the loop to keep the window answering)
                yield tick
            elif (tick := self(tick)) is not None:
                yield tick


# ----------------------------------------------------------------------------- 60 Hz clock
class Clock60:
    """Puts a variable-rate source on a steady beat (60 Hz by default), the way a console's HDMI output delivers
    frames: on each beat the newest frame is handed on, or the previous one again if nothing came since. The
    source is read on a thread of its own, so a late frame never holds up the beat.

    Each beat's tick: a fresh frame keeps the source's own repeat hint (usually None: the pipeline compares
    pictures); a beat with no fresh frame is a repeat for certain (new_hint False). t_arrival stays when the frame
    really arrived (so latency includes the wait for the beat); t_ready is when the beat handed it on, the first
    moment the pipeline could have it. Frames overtaken by a newer one before their beat are counted in
    `overtaken`.
    """

    def __init__(self, source, hz: float = 60.0, clock=time.perf_counter, sleep=time.sleep) -> None:
        self.source, self.period = source, 1.0 / hz
        self._clock, self._sleep = clock, sleep
        self._cond = threading.Condition()
        self._latest = None
        self._serial = 0  # frames received so far
        self._done = False
        self.overtaken = 0

    def __getattr__(self, name: str):  # w, h, info, close... come from the source
        return getattr(self.source, name)

    def _reader(self) -> None:
        from .io import Waiting
        try:
            for tick in self.source:
                if isinstance(tick, Waiting):  # the beat goes on by itself: it repeats the last frame
                    continue
                with self._cond:
                    if self._latest is not None and self._serial > self._taken:
                        self.overtaken += 1
                    self._latest, self._serial = tick, self._serial + 1
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def __iter__(self):
        from .io import Tick
        self._taken = 0  # the serial of the frame handed on last
        threading.Thread(target=self._reader, name="clock60", daemon=True).start()
        with self._cond:  # the first beat is the first frame
            self._cond.wait_for(lambda: self._latest is not None or self._done)
        beat, index = self._clock(), 0
        while True:
            with self._cond:
                latest, serial, done = self._latest, self._serial, self._done
                fresh = serial != self._taken
                self._taken = serial
            if latest is None or (done and not fresh):
                return
            handed = self._clock()  # the beat, or a moment after it if the sleep overshot
            yield Tick(latest.frame, latest.t_arrival if fresh else handed, index, latest.content,
                       latest.truth_new if fresh else None,
                       new_hint=latest.new_hint if fresh else False, damaged=latest.damaged and fresh,
                       t_ready=handed, t_source=latest.t_source, extra=dict(latest.extra) if fresh else {})
            index += 1
            beat += self.period
            now = self._clock()
            if beat < now - self.period:  # fell a whole beat behind (the pipeline stalled): restart the beat from now
                beat = now
            elif beat > now:
                self._sleep(beat - now)


class BeatBuffer:
    """The steady beat for an app's window (Steady60), without the threads: every frame the app draws is queued,
    and each beat hands on the oldest one, so each frame gets a beat of its own and none is overtaken.

    Why: an app draws its stream when frames come off the network, and the desktop shows them on its own refresh,
    so frames reach the screen unevenly. On a 175 Hz desktop, a 60 Hz Xbox app stream's frames came 5.7 to 28.6 ms
    apart (in steps of the desktop's 5.7 ms refresh). Clock60 hands on the newest frame on each beat of a clock of
    its own: a frame a little late lands a beat later, the beat before gets a repeat and the one after gets two
    frames (one overtaken), so a 30 fps game's new, repeat, new, repeat comes out of step about one beat in five.
    That read as a game at 35 to 45 fps and switched Even out 40-60 fps on and off (issue 33's log, and the same
    in a recorded Xbox session: 0 to 28% of the time, by where Clock60's beat happened to fall).

    The beat is steered to just after the frames come: over the last WINDOW frames handed on, the shortest wait
    for the beat is kept near MARGIN, moving the beat by at most STEP a beat (so it follows an app slightly
    faster or slower than 60 Hz). A beat that found nothing, followed within half a beat by a frame, was early:
    the next beat comes STEP later. More than MAX_QUEUE frames waiting (the pipeline stalled, or the app draws
    faster than the beat) drops the oldest: they're counted in `dropped`.

    In two recorded Xbox app sessions, 92.5% and 88.1% of the game's frames were on screen for exactly two beats,
    close to the app's own 93.8% and 90.2% (Clock60: 74 to 81%, and 7 to 10% for a single beat, a stutter), and
    Even out 40-60 fps no longer switched on in scenes with motion. A frame waits about 12 to 17 ms for its beat
    (median; Clock60: 6 ms), the price of an even picture from uneven frames; in a stand-in session through the
    whole pipeline, the delay to the screen stayed the same at the median and fell from 64 to 46 ms at the 95th
    percentile, with Even out 40-60 fps off.
    """

    MARGIN, STEP, GAIN, WINDOW, MAX_QUEUE = 0.002, 0.0005, 0.1, 30, 2

    def __init__(self, period: float) -> None:
        from collections import deque
        self.period = period
        self._queue: deque = deque()  # (item, its arrival), oldest first
        self._waits: deque = deque(maxlen=self.WINDOW)  # how long the last frames handed on waited for their beat
        self._empty_at: float | None = None  # the last beat, if it found nothing to hand on
        self._early = False  # a frame came soon after a beat that found nothing
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._queue)

    def put(self, item, t_arrival: float) -> None:
        if self._empty_at is not None and t_arrival - self._empty_at < 0.5 * self.period:
            self._early = True
        self._empty_at = None
        self._queue.append((item, t_arrival))

    def take(self, now: float):
        """The beat at `now`: the oldest frame waiting, or None (a repeat)."""
        while len(self._queue) > self.MAX_QUEUE:
            self._queue.popleft()
            self.dropped += 1
        if not self._queue:
            self._empty_at = now
            return None
        item, t_arrival = self._queue.popleft()
        self._waits.append(now - t_arrival)
        return item

    def next_beat(self, beat: float) -> float:
        """When the beat after the one at `beat` falls."""
        shift = 0.0
        if self._early:
            self._early, shift = False, self.STEP
        elif len(self._waits) >= 5:
            shift = -max(-self.STEP, min(self.STEP, self.GAIN * (min(self._waits) - self.MARGIN)))
        return beat + self.period + shift


class Steady60(Clock60):
    """Clock60 for an app's window: each frame the app draws is handed on, one a beat, the beat kept just after the
    frames come (BeatBuffer has why). A beat with nothing new is a repeat for certain (new_hint False), as with
    Clock60, and so are t_arrival and t_ready. Frames dropped because too many were waiting are in `overtaken`."""

    def __init__(self, source, hz: float = 60.0, clock=time.perf_counter, sleep=time.sleep) -> None:
        super().__init__(source, hz, clock, sleep)
        self.buffer = BeatBuffer(self.period)

    def _reader(self) -> None:
        from .io import Waiting
        try:
            for tick in self.source:
                if isinstance(tick, Waiting):
                    continue
                with self._cond:
                    self.buffer.put(tick, tick.t_arrival)
                    self._latest = self._latest or tick
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def __iter__(self):
        from .io import Tick
        threading.Thread(target=self._reader, name="steady60", daemon=True).start()
        with self._cond:
            self._cond.wait_for(lambda: self._latest is not None or self._done)
        beat, index, last = self._clock(), 0, None
        while True:
            handed = self._clock()  # the beat, or a moment after it if the sleep overshot
            with self._cond:
                fresh, done = self.buffer.take(handed), self._done
                self.overtaken = self.buffer.dropped
            if fresh is None and (done or last is None):
                return
            if fresh is not None:
                last = fresh
                yield replace(fresh, index=index, t_ready=handed, extra=dict(fresh.extra))
            else:
                yield Tick(last.frame, handed, index, last.content, None, new_hint=False, t_ready=handed,
                           t_source=last.t_source)
            index += 1
            beat = self.buffer.next_beat(beat)
            now = self._clock()
            if beat < now - self.period:  # fell a whole beat behind (the pipeline stalled): restart the beat from now
                beat = now
            elif beat > now:
                self._sleep(beat - now)


# ----------------------------------------------------------------------------- repeat detector
def make_detector(info: SourceInfo, threshold: float = 2.0):
    """The repeat detector for a source. Files compare each frame with the previous one; a capture card compares with
    the last *new* frame, so a slow change that stays under the threshold from frame to frame still adds up and is
    caught; PS5 Remote Play, an app's window, or an app's decoded frames through the frame tap learn where the line
    between their repeats and new frames is (a decoded stream's repeats are never quite identical, but change far
    less than new frames)."""
    from .cadence import DuplicateDetector
    return DuplicateDetector(threshold, reference="new" if info.kind == "capture" else "previous",
                             learn=info.kind in ("chiaki", "window", "tap"))


# ----------------------------------------------------------------------------- the list for the launcher
GUIDES = {
    "chiaki": "Connect your PS5 with chiaki-ng (SmoovPlay sets it up), then press Play.",
    "capture": "Connect the console's HDMI to the capture card and the card to this PC. On a PS5, turn off HDCP "
               "while you use it (Settings > System > HDMI). Set the console to 1080p, 60 Hz, HDR off (or the Capture card "
               "resolution picked in Settings).",
    "window": "Open the app you stream with (PS Remote Play, the Xbox app, Moonlight, or a browser) and start the "
              "game, pick its window, then press Play. Keep the app's window on screen: SmoovPlay shows the smooth "
              "picture on top of it, and the app keeps the controller and the keyboard. SmoovPlay's own keys use Ctrl+Alt "
              "(Ctrl+Alt+M menu, Ctrl+Alt+F fullscreen, Ctrl+Alt+Esc quit), and can be changed on the Play page.",
    "tap": "For app developers: an app built to send its frames straight to SmoovPlay (docs/frame-tap.md in "
           "SmoovPlay's repo). For the Xbox app, PS Remote Play, Moonlight or a browser, pick App window instead. "
           "Press Play, then start the app: its smooth video opens in its own window, and the app keeps the "
           "controller and its sound.",
    "file": "For testing: a video file played as a source.",
}


def describe(kind: str) -> dict:
    """One source for the launcher: its info, whether it's offered, and a line on setting it up."""
    return {**asdict(KINDS[kind]), "hidden": kind in HIDDEN, "experimental": kind in EXPERIMENTAL,
            "guide": GUIDES[kind]}


def list_sources(show_all: bool = False) -> list[dict]:
    out = []
    for kind in ("chiaki", "capture", "window", "tap", "file"):
        d = describe(kind)
        if show_all:
            d["hidden"] = False
        out.append(d)
    return out


def main(argv: list[str] | None = None) -> int:
    import os
    argv = sys.argv[1:] if argv is None else argv
    show_all = os.environ.get("SMOOVPLAY_SOURCES", "").lower() == "all" or "--all" in argv
    sources = list_sources(show_all)
    if "--json" in argv:
        print(json.dumps({"sources": sources}))
    else:
        for s in sources:
            print(f"{s['kind']:8s} {s['name']}{'  (hidden)' if s['hidden'] else ''}\n         {s['guide']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
