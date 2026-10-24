"""Keyboard operator console for the drive server (the console window must have focus).

    SPACE  emergency stop: brake to standstill, then stop sending commands (counts as a takeover)
    r      resume after an emergency stop
    i      mark a takeover / intervention (car would have needed a human)
    o      mark an off-track event
    l      mark "lap completed" (your own observation of the simulator)
    q      quit: brake, then shut the server down cleanly
Ctrl+C in the console also stops the server.
"""

from __future__ import annotations

import logging
import sys
import threading

log = logging.getLogger(__name__)

HELP = __doc__.split("\n", 2)[2]


class OperatorConsole(threading.Thread):
    def __init__(self, session, on_quit):
        super().__init__(name="operator-console", daemon=True)
        self.session = session
        self.on_quit = on_quit
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def handle_key(self, key: str) -> None:
        s = self.session
        if key == " ":
            s.emergency_stop("operator", "SPACE pressed")
        elif key == "r":
            s.resume()
        elif key == "i":
            s.mark("intervention")
        elif key == "o":
            s.mark("off_track")
        elif key == "l":
            s.mark("lap_complete")
        elif key == "q":
            s.request_quit()
            self.on_quit()

    def run(self) -> None:
        if sys.platform != "win32":
            log.warning("keyboard console is only implemented for Windows (msvcrt); use Ctrl+C to stop")
            return
        import msvcrt
        import time

        while not self._stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # arrow/function key prefix: discard the second code
                    msvcrt.getwch()
                    continue
                self.handle_key(ch.lower())
            else:
                time.sleep(0.02)
