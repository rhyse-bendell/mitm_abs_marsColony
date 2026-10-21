"""Realtime testbed scheduling (`execution_mode = realtime_testbed`).

`simulation` mode (the default, unchanged) advances simulated time only when a driver calls `SimulationState.update(base_dt)` in a loop.
`realtime_testbed` keeps the SAME `update` and the same per-tick semantics, but a `RealtimeScheduler` decides *how many* ticks to run from the wall clock:

    simulated seconds elapsed == wall seconds elapsed * time_scale   (time_scale = 1.0, i.e. 1x, by default)

Consequences (all by construction, nothing here touches world rules):
* The mission clock keeps running while a person thinks: ticks are driven by the wall clock, not by anyone's input.
* No controller can freeze the world. A human with no intent simply does nothing; an Ollama call runs on a worker and only its own agent waits
  (the global inference barrier is bypassed in this mode, see SimulationState.update).
* Actor-specific task durations / busy states are ordinary simulated time and apply unchanged.
* Pause pauses everything (the wall anchor is reset on resume, so no burst of catch-up ticks); Stop ends the run safely and once.
* If the host stalls (a slow GUI redraw), at most `max_catchup_s` of wall time is replayed per call; the rest is recorded as `dropped_wall_s`
  rather than silently bursting the simulation forward.

The scheduler is single-threaded by contract: call `advance()` from the one thread that owns the SimulationState (the Tk loop, or `run_blocking`).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional


class RealtimeScheduler:
    def __init__(self, sim, *, tick_s: float = 0.1, time_scale: float = 1.0, max_catchup_s: float = 1.0, clock: Callable[[], float] = time.monotonic):
        if tick_s <= 0 or time_scale <= 0:
            raise ValueError("tick_s and time_scale must be positive")
        self.sim = sim
        self.tick_s = float(tick_s)
        self.time_scale = float(time_scale)
        self.max_catchup_s = float(max_catchup_s)
        self._clock = clock
        self._last: Optional[float] = None
        self._acc = 0.0
        self.paused = False
        self.stopped = False
        self.sim_steps = 0
        self.dropped_wall_s = 0.0
        self.wall_running_s = 0.0
        sim.speed_multiplier = 1.0  # update(dt) uses dt * speed_multiplier; pacing belongs to this scheduler alone

    # -- control ---------------------------------------------------------------------------------------------------------
    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self._last = None  # never replay the paused interval

    def stop(self) -> None:
        if not self.stopped:
            self.stopped = True
            self.sim.stop()

    @property
    def finished(self) -> bool:
        return self.stopped or bool(getattr(self.sim, "terminal_state", None))

    # -- stepping --------------------------------------------------------------------------------------------------------
    def advance(self) -> int:
        """Run the ticks the wall clock has earned since the last call. Returns how many were run."""
        now = self._clock()
        if self.stopped or self.paused or self._last is None:
            self._last = now
            return 0
        elapsed = max(0.0, now - self._last)
        self._last = now
        if elapsed > self.max_catchup_s:
            self.dropped_wall_s += elapsed - self.max_catchup_s
            elapsed = self.max_catchup_s
        self.wall_running_s += elapsed
        self._acc += elapsed * self.time_scale
        steps = 0
        while self._acc >= self.tick_s and not self.finished:
            self.sim.update(self.tick_s)
            self._acc -= self.tick_s
            steps += 1
        self.sim_steps += steps
        if getattr(self.sim, "terminal_state", None) and not self.stopped:
            self.stop()
        return steps

    def run_blocking(self, *, poll_s: float = 0.02, until_sim_time: Optional[float] = None, on_tick: Optional[Callable[[], None]] = None) -> None:
        """Headless host loop (no Tk). Ends at terminal state, `until_sim_time`, or stop()."""
        self.resume()
        while not self.finished:
            self.advance()
            if on_tick is not None:
                on_tick()
            if until_sim_time is not None and self.sim.time >= until_sim_time:
                break
            time.sleep(poll_s)
        self.stop()

    def stats(self) -> Dict[str, float]:
        return {"execution_mode": "realtime_testbed", "time_scale": self.time_scale, "tick_s": self.tick_s, "sim_steps": self.sim_steps, "sim_time": float(self.sim.time),
                "wall_running_s": round(self.wall_running_s, 3), "dropped_wall_s": round(self.dropped_wall_s, 3), "paused": self.paused, "stopped": self.stopped}


class HumanTestbedSession:
    """Everything a realtime session with human participants needs, wired in one place: the scheduler, one ParticipantGateway per human agent and the local
    participant server. The host (Tk observer GUI or the headless script) calls `tick()` from the thread that owns the simulation."""

    def __init__(self, sim, *, bind_host: str = "127.0.0.1", port: int = 0, token: Optional[str] = None, time_scale: float = 1.0, tick_s: float = 0.1,
                 clock: Callable[[], float] = time.monotonic):
        from modules.participant_gateway import ParticipantGateway
        from modules.participant_server import ParticipantServer

        if getattr(sim, "execution_mode", None) != "realtime_testbed":
            raise ValueError("a HumanTestbedSession requires execution_mode='realtime_testbed'")
        self.sim = sim
        self.scheduler = RealtimeScheduler(sim, tick_s=tick_s, time_scale=time_scale, clock=clock)
        self.gateways = {a.role: ParticipantGateway(sim, a.name, scheduler=self.scheduler) for a in sim.agents if a.is_human_participant()}
        if not self.gateways:
            raise ValueError("no human_participant agent in this simulation")
        self.server = ParticipantServer(self.gateways, bind_host=bind_host, port=port, token=token)

    def start(self) -> "HumanTestbedSession":
        for gw in self.gateways.values():
            gw.publish(force=True)
        self.server.start()
        self.scheduler.resume()
        return self

    def urls(self) -> Dict[str, str]:
        return {role: self.server.url_for(role) for role in self.gateways}

    def tick(self) -> int:
        steps = self.scheduler.advance()
        for gw in self.gateways.values():
            gw.publish(force=bool(steps) and self.sim.terminal_state is not None)
        return steps

    def pause(self) -> None:
        self.scheduler.pause()

    def resume(self) -> None:
        self.scheduler.resume()

    def stop(self) -> None:
        self.scheduler.stop()
        for gw in self.gateways.values():
            gw.publish(force=True)  # participants see "ended" before the socket closes
        self.server.stop()

    @property
    def finished(self) -> bool:
        return self.scheduler.finished
