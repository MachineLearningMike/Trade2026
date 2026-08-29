# telepred.py
# Single owner of TelePred indexing (X(t,d), Y(t)).
# See ensemble_plan.txt. Change definition here + bump DEFINITION_ID.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# Bump when X(t,d) / Y(t) slice math changes. Hashed into dataset signature
# only when != DEFINITION_ID_V1 (backward compat for existing TelePred(0) keys).
DEFINITION_ID_V1 = "telepred_v1"
DEFINITION_ID = DEFINITION_ID_V1


@dataclass(frozen=True)
class TelePredSpec:
    delay: int
    delta1: int
    delta2: int
    definition_id: str = DEFINITION_ID

    def __post_init__(self):
        if self.delay < 0:
            raise ValueError(f"delay must be >= 0, got {self.delay}")
        if self.delta1 < 1:
            raise ValueError(f"delta1 must be >= 1, got {self.delta1}")
        if self.delta2 < 1:
            raise ValueError(f"delta2 must be >= 1, got {self.delta2}")


def resolve_past_window(config: dict) -> int:
    """TelePred delta1 / model input length (bars)."""
    if config.get("past_window") is not None:
        return int(config["past_window"])
    # Legacy read shim (removed from config.py)
    if config.get("past_hours") is not None:
        return int(60 * float(config["past_hours"]))
    return 1440


def resolve_delta1(config: dict) -> int:
    """Alias: delta1 == past_window."""
    return resolve_past_window(config)


def resolve_delta2(config: dict) -> int:
    """TelePred delta2 / label future length (bars)."""
    from labels import resolve_future_window

    return int(resolve_future_window(config))


def resolve_delay(config: dict) -> int:
    return int(config.get("telepred_delay", 0) or 0)


def resolve_definition_id(config: dict) -> str:
    return str(config.get("telepred_definition_id", DEFINITION_ID) or DEFINITION_ID)


def parse_delay_list(value) -> List[int]:
    """Parse telepred_delays from list/tuple/str ('0,5,15')."""
    if value is None:
        return [0]
    if isinstance(value, (list, tuple)):
        out = [int(x) for x in value]
    else:
        s = str(value).strip()
        if not s:
            return [0]
        out = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if not out:
        return [0]
    for d in out:
        if d < 0:
            raise ValueError(f"telepred delay must be >= 0, got {d}")
    # unique, preserve order
    seen = set()
    uniq: List[int] = []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def spec_from_config(config: dict, delay: Optional[int] = None) -> TelePredSpec:
    d = int(delay) if delay is not None else resolve_delay(config)
    return TelePredSpec(
        delay=d,
        delta1=resolve_delta1(config),
        delta2=resolve_delta2(config),
        definition_id=resolve_definition_id(config),
    )


def valid_bot_times(
    T: int,
    delta1: int,
    delta2: int,
    delay: int = 0,
    step: int = 1,
) -> List[int]:
    """
    Valid bot times t where X(t,d) and Y(t) fit in [0, T).

    t_min = (delta1 - 1) + delay
    t_max = T - 1 - delta2
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    t_min = (delta1 - 1) + int(delay)
    t_max = int(T) - 1 - int(delta2)
    if t_max < t_min:
        raise ValueError(
            f"No valid bot times: T={T}, delta1={delta1}, delta2={delta2}, "
            f"delay={delay} → t_min={t_min}, t_max={t_max}"
        )
    return list(range(t_min, t_max + 1, step))


def input_slice(t: int, delay: int, delta1: int) -> slice:
    """
    Python slice for X(t,d) = X[t+1-d-delta1 : t+1-d].
    Last included bar index = t - delay.
    """
    start = t + 1 - delay - delta1
    end = t + 1 - delay
    return slice(start, end)


def future_slice(t: int, delta2: int) -> slice:
    """Python slice for p[t+1 : t+1+delta2] (length delta2)."""
    return slice(t + 1, t + 1 + delta2)


def label_history_slice(t: int, delta1: int) -> slice:
    """
    History window ending at p[t] for label families that need a past path
    (trend / latent). Default: p[t+1-delta1 : t+1].

    Yield uses the scalar anchor p[t] directly — see Label_Yield.find(future, current).
    """
    return slice(t + 1 - delta1, t + 1)


def windows_for_sample(
    t: int,
    delay: int,
    delta1: int,
    delta2: int,
) -> Dict[str, object]:
    """Return slices + inclusive endpoint metadata for one sample."""
    inp = input_slice(t, delay, delta1)
    fut = future_slice(t, delta2)
    hist = label_history_slice(t, delta1)
    last_input = t - delay
    return {
        "t": int(t),
        "delay": int(delay),
        "delta1": int(delta1),
        "delta2": int(delta2),
        "input": inp,
        "future": fut,
        "label_history": hist,
        "last_input_index": last_input,
        "bot_time": int(t),
    }


def signature_fields(config: dict) -> dict:
    """
    Extra key_params for generate_dataset_signature.

    Omit telepred_delay when 0 and definition_id when v1 so existing
    TelePred(0) signatures stay stable.
    """
    out = {}
    d = resolve_delay(config)
    if d != 0:
        out["telepred_delay"] = d
    def_id = resolve_definition_id(config)
    if def_id != DEFINITION_ID_V1:
        out["telepred_definition_id"] = def_id
    return out
