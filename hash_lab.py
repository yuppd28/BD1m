
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import argparse, json

M = 5000
EMPTY = None
TOMBSTONE = object()



def hash_func(key: int, m: int = M) -> int:

    return ((key * 2654435761) & 0xFFFFFFFF) % m


@dataclass
class Flight:
    flight_id: int  # ключ
    route: str
    dep_time: str
    airline_id: int
    plane_id: int
    from_airport: str
    to_airport: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flight_id": self.flight_id,
            "route": self.route,
            "dep_time": self.dep_time,
            "airline_id": self.airline_id,
            "plane_id": self. plane_id,
            "from_airport": self.from_airport,
            "to_airport": self.to_airport,
        }

    @staticmethod
    def from_insert_fields(fields: List[str]) -> "Flight":
        # fields: [id, route, dep_time, airline_id, plane_id, from, to]
        if len(fields) < 7:
            raise ValueError("INSERT requires 7 fields: id,route,dep_time,airline_id,plane_id,from,to")
        return Flight(
            flight_id=int(fields[0]),
            route=fields[1],
            dep_time=fields[2],
            airline_id=int(fields[3]),
            plane_id=int(fields[4]),
            from_airport=fields[5],
            to_airport=fields[6],
        )


class HashFile:
    def __init__(self, m: int = M):
        self.m = m
        self.slots: List[Any] = [EMPTY] * m
        self.count = 0  # кількість активних записів

    def load_factor(self) -> float:
        return self.count / self.m

    def _probe(self, key: int) -> Tuple[int, bool]:

        start = idx = hash_func(key, self.m)
        first_tombstone = None

        while True:
            slot = self.slots[idx]

            if slot is EMPTY:
                return (first_tombstone if first_tombstone is not None else idx, False)

            if slot is TOMBSTONE:
                if first_tombstone is None:
                    first_tombstone = idx
            else:
                # OCCUPIED
                if slot.flight_id == key:
                    return (idx, True)

            idx = (idx + 1) % self.m
            if idx == start:
                raise RuntimeError("Hash file full")

    def insert(self, rec: Flight) -> bool:
        idx, found = self._probe(rec.flight_id)
        if found:

            self.slots[idx] = rec
            return False
        else:
            if self.slots[idx] is EMPTY or self.slots[idx] is TOMBSTONE:
                self.slots[idx] = rec
                self.count += 1
                return True
            raise RuntimeError("Unexpected state in insert")

    def select(self, key: int) -> Optional[Flight]:
        idx, found = self._probe(key)
        return self.slots[idx] if found else None

    def delete(self, key: int) -> bool:
        idx, found = self._probe(key)
        if found:
            self.slots[idx] = TOMBSTONE
            self.count -= 1
            return True
        return False

    def occupancy_array(self) -> List[int]:

        return [1 if (s is not EMPTY and s is not TOMBSTONE) else 0 for s in self.slots]

    def dense_zones_by_runs(self, min_run: int = 20) -> List[Tuple[int, int, int]]:

        occ = self.occupancy_array()
        zones: List[Tuple[int, int, int]] = []
        i, n = 0, len(occ)
        while i < n:
            if occ[i] == 1:
                j = i
                while j < n and occ[j] == 1:
                    j += 1
                run_len = j - i
                if run_len >= min_run:
                    zones.append((i, j - 1, run_len))
                i = j
            else:
                i += 1
        return zones

    def dense_zones_by_window(self, window: int = 50, threshold: float = 0.8) -> List[Tuple[int, int, float]]:

        occ = self.occupancy_array()
        n = len(occ)
        if window > n:
            window = n

        pref = [0]
        for v in occ:
            pref.append(pref[-1] + v)

        active = []
        for start in range(0, n - window + 1):
            ones = pref[start + window] - pref[start]
            ratio = ones / window
            if ratio >= threshold:
                active.append((start, start + window - 1, ratio))

        merged: List[Tuple[int, int, float]] = []
        for st, en, r in active:
            if not merged or st > merged[-1][1] + 1:
                merged.append([st, en, r])
            else:
                merged[-1][1] = max(merged[-1][1], en)
                merged[-1][2] = max(merged[-1][2], r)
        return [(a, b, d) for a, b, d in merged]


def parse_and_execute(commands_path: str, hf: HashFile) -> Dict[str, Any]:

    ops_log = []
    with open(commands_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("/")]
            cmd = parts[0].lower()

            if cmd not in {"insert", "ins", "select", "sel", "delete", "del"}:
                ops_log.append({"line": lineno, "error": f"Unknown command '{cmd}'"})
                continue

            if len(parts) < 2:
                ops_log.append({"line": lineno, "error": "Missing key"})
                continue

            key = int(parts[1])

            if cmd in {"insert", "ins"}:
                try:
                    rec = Flight.from_insert_fields(parts[1:8])  # включно з key у parts[1]
                except Exception as e:
                    ops_log.append({"line": lineno, "error": f"INSERT parse: {e}"})
                    continue
                new = hf.insert(rec)
                ops_log.append({"line": lineno, "op": "insert", "key": key, "result": "new" if new else "updated"})

            elif cmd in {"select", "sel"}:
                rec = hf.select(key)
                ops_log.append({"line": lineno, "op": "select", "key": key, "found": rec is not None})

            elif cmd in {"delete", "del"}:
                ok = hf.delete(key)
                ops_log.append({"line": lineno, "op": "delete", "key": key, "deleted": ok})

    return {
        "log": ops_log,
        "n_ops": len(ops_log),
        "active": hf.count,
        "m": hf.m,
        "alpha": round(hf.load_factor(), 4),
    }


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser(description="Hash-file lab runner (Flights domain)")
    parser.add_argument("--commands", required=True, help="Path to commands .txt")
    parser.add_argument("--plot", help="Path to save occupancy plot (PNG)")
    parser.add_argument("--runs_min", type=int, default=20, help="Min run length for dense zones (run-based)")
    parser.add_argument("--win", type=int, default=50, help="Window size for density-based zones")
    parser.add_argument("--thr", type=float, default=0.8, help="Density threshold for window-based zones")
    args = parser.parse_args()

    hf = HashFile(M)
    stats = parse_and_execute(args.commands, hf)

    zones_runs = hf.dense_zones_by_runs(min_run=args.runs_min)
    zones_windows = hf.dense_zones_by_window(window=args.win, threshold=args.thr)

    out = {
        "stats": stats,
        "zones_runs": [{"start": a, "end": b, "length": l} for a, b, l in zones_runs],
        "zones_windows": [{"start": a, "end": b, "density_max": round(r, 3)} for a, b, r in zones_windows],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.plot:

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("WARN: matplotlib не встановлено. Встановіть: pip install matplotlib", flush=True)
            return

        occ = hf.occupancy_array()
        plt.figure(figsize=(10, 3))
        plt.plot(occ, linewidth=0.8)
        plt.title("Occupancy by slot index (1=occupied, 0=empty/tombstone)")
        plt.xlabel("Slot index")
        plt.ylabel("Occupied?")
        # підсвічуємо зони за run-based критерієм
        for a, b, _ in zones_runs:
            plt.axvspan(a, b, alpha=0.2)
        plt.tight_layout()
        plt.savefig(args.plot)


if __name__ == "__main__":
    main()
