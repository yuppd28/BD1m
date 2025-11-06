
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

KEYSPACE = 100_000       # простір ключів (ID рейсу)
M = 5_000                # розмір файлу (к-сть слотів)
EMPTY = None
TOMBSTONE = object()

def hash_func(key: int, m: int = M) -> int:
    return ((key * 2654435761) & 0xFFFFFFFF) % m

@dataclass
class Flight:
    flight_id: int
    route: str
    dep_time: str
    airline_id: int
    plane_id: int
    from_airport: str
    to_airport: str

    @staticmethod
    def from_insert_fields(fields: List[str]) -> "Flight":
        # fields = [id, route, dep_time, airline_id, plane_id, from, to]
        if len(fields) < 7:
            raise ValueError("INSERT потребує 7 полів: id,route,dep_time,airline_id,plane_id,from,to")
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
        self.count = 0

    def load_factor(self) -> float:
        return self.count / self.m

    def _probe(self, key: int) -> Tuple[int, bool]:
        start = idx = hash_func(key, self.m)
        first_tomb = None
        while True:
            s = self.slots[idx]
            if s is EMPTY:
                return (first_tomb if first_tomb is not None else idx, False)
            if s is TOMBSTONE:
                if first_tomb is None:
                    first_tomb = idx
            else:
                if s.flight_id == key:
                    return (idx, True)
            idx = (idx + 1) % self.m
            if idx == start:
                raise RuntimeError("Hash file full")

    def insert(self, rec: Flight) -> bool:
        idx, found = self._probe(rec.flight_id)
        if found:
            self.slots[idx] = rec
            return False
        self.slots[idx] = rec
        self.count += 1
        return True

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
        # 1 — зайнято; 0 — порожньо або tombstone
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
                L = j - i
                if L >= min_run:
                    zones.append((i, j - 1, L))
                i = j
            else:
                i += 1
        return zones

    def dense_zones_by_window(self, window: int = 50, threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        occ = self.occupancy_array()
        n = len(occ)
        if window > n: window = n
        pref = [0]
        for v in occ: pref.append(pref[-1] + v)
        active = []
        for s in range(0, n - window + 1):
            ones = pref[s + window] - pref[s]
            ratio = ones / window
            if ratio >= threshold:
                active.append((s, s + window - 1, ratio))
        merged: List[Tuple[int, int, float]] = []
        for st, en, r in active:
            if not merged or st > merged[-1][1] + 1:
                merged.append([st, en, r])
            else:
                merged[-1][1] = max(merged[-1][1], en)
                merged[-1][2] = max(merged[-1][2], r)
        return [(a, b, float(r)) for a, b, r in merged]

def parse_and_execute(text: str, hf: HashFile) -> Dict[str, Any]:
    log = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("/")]
        cmd = parts[0].lower()
        if cmd not in {"insert", "ins", "select", "sel", "delete", "del"}:
            log.append({"line": lineno, "error": f"Unknown command '{cmd}'"})
            continue
        if len(parts) < 2:
            log.append({"line": lineno, "error": "Missing key"})
            continue

        key = int(parts[1])

        if cmd in {"insert", "ins"}:
            try:
                rec = Flight.from_insert_fields(parts[1:8])
            except Exception as e:
                log.append({"line": lineno, "error": f"INSERT parse: {e}"})
                continue
            new = hf.insert(rec)
            log.append({"line": lineno, "op": "insert", "key": key, "result": "new" if new else "updated"})

        elif cmd in {"select", "sel"}:
            found = hf.select(key) is not None
            log.append({"line": lineno, "op": "select", "key": key, "found": found})

        elif cmd in {"delete", "del"}:
            deleted = hf.delete(key)
            log.append({"line": lineno, "op": "delete", "key": key, "deleted": deleted})

    return {"log": log, "n_ops": len(log), "active": hf.count, "m": hf.m, "alpha": hf.load_factor()}

CITIES = ["KBP->LWO","KBP->ODS","KBP->DNK","LWO->KBP","ODS->KBP","KBP->AMS","KBP->WAW","IEV->KBP"]
MINUTES = [0,10,20,30,40,50]

def rroute(): return random.choice(CITIES)
def rtime():
    h = random.randint(0, 23); m = random.choice(MINUTES)
    return f"{h:02d}:{m:02d}"

def generate_commands(n_inserts=1000, n_selects=120, n_deletes=60, seed=42) -> str:
    random.seed(seed)
    keys = random.sample(range(KEYSPACE), n_inserts)
    lines = []
    for k in keys:
        route = rroute(); dep = rtime()
        airline = random.randint(1, 40); plane = random.randint(100, 999)
        frm, to = route.split("->")
        lines.append(f"insert/{k}/{route}/{dep}/{airline}/{plane}/{frm}/{to}")
    for _ in range(n_selects):
        lines.append(f"select/{random.choice(keys)}")
    for k in random.sample(keys, min(n_deletes, len(keys))):
        lines.append(f"delete/{k}")
    random.shuffle(lines)
    return "\n".join(lines)


def expected_collisions_linear_probing(n: int, m: int) -> float:

    return n * (n - 1) / (2 * m)

st.set_page_config(page_title="Hash-File — Рейси (ER)", layout="wide")
st.title("Хеш-файл для сутності «Рейс». Графічний аналіз заповнення")

with st.sidebar:
    st.header("Параметри зон")
    runs_min = st.number_input("Мін. довжина суцільного кластера (runs)", 1, 200, 12, step=1)
    win = st.number_input("Розмір ковзного вікна", 5, 500, 40, step=1)
    thr = st.slider("Поріг щільності у вікні", 0.10, 1.00, 0.70, 0.01)

    st.markdown("---")
    st.subheader("Дані")
    mode = st.radio("Джерело команд", ["Завантажити файл", "Згенерувати тест"], index=1)

    if mode == "Згенерувати тест":
        n_ins = st.slider("К-сть INSERT", 100, 4000, 1200, 50)
        n_sel = st.slider("К-сть SELECT", 0, 500, 120, 10)
        n_del = st.slider("К-сть DELETE", 0, 500, 60, 10)
        seed = st.number_input("Seed", 0, 10000, 42)
        if st.button("Згенерувати"):
            st.session_state["commands_editor"] = generate_commands(n_ins, n_sel, n_del, seed)
    else:
        up = st.file_uploader("Завантажте .txt з командами", type=["txt"])
        if up is not None:
            st.session_state["commands_editor"] = up.read().decode("utf-8", errors="ignore")

default_text = (
    "# формат: команда/ключ/інші поля (для insert)\n"
    "# insert/<id>/<route>/<time>/<airline>/<plane>/<from>/<to>\n"
    "insert/91506/KBP->LWO/09:20/5/301/KBP/LWO\n"
    "insert/48217/KBP->ODS/10:40/7/742/KBP/ODS\n"
    "select/91506\n"
    "delete/48217\n"
)

if "commands_editor" not in st.session_state:
    st.session_state["commands_editor"] = default_text

st.subheader("Файл команд")
st.caption("Підтримуються: insert, select, delete (або ins/sel/del).")
st.text_area("Редагуйте або вставляйте свої команди:", key="commands_editor", height=220)

if st.button("Виконати аналіз", type="primary"):
    text = st.session_state["commands_editor"].strip()
    if not text:
        st.error("Немає даних для аналізу. Додайте команди або згенеруйте тест.")
    else:
        hf = HashFile(M)
        stats = parse_and_execute(text, hf)

        st.markdown("### Теоретичні параметри")
        n_theory = 1000
        alpha_theory = n_theory / M
        exp_coll = expected_collisions_linear_probing(n_theory, M)
        tdf = pd.DataFrame({
            "Параметр": [
                "Простір ключів",
                "Розмір файлу (слотів)",
                "К-сть реальних записів (теорія)",
                "Коеф. заповнення α (теорія)",
                "Очікувані колізії (≈)"
            ],
            "Значення": [f"{KEYSPACE:,}", f"{M}", f"{n_theory}", f"{alpha_theory:.3f}", f"{exp_coll:.0f}"]
        })
        st.table(tdf)

        st.markdown("### Фактичний результат виконання")
        sdf = pd.DataFrame({
            "Метрика": ["Активних записів", "Розмір файлу", "α фактична", "К-сть операцій"],
            "Значення": [stats["active"], stats["m"], f"{stats['alpha']:.4f}", stats["n_ops"]],
        })
        st.table(sdf)


        occ = hf.occupancy_array()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(range(len(occ)), occ, linewidth=1.0)
        ax.set_title("Розподіл заповнення слотів у хеш-файлі", fontsize=14, pad=10)
        ax.set_xlabel("Номер слота у файлі", fontsize=12)
        ax.set_ylabel("Стан комірки", fontsize=12)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0 – порожньо", "1 – зайнято"])
        ax.grid(True, linestyle="--", alpha=0.35)

        zones_runs = hf.dense_zones_by_runs(min_run=int(runs_min))
        for a, b, _L in zones_runs:
            ax.axvspan(a, b, alpha=0.20)

        st.pyplot(fig, clear_figure=True)


        if zones_runs:
            st.markdown("### Зони щільного заповнення (суцільні пробіги)")
            zdf = pd.DataFrame([{"start": a, "end": b, "length": L} for a, b, L in zones_runs])
            st.dataframe(zdf, use_container_width=True)
        else:
            st.info("За run-критерієм зони не знайдені. Зменште поріг або збільшіть кількість INSERT (α).")

        zones_win = hf.dense_zones_by_window(window=int(win), threshold=float(thr))
        if zones_win:
            st.markdown("### Зони щільного заповнення (ковзне вікно)")
            zwdf = pd.DataFrame([{"start": a, "end": b, "density_max": round(r, 3)} for a, b, r in zones_win])
            st.dataframe(zwdf, use_container_width=True)
        else:
            st.info("За віконним критерієм зони не знайдені. Спробуйте зменшити поріг щільності або збільшити α.")

