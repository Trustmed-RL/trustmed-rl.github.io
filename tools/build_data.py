"""Build demo/data.js + demo/img/ from real evaluation trajectories.

Usage: python demo/tools/build_data.py   (run from the repo root)
"""
import json, os, re, shutil, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO = os.path.join(ROOT, "demo")
RUN = os.path.join(ROOT, "ICLR_Submission", "Eval", "runs", "baselines_v2bank",
                   "qwen3-vl-8b-gigpo-seed2-v7", "pass0")
IMG = os.path.join(ROOT, "ICLR_Submission", "Image")
MAIN_PMCID, SWAP_PMCID = "PMC2940095", "PMC4253624"


def find(path, pmcid):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if pmcid in line[:4000]:
                d = json.loads(line)
                if d.get("meta", {}).get("pmcid") == pmcid:
                    return d
    raise SystemExit(f"{pmcid} not found in {path}")


def local(path):
    """Map a recorded absolute path onto this checkout."""
    rel = re.split(r"ICLR_Submission[\\/]", path, maxsplit=1)[1].replace("\\", "/")
    return os.path.join(ROOT, "ICLR_Submission", rel)


def copy_img(src, name):
    os.makedirs(os.path.join(DEMO, "img"), exist_ok=True)
    shutil.copyfile(src, os.path.join(DEMO, "img", name))
    return "img/" + name


def asset_file(asset_id):
    pmcid, rest = asset_id.split("_img_")
    tag = rest.split("#")[1]
    hits = [f for f in os.listdir(os.path.join(IMG, pmcid)) if f.endswith(f"__{tag}.jpg")]
    if not hits:
        raise SystemExit(f"no crop for {asset_id}")
    return os.path.join(IMG, pmcid, hits[0])


def parse_results(obs):
    rows = []
    for line in obs.split("\n")[1:]:
        line = line.strip().lstrip("- ").strip()
        if not line:
            continue
        head, _, body = line.partition(": ")
        cat, _, name = head.partition(" — ")
        img = re.search(r"\[image (IMG-\d+) attached", body)
        rows.append({"cat": cat.strip(), "name": name.strip(),
                     "img": img.group(1) if img else None,
                     "text": None if img else body.strip()})
    return rows


def parse_gate(think):
    out = []
    for m in re.finditer(r"(IMG-\d+):\s*\n(.*?)(?=\nIMG-\d+:|\Z)", think, re.S):
        block = m.group(2)
        mod = re.search(r"Modality[^:\n]*:\s*(.+)", block)
        site = re.search(r"Anatomical site[^:\n]*:\s*(.+)", block)
        out.append({"img": m.group(1),
                    "modality": mod.group(1).strip() if mod else "",
                    "site": site.group(1).strip() if site else ""})
    return out


def parse_search(obs):
    hits = []
    body = obs.replace("<information>", "").replace("</information>", "")
    pattern = r"\[(E\d+) \| ([^\]]+)\] ([^\n]*)\n(.*?)(?=\n\s*\n\[E\d+ \||\Z)"
    for m in re.finditer(pattern, body, re.S):
        text = m.group(4)
        url = re.search(r"URL: (\S+)", text)
        text = re.sub(r"\nURL: \S+", "", text).strip()
        hits.append({"id": m.group(1), "source": m.group(2).split(" ")[0],
                     "title": m.group(3).strip(),
                     "snippet": text[:260].rsplit(" ", 1)[0] + "…",
                     "url": url.group(1) if url else None})
    return hits


def build_main():
    d = find(os.path.join(RUN, "clean", "trajectories.jsonl"), MAIN_PMCID)
    alias = d["budgets"]["asset_aliases"]
    images = {}
    for asset, name in alias.items():
        images[name] = copy_img(asset_file(asset), f"case-{name.lower()}.jpg")
    turns = []
    for t in d["turns"]:
        o = {"n": t["turn"], "phase": t["phase"], "verb": t["verb"],
             "payload": t.get("payload"), "think": (t.get("think") or "").strip()}
        obs = t.get("observation") or ""
        if t.get("patient"):
            o["patient"] = t["patient"]
        if t["verb"] in ("physical_examination", "order_tests"):
            o["results"] = parse_results(obs)
        elif t["verb"] == "gate_verdict":
            g = t["gate"]
            o["gate"] = {"images": [alias[a] for a in g["assets"]], "verdict": g["verdict"],
                         "reward": g["reward"], "checks": parse_gate(o["think"])}
        elif t["verb"] == "consult_specialist":
            o["report"] = [ln.lstrip("- ").strip() for ln in obs.split("\n")[1:] if ln.strip()]
        elif t["verb"] == "search":
            o["hits"] = parse_search(obs)
        elif obs:
            o["note"] = obs
        turns.append(o)
    m = d["metrics"]
    panel = m["panel_v2s55"]
    onto = panel["ontology"]
    return {
        "pmcid": MAIN_PMCID, "doctor": "TrustMed-RL (Qwen3-VL-8B, GiGPO)",
        "patientModel": d["models"]["patient"],
        "opening": d["opening"], "turns": turns, "images": images,
        "facts": d["deck_final"]["known_facts"],
        "budgets": {"asks": d["budgets"]["max_asks"], "workup": d["budgets"]["max_workup_turns"],
                    "searches": d["budgets"]["max_searches"]},
        "metrics": {"factsTotal": m["facts_total"], "factsRevealed": m["facts_revealed"],
                    "asks": m["asks_used"], "workupTurns": m["workup_turns"]},
        "judge": {"gold": onto["a"], "pred": onto["b"], "verdict": onto["verdict"],
                  "reward": panel["fused_t_reward"], "judge": panel["judges_present"][0],
                  "relations": onto["relations"], "direction": onto["direction"],
                  "labels": {k: {"a": v["a"].get("label") or v["a"].get("title"),
                                 "b": v["b"].get("label") or v["b"].get("title")}
                             for k, v in onto["per_source"].items()}},
    }


def build_swap():
    d = find(os.path.join(RUN, "mm_anatomy", "trajectories.jsonl"), SWAP_PMCID)
    c = d["corruption"]
    gate_turn = next(t for t in d["turns"] if t["verb"] == "gate_verdict")
    order_turn = d["turns"][d["turns"].index(gate_turn) - 1]
    rows = [r for r in parse_results(order_turn["observation"]) if r["img"]]
    alias = d["budgets"]["asset_aliases"]
    imgs = []
    for asset in gate_turn["gate"]["assets"]:
        swapped = asset == c["asset_id"]
        src = local(c["swap_path"]) if swapped else asset_file(asset)
        name = alias[asset]
        imgs.append({"img": name, "swapped": swapped,
                     "src": copy_img(src, f"swap-{name.lower()}{os.path.splitext(src)[1]}"),
                     "ordered": next(r["name"] for r in rows if r["img"] == name)})
    return {
        "pmcid": SWAP_PMCID, "donor": c["swap_pmcid"],
        "patient": order_turn["observation"].split("\n")[0],
        "order": order_turn["payload"], "images": imgs,
        "original": copy_img(local(c["original_path"]), "swap-original.jpg"),
        "victim": c["victim_key"], "donorKey": c["donor_key"],
        "verdict": gate_turn["gate"]["verdict"], "axes": gate_turn["gate"]["axes"],
        "reward": gate_turn["gate"]["reward"], "think": gate_turn["think"].strip(),
        "action": f"gate_verdict: {gate_turn['payload']}",
        "outcome": d["metrics"].get("final_answer"),
    }


if __name__ == "__main__":
    data = {"episode": build_main(), "swap": build_swap()}
    with open(os.path.join(DEMO, "data.js"), "w", encoding="utf-8") as f:
        f.write("// Generated by tools/build_data.py from real evaluation trajectories. Do not edit.\n")
        f.write("window.TRUSTMED = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n")
    sys.stdout.reconfigure(encoding="utf-8")
    e, sw = data["episode"], data["swap"]
    print("turns", len(e["turns"]), "images", len(e["images"]), "facts", len(e["facts"]))
    for t in e["turns"]:
        extra = {k: (len(v) if isinstance(v, list) else v) for k, v in t.items()
                 if k in ("results", "report", "hits", "note")}
        print(" ", t["n"], t["phase"], t["verb"], extra, t.get("gate", {}).get("checks", ""))
    print("judge", e["judge"])
    print("swap", {k: v for k, v in sw.items() if k != "think"})
