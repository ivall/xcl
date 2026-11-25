"""Compare two runs of the compile-time benchmark (`compile_bench.json` from
`scripts/compile_bench.ps1`), e.g. `main` against a pull request's head, program by program.

    python -I scripts/compile_bench_compare.py <base.json> <head.json> [--noise 0.05]

Prints a Markdown table: instructions, HLSL bytes, DXIL bytes and the median DXC time of each
program in both runs, with the head / base ratio, and the geometric mean of the ratios. The sizes
are deterministic for one DXC; the times are only comparable when both runs come from the same
machine, build and DXC, which the environment check at the top says. A size change is marked
whatever its size; a time change only beyond the run-to-run noise (`--noise`, a fraction). It
reports, it does not
judge: the exit status is 0 whatever the numbers are (2 for unreadable input).
"""

import json
import math
import sys

# what the times depend on: a difference here makes them not comparable
SAME_FOR_TIMES = ("profile", "cpu", "logical_cpus", "set", "reps")

# the run-to-run spread of a program's median DXC time on one machine, with margin: on a
# Ryzen 5 3600 the small set's medians moved by at most 1.9 % over 5 runs, and the 3 compiles
# of one full-set run by 1.4–2.8 % (median), single compiles by up to 17 %
TIME_NOISE = 0.05


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc["environment"], {p["name"]: p for p in doc["programs"]}


def dxc_version(env):
    return ((env.get("dxc") or {}).get("dxcompiler") or {}).get("file_version", "")


def environment_notes(base, head):
    notes = []
    for side, env in (("base", base), ("head", head)):
        if env.get("dirty"):
            notes.append(f"{side} was measured with uncommitted changes")
    if dxc_version(base) != dxc_version(head):
        notes.append(f"different DXC: {dxc_version(base) or '?'} / {dxc_version(head) or '?'} (sizes and times not comparable)")
    for key in SAME_FOR_TIMES:
        if base.get(key) != head.get(key):
            notes.append(f"different {key}: {base.get(key)!r} / {head.get(key)!r} (times not comparable)")
    return notes


def ratio(a, b):
    return b / a if a else math.nan


def geomean(values):
    values = [v for v in values if v > 0 and not math.isnan(v)]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else math.nan


def changed(row, noise):
    """"size" when the HLSL or DXIL size differs, else "time" when the DXC time moved by more
    than `noise` either way, else ""."""
    if row["hlsl"][0] != row["hlsl"][1] or row["dxil"][0] != row["dxil"][1]:
        return "size"
    r = row["dxc"][2]
    return "time" if not math.isnan(r) and (r > 1 + noise or r < 1 / (1 + noise)) else ""


def compare(base, head, noise=TIME_NOISE):
    """{rows, only_base, only_head, means, size_changes, time_changes, noise} for two
    {name: program} maps."""
    rows = []
    for name in sorted(set(base) & set(head)):
        b, h = base[name], head[name]
        rows.append({
            "name": name,
            "instructions": (b["instructions"], h["instructions"]),
            "hlsl": (b["hlsl_bytes"], h["hlsl_bytes"], ratio(b["hlsl_bytes"], h["hlsl_bytes"])),
            "dxil": (b["dxil_bytes"], h["dxil_bytes"], ratio(b["dxil_bytes"], h["dxil_bytes"])),
            "dxc": (b["dxc_ms_median"], h["dxc_ms_median"], ratio(b["dxc_ms_median"], h["dxc_ms_median"])),
        })
    means = {k: geomean([r[k][2] for r in rows]) for k in ("hlsl", "dxil", "dxc")}
    for r in rows:
        r["changed"] = changed(r, noise)
    size_changes = sum(1 for r in rows if r["changed"] == "size")
    return {
        "rows": rows,
        "only_base": sorted(set(base) - set(head)),
        "only_head": sorted(set(head) - set(base)),
        "means": means,
        "size_changes": size_changes,
        "time_changes": sum(1 for r in rows if r["changed"] == "time"),
        "noise": noise,
    }


def fmt_ratio(r):
    return "–" if math.isnan(r) else f"{r:.3f}"


def render(base_env, head_env, result):
    out = [
        f"base `{base_env.get('commit', '')[:10]}` / head `{head_env.get('commit', '')[:10]}`, "
        f"set {head_env.get('set')}, DXC {dxc_version(head_env) or '?'}, {head_env.get('profile')}",
        "",
    ]
    notes = environment_notes(base_env, head_env)
    out += [f"- note: {note}" for note in notes] + ([""] if notes else [])
    out += [
        "| program | instructions | HLSL bytes | × | DXIL bytes | × | DXC ms (median) | × | changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in result["rows"]:
        (ib, ih), (hb, hh, hr), (db, dh, dr), (tb, th, tr) = r["instructions"], r["hlsl"], r["dxil"], r["dxc"]
        insts = str(ib) if ib == ih else f"{ib} → {ih}"
        out.append(f"| {r['name']} | {insts} | {hb} → {hh} | {fmt_ratio(hr)} | {db} → {dh} | {fmt_ratio(dr)} | {tb:.1f} → {th:.1f} | {fmt_ratio(tr)} | {r['changed']} |")
    m = result["means"]
    out += [
        "",
        f"geometric mean head / base over {len(result['rows'])} programs: HLSL {fmt_ratio(m['hlsl'])}, "
        f"DXIL {fmt_ratio(m['dxil'])}, DXC time {fmt_ratio(m['dxc'])}; "
        f"programs whose HLSL or DXIL size changed: {result['size_changes']}; "
        f"whose DXC time alone moved by more than {result['noise'] * 100:g} % (the noise): {result['time_changes']}",
    ]
    for side in ("only_base", "only_head"):
        if result[side]:
            out.append(f"only in {side[5:]}: {', '.join(result[side])}")
    return "\n".join(out)


def main(argv):
    noise = TIME_NOISE
    if len(argv) == 4 and argv[2] == "--noise":
        try:
            noise = float(argv[3])
        except ValueError:
            noise = -1.0
        argv = argv[:2]
        if not 0 <= noise < 10:
            print("compile_bench_compare: --noise takes a fraction, e.g. 0.05", file=sys.stderr)
            return 2
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[3].strip(), file=sys.stderr)
        return 2
    try:
        (base_env, base), (head_env, head) = load(argv[0]), load(argv[1])
    except (OSError, ValueError, KeyError) as e:
        print(f"compile_bench_compare: {e}", file=sys.stderr)
        return 2
    print(render(base_env, head_env, compare(base, head, noise)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
