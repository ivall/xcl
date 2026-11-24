"""Summarize observed D3D12.X slot stubs and unknown IIDs, without running a title.

python scripts/triage_d3d12.py report.json [--log xcl.log] [--json]
Supply files from the same run. First-call log lines never increase report counts.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import uuid


DLLS = {"d3d12_xs.dll", "d3d12_x.dll"}
SLOT = re.compile(r"(I[A-Za-z0-9_]+)::#([0-9]{1,10})(?: \(([^()\r\n]+)\))?")
STUB_LOG = re.compile(r"\[STUB\] ([^!\s]+)!(.*?) first called from (.*?) (?:→|->) returning 0x[0-9a-fA-F]+\s*$")
IID_LOG = re.compile(r"D3D12\.X: QueryInterface\(([^)]+)\) (?:–|-) interface unknown on the PC \(console-only\?\)\s*$")


def analyze(report, lines=()):
    """Return deterministic evidence, not a diagnosis or a compatibility verdict."""
    slots, iids, warnings = {}, {}, set()

    def slot(dll, function, calls, caller, source):
        if not isinstance(dll, str):
            warnings.add("Invalid called_stubs DLL; evidence may be incomplete.")
            return
        dll = dll.lower()
        if dll not in DLLS:
            return
        if not isinstance(function, str):
            warnings.add("Invalid D3D12.X stub name; evidence may be incomplete.")
            return
        match = SLOT.fullmatch(function)
        if not match:
            if "::#" in function:
                warnings.add("Unrecognized D3D12.X slot name; evidence may be incomplete.")
            return
        if source == "report" and (type(calls) is not int or not 0 <= calls <= 2**64 - 1):
            warnings.add("Invalid D3D12.X slot call count; evidence may be incomplete.")
            return
        if source == "report" and calls == 0:
            return
        if not isinstance(caller, str):
            warnings.add("Invalid D3D12.X first caller; retaining slot without caller.")
            caller = ""
        interface, number, variant = match.groups()
        key = (dll, interface, int(number), variant or "")
        entry = slots.setdefault(key, dict(dll=dll, interface=interface, slot=int(number),
                                         variant=variant or "", calls=None, first_callers=set(), sources=set()))
        if source == "report":
            entry["calls"] = (entry["calls"] or 0) + calls
        if caller:
            entry["first_callers"].add(caller)
        entry["sources"].add(source)

    def iid(value, source):
        try:
            if not isinstance(value, str):
                raise ValueError()
            canonical = str(uuid.UUID(value))
        except ValueError:
            warnings.add("Invalid unknown IID; evidence may be incomplete.")
            return
        iids.setdefault(canonical, set()).add(source)

    if report is None:
        warnings.add("No usable report; only first-call log evidence is available and call totals are unknown.")
    else:
        if not isinstance(report, dict):
            raise ValueError("report must be a JSON object")
        records = report.get("called_stubs")
        if not isinstance(records, list):
            warnings.add("Missing or invalid called_stubs; slot evidence may be incomplete.")
        else:
            for record in records:
                if not isinstance(record, dict):
                    warnings.add("Invalid called_stubs entry; evidence may be incomplete.")
                    continue
                slot(record.get("dll"), record.get("function"), record.get("calls"),
                     record.get("first_caller", ""), "report")
        d3d12 = report.get("d3d12")
        unknown = d3d12.get("unknown_iids") if isinstance(d3d12, dict) else None
        if not isinstance(unknown, list):
            warnings.add("Missing or invalid d3d12.unknown_iids; IID evidence may be incomplete.")
        else:
            for value in unknown:
                iid(value, "report")

    for line in lines:
        if "\ufffd" in line:
            warnings.add("Log contains replacement characters; evidence may be incomplete.")
        match = STUB_LOG.search(line)
        if match:
            dll, function, caller = match.groups()
            slot(dll, function, None, caller, "log")
        elif "[STUB]" in line and any(dll in line.lower() for dll in DLLS) and "::#" in line:
            warnings.add("Unrecognized D3D12.X slot log entry; evidence may be incomplete.")
        match = IID_LOG.search(line)
        if match:
            iid(match[1], "log")
        elif "D3D12.X: QueryInterface(" in line and "interface unknown" in line:
            warnings.add("Unrecognized unknown-IID log entry; evidence may be incomplete.")

    for entry in slots.values():
        entry["sources"] = sorted(entry["sources"])
        entry["first_callers"] = sorted(entry["first_callers"])
    return dict(schema_version=1, slots=[slots[key] for key in sorted(slots)],
                unknown_iids=[dict(iid=key, sources=sorted(iids[key])) for key in sorted(iids)],
                warnings=sorted(warnings))


def render(result):
    out = ["D3D12.X observations (not proven crash causes)"]
    for entry in result["slots"]:
        count = str(entry["calls"]) if entry["calls"] is not None else "unknown"
        variant = f" ({entry['variant']})" if entry["variant"] else ""
        out.append(f"{entry['dll']}!{entry['interface']}::#{entry['slot']}{variant}: "
                   f"calls={count}; sources={','.join(entry['sources'])}")
        out.extend(f"  first caller: {json.dumps(caller, ensure_ascii=True)}" for caller in entry["first_callers"])
    out.extend(f"unknown IID {entry['iid']}; sources={','.join(entry['sources'])}"
               for entry in result["unknown_iids"])
    if not result["slots"] and not result["unknown_iids"]:
        out.append("No matching observations found; this does not establish interface support.")
    out.append("Next: verify the caller and console layout before implementing a slot; IIDs may be optional probes.")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--log", type=Path, help="matching run's UTF-8 log, also used if the report is unusable")
    parser.add_argument("--json", action="store_true", help="machine-readable evidence on stdout")
    args = parser.parse_args(argv)
    warnings, report = [], None
    try:
        report = json.loads(args.report.read_text(encoding="utf-8-sig"))
        if not isinstance(report, dict):
            raise ValueError("report must be a JSON object")
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        warnings.append(f"Cannot use report: {error}")
        report = None
    result = None
    if args.log:
        try:
            with args.log.open(encoding="utf-8-sig", errors="replace") as lines:
                result = analyze(report, lines)
        except OSError as error:
            warnings.append(f"Cannot use log: {error}")
    if result is None:
        if report is None:
            for warning in warnings:
                print(f"warning: {warning}", file=sys.stderr)
            print("error: no usable report or log", file=sys.stderr)
            return 2
        result = analyze(report)
    result["warnings"] = sorted(set(result["warnings"] + warnings))
    for warning in result["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    print(json.dumps(result, indent=2, ensure_ascii=True) if args.json else render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
