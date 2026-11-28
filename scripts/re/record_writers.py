"""Find the D3D12.X inline command-list record writers in XboxBvhBuilder.dll (objdump AT&T
listing): `mov 0x18(%L),%P ; cmp 0x30|0x28(%L),%P ; jae fallback` … `add $N,%P ; mov %P,0x18(%L)`.
Per site: list register, bound field, record length, header (immediate or list field) and
the vtable slot of the fallback call. Branch targets may have or omit the 0x prefix.

usage: objdump -d --no-show-raw-insn XboxBvhBuilder.dll > bvhb.asm; python record_writers.py [-h] <bvhb.asm> [<bvhb.asm> ...]"""
import collections
import os
import re
import sys


def find_record_writers(lines):
    """Find record writer sites in objdump AT&T disassembly lines.

    Returns a list of tuples:
      (address, list_reg, bound, length, header, fallback_slot)
    """
    ins = []
    for l in lines:
        m = re.match(r'\s+([0-9a-f]+):\s+(.*)$', l)
        if m:
            ins.append((int(m.group(1), 16), m.group(2).strip()))
    idx = {a: i for i, (a, _) in enumerate(ins)}
    sites = []
    for i, (a, t) in enumerate(ins):
        m = re.match(r'mov\s+0x18\(%(\w+)\),%(\w+)$', t)
        if not m:
            continue
        L, P = m.groups()
        if i + 2 >= len(ins):
            continue
        m2 = re.match(rf'cmp\s+0x(30|28)\(%{L}\),%{P}$', ins[i + 1][1])
        m3 = re.match(r'jae\s+(?:0x)?([0-9a-f]+)', ins[i + 2][1])
        if not (m2 and m3):
            continue
        bound, fallback = m2.group(1), int(m3.group(1), 16)
        body = []
        length = None
        for j in range(i + 3, min(i + 60, len(ins))):
            tj = ins[j][1]
            body.append(tj)
            ma = re.match(rf'(?:add\s+\$0x([0-9a-f]+),%{P}|lea\s+0x([0-9a-f]+)\(%(\w+)\),%\w+)$', tj)
            if ma and ma.group(1):
                length = int(ma.group(1), 16)
            if re.match(rf'mov\s+%\w+,0x18\(%{L}\)$', tj):
                if length is None:
                    # lea 0xN(%rcx),%rax style
                    for tb in body:
                        ml = re.match(r'lea\s+0x([0-9a-f]+)\(%\w+\),%\w+$', tb)
                        if ml:
                            length = int(ml.group(1), 16)
                break
        # header: first store to (%P) / 0x0(%P) – immediate or from a list field
        header = '?'
        for tb in body:
            mh = re.match(rf'movl\s+\$0x([0-9a-f]+),\(%{P}\)$', tb)
            if mh:
                header = 'imm ' + mh.group(1)
                break
            mf = re.match(rf'mov\s+0x([0-9a-f]+)\(%{L}\),%(\w+)$', tb)
            if mf:
                header = f'list+{mf.group(1)}'
                break
        # fallback: first indirect call at/after the jae target
        slot = '?'
        k = idx.get(fallback)
        if k is not None:
            for j in range(k, min(k + 12, len(ins))):
                mc = re.match(r'call\s+\*0x([0-9a-f]+)\(%\w+\)$', ins[j][1])
                if mc:
                    slot = int(mc.group(1), 16) // 8
                    break
        sites.append((a, L, bound, length, header, slot))
    return sites


def dump_record_writers(lines, file=None):
    """Print record writer summary for disassembly lines to file (defaults to stdout)."""
    if file is None:
        file = sys.stdout
    sites = find_record_writers(lines)
    print(f'{len(sites)} record write sites', file=file)
    kinds = collections.Counter((h, ln, s, b) for _, _, b, ln, h, s in sites)
    for (h, ln, s, b), n in kinds.most_common():
        print(f'  {n:3}× header {h:18} length {ln} B  bound +0x{b}  fallback slot {s}', file=file)
    return sites


def main(argv=None, out=None):
    if argv is None:
        argv = sys.argv[1:]
    if out is None:
        out = sys.stdout
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip(), file=out)
        return 0 if (argv and argv[0] in ("-h", "--help")) else 2

    has_error = False
    for path in argv:
        if not os.path.isfile(path):
            print(f"error: file not found: '{path}'", file=sys.stderr)
            has_error = True
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
            dump_record_writers(lines, file=out)
        except Exception as e:
            print(f"error: failed to read '{path}': {e}", file=sys.stderr)
            has_error = True
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
