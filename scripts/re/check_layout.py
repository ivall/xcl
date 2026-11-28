"""Check the root-argument table hypothesis on all builder shaders: user SGPRs s[0:1] point
to the root arguments packed in order (constant = 4 B, root descriptor = 8 B, no padding).
Every scalar load from the table (s_load from the register pair holding s0/s1) must start
at a descriptor boundary or inside the constants.

usage: python check_layout.py <dir with the .dxbc files of bvhb_shaders.py> [--xcl <path>]"""
import argparse
import os
import re
import struct
import subprocess
import sys

DEFAULT_XCL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'target', 'debug', 'xcl.exe')


def parts(blob):
    """Parse DXBC chunk FourCCs and payloads, validating header and chunk bounds."""
    if len(blob) < 32 or blob[:4] != b'DXBC':
        raise ValueError('not a DXBC container')
    n = struct.unpack_from('<I', blob, 28)[0]
    table_end = 32 + 4 * n
    if table_end > len(blob):
        raise ValueError('truncated DXBC chunk offset table')
    chunks = []
    for k in range(n):
        off = struct.unpack_from('<I', blob, 32 + 4 * k)[0]
        if off < table_end or off + 8 > len(blob):
            raise ValueError(f'chunk {k} header out of bounds')
        cc = blob[off:off + 4].decode('latin1')
        size = struct.unpack_from('<I', blob, off + 4)[0]
        if off + 8 + size > len(blob):
            raise ValueError(f'chunk {cc!r} payload truncated')
        chunks.append((cc, blob[off + 8:off + 8 + size]))
    return chunks


def layout(r):
    """Compute root-argument byte layout (constants ranges, descriptor offsets, total size)."""
    if r is None:
        return None
    if len(r) < 12:
        raise ValueError('truncated RTS0 header')
    ver, n, poff = struct.unpack_from('<3I', r, 0)
    if poff + 12 * n > len(r):
        raise ValueError('truncated RTS0 parameter table')
    at, consts, descs = 0, [], {}
    for k in range(n):
        ty, vis, pay = struct.unpack_from('<3I', r, poff + 12 * k)
        if ty == 1:
            if pay + 12 > len(r):
                raise ValueError('truncated RTS0 constant entry')
            num = struct.unpack_from('<3I', r, pay)[2]
            consts.append((at, at + 4 * num))
            at += 4 * num
        elif ty in (2, 3, 4):
            descs[at] = ('CBV', 'SRV', 'UAV')[ty - 2]
            at += 8
        else:
            return None
    return consts, descs, at


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Check root-argument table scalar loads in DXBC shaders.'
    )
    parser.add_argument('dir', help='directory containing .dxbc files')
    parser.add_argument('--xcl', default=DEFAULT_XCL, help='path to xcl disassembler')
    return parser.parse_args(argv)


def check_listing(name, lst, consts, descs):
    """Check scalar loads from the root-argument table register pair in one disassembly."""
    # the pair the table pointer was copied to: s_mov sA, s0 ; s_and sB, s1, 0xff
    m = re.search(r's_and_b32 s(\d+), s1, 0xff(?:\s|$)', lst)
    if not m or int(m.group(1)) < 1:
        print(name, 'no table pointer pattern')
        return 0, 0
    hi_reg = int(m.group(1))
    lo_reg = hi_reg - 1
    if not re.search(rf's_mov_b32 s{lo_reg}, s0(?:\s|$)', lst):
        print(name, 'no table pointer pattern')
        return 0, 0
    pair = f's[{lo_reg}:{hi_reg}]'
    loads = re.findall(
        r's_load_dword(x\d+)? s\[?(\d+)(?::(\d+))?\]?, ' + re.escape(pair) + r', (0x[0-9a-f]+)',
        lst,
    )
    ok = bad = 0
    for w, lo, hi, off in loads:
        off = int(off, 16)
        nbytes = 4 * (int(w[1:]) if w else 1)
        good = off in descs or any(a <= off < b for a, b in consts) or (off - 4) in descs  # high half of a VA
        if good:
            ok += 1
        else:
            bad += 1
            print(f'{name}: load {nbytes} B @{off:#x} not on a parameter (descriptors {sorted(descs)}, constants {consts})')
    return ok, bad


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isdir(args.dir):
        print(f'error: not a directory: {args.dir}', file=sys.stderr)
        return 2
    if not os.path.isfile(args.xcl):
        print(f'error: xcl not found: {args.xcl}', file=sys.stderr)
        return 1

    ok = bad = errors = 0
    for name in sorted(os.listdir(args.dir)):
        path = os.path.join(args.dir, name)
        if not os.path.isfile(path) or not name.lower().endswith('.dxbc'):
            continue
        with open(path, 'rb') as f:
            blob = f.read()
        try:
            rs = next((p for cc, p in parts(blob) if cc == 'RTS0'), None)
            if rs is None:
                continue
            lay = layout(rs)
        except ValueError as exc:
            print(f'{name}: skipping invalid DXBC ({exc})', file=sys.stderr)
            continue
        if lay is None:
            continue
        consts, descs, _ = lay
        try:
            proc = subprocess.run(
                [args.xcl, 'disasm', path],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
            )
        except OSError as exc:
            errors += 1
            print(f'{name}: failed to run {args.xcl}: {exc}', file=sys.stderr)
            continue
        if proc.returncode != 0:
            errors += 1
            detail = proc.stderr.strip() or f'exit code {proc.returncode}'
            print(f'{name}: xcl disasm failed ({proc.returncode}): {detail}', file=sys.stderr)
            continue
        file_ok, file_bad = check_listing(name, proc.stdout, consts, descs)
        ok += file_ok
        bad += file_bad

    print(f'loads on parameter boundaries: {ok}, off: {bad}')
    if errors:
        print(f'disassembler errors: {errors}', file=sys.stderr)
    return 1 if (bad > 0 or errors > 0) else 0


if __name__ == '__main__':
    raise SystemExit(main())
