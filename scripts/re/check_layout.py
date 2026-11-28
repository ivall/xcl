"""Check the root-argument table hypothesis on all builder shaders: user SGPRs s[0:1] point
to the root arguments packed in order (constant = 4 B, root descriptor = 8 B, no padding).
Every scalar load from the table (s_load from the register pair holding s0/s1) must start
at a descriptor boundary or inside the constants.

usage: python check_layout.py <dir with the .dxbc files of bvhb_shaders.py> (needs xcl.exe)"""
import os, re, struct, subprocess, sys


def find_xcl():
    if os.environ.get('XCL_EXE'):
        return os.environ['XCL_EXE']
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    for rel in ('target/debug/xcl.exe', 'target/release/xcl.exe', 'target/debug/xcl', 'target/release/xcl'):
        candidate = os.path.normpath(os.path.join(root, rel))
        if os.path.isfile(candidate):
            return candidate
    return os.path.normpath(os.path.join(root, 'target', 'debug', 'xcl.exe'))


def parts(blob):
    n = struct.unpack_from('<I', blob, 28)[0]
    for k in range(n):
        off = struct.unpack_from('<I', blob, 32 + 4 * k)[0]
        yield blob[off:off + 4].decode('latin1'), blob[off + 8:off + 8 + struct.unpack_from('<I', blob, off + 4)[0]]


def layout(r):
    ver, n, poff = struct.unpack_from('<3I', r, 0)
    at, consts, descs = 0, [], {}
    for k in range(n):
        ty, vis, pay = struct.unpack_from('<3I', r, poff + 12 * k)
        if ty == 1:
            num = struct.unpack_from('<3I', r, pay)[2]
            consts.append((at, at + 4 * num))
            at += 4 * num
        elif ty in (2, 3, 4):
            descs[at] = 'CBV SRV UAV'.split()[ty - 2]
            at += 8
        else:
            return None
    return consts, descs, at


def check_directory(dirpath, xcl_bin=None):
    xcl = xcl_bin or find_xcl()
    if not os.path.isfile(xcl):
        print(f"error: xcl binary not found at '{xcl}'; build target/debug/xcl.exe, target/release/xcl.exe or set XCL_EXE", file=sys.stderr)
        return False

    ok = bad = 0
    for name in sorted(os.listdir(dirpath)):
        filepath = os.path.join(dirpath, name)
        if not os.path.isfile(filepath):
            continue
        blob = open(filepath, 'rb').read()
        rs = next((p for cc, p in parts(blob) if cc == 'RTS0'), None)
        if not rs:
            continue
        lay = layout(rs)
        if lay is None:
            continue
        consts, descs, size = lay
        lst = subprocess.run([xcl, 'disasm', filepath], capture_output=True, text=True, encoding='utf-8', errors='replace').stdout
        # the pair the table pointer was copied to: s_mov sA, s0 ; s_and sB, s1, 0xff
        m = re.search(r's_and_b32 s(\d+), s1, 0xff\s', lst)
        if not m or not re.search(rf's_mov_b32 s{int(m.group(1)) - 1}, s0\s', lst):
            print(name, 'no table pointer pattern')
            continue
        pair = f's[{int(m.group(1)) - 1}:{m.group(1)}]'
        loads = re.findall(r's_load_dword(x\d+)? s\[?(\d+)(?::(\d+))?\]?, ' + re.escape(pair) + r', (0x[0-9a-f]+)', lst)
        for w, lo, hi, off in loads:
            off = int(off, 16)
            nbytes = 4 * (int(w[1:]) if w else 1)
            good = off in descs or any(a <= off < b for a, b in consts) or (off - 4) in descs  # high half of a VA
            if good:
                ok += 1
            else:
                bad += 1
                print(f'{name}: load {nbytes} B @{off:#x} not on a parameter (descriptors {sorted(descs)}, constants {consts})')
    print(f'loads on parameter boundaries: {ok}, off: {bad}')
    return bad == 0


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print(__doc__.strip())
        return 0 if (args and args[0] in ('-h', '--help')) else 2
    dirpath = args[0]
    if not os.path.isdir(dirpath):
        print(f"error: directory not found: '{dirpath}'", file=sys.stderr)
        return 1
    return 0 if check_directory(dirpath) else 1


if __name__ == '__main__':
    sys.exit(main())
