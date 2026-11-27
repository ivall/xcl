"""Print a PE's exports: ordinal, all names for that ordinal, RVA (forwarders shown)."""
import struct, sys

for path in sys.argv[1:]:
    b = open(path, 'rb').read()
    pe = struct.unpack_from('<I', b, 0x3C)[0]
    nsec = struct.unpack_from('<H', b, pe + 6)[0]
    opt = pe + 24
    magic = struct.unpack_from('<H', b, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    exp_rva, exp_size = struct.unpack_from('<II', b, dd)
    secs = []
    so = opt + struct.unpack_from('<H', b, pe + 20)[0]
    for i in range(nsec):
        va, raw_sz, raw = struct.unpack_from('<III', b, so + 40 * i + 12)
        # A section's virtual tail is zero-filled by the loader and has no file offset.
        secs.append((va, raw_sz, raw))

    def raw_span(rva, size=1):
        for va, raw_sz, raw in secs:
            if va <= rva and rva - va + size <= raw_sz:
                offset = rva - va + raw
                end = min(raw + raw_sz, len(b))
                if offset + size <= end:
                    return offset, end
        raise ValueError(f'export data at RVA {rva:#x} is outside file-backed section data')

    def cstr(rva):
        start, end = raw_span(rva)
        nul = b.find(b'\0', start, end)
        if nul < 0:
            raise ValueError(f'export string at RVA {rva:#x} has no terminator in its section')
        return b[start:nul].decode('latin1')

    print(f'== {path}')
    if not exp_rva:
        print('  (no exports)')
        continue
    e, _ = raw_span(exp_rva, 40)
    name_rva, base, nfun, nnames, afun, anames, aords = struct.unpack_from('<IIIIIII', b, e + 12)
    names = {}
    for i in range(nnames):
        nr, = struct.unpack_from('<I', b, raw_span(anames + 4 * i, 4)[0])
        o, = struct.unpack_from('<H', b, raw_span(aords + 2 * i, 2)[0])
        names.setdefault(o, []).append(cstr(nr))
    print(f'  {cstr(name_rva)}: {nfun} functions, {nnames} names, ordinal base {base}')
    for i in range(nfun):
        rva, = struct.unpack_from('<I', b, raw_span(afun + 4 * i, 4)[0])
        if not rva:
            continue
        fwd = f' -> {cstr(rva)}' if exp_rva <= rva < exp_rva + exp_size else ''
        print(f'  #{base + i:<4} {", ".join(names.get(i, ["(ordinal only)"]))} @{rva:#x}{fwd}')
