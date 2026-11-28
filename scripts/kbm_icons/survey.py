r"""Survey the retail title's UI movies for button icons: which icon libraries exist and whether any
keyboard or mouse icon is among them. Reads only; prints a report.

    python scripts/kbm_icons/survey.py D:\Game\data.rpf [D:\Game\packs.rpf ...]

Each archive entry is a plain zlib stream on a 16-byte boundary; big entries are runs of streams of 64 KiB
unpacked each. The archive's table of contents is encrypted and is not read: entries are found by content,
and nothing is decrypted. A Scaleform movie starts with "GFX" (uncompressed) or "CFX" (zlib after 8 bytes).
"""
import re, sys, zlib
from collections import Counter

CHUNK = 65536
HEADERS = {b"\x78\x01", b"\x78\x5e", b"\x78\x9c", b"\x78\xda"}
# identifiers (symbol, class, function or text names) that would point at keyboard/mouse icons
IDENT = re.compile(rb"[A-Za-z_][A-Za-z0-9_.:]{2,}")
KBM = re.compile(rb"(?i)keyboard|mouse|kbm|keycap|^key_|_key$|platform_(?!xbsx)")


def inflate(buf):
    d = zlib.decompressobj()
    try: out = d.decompress(buf)
    except zlib.error: return None
    return (out, len(buf) - len(d.unused_data)) if d.eof else None


def movies(path):
    """Yield (offset, movie bytes) for every Scaleform movie in the archive."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0; n = len(data)
    while pos < n - 2:
        if data[pos:pos + 2] in HEADERS:
            hit = inflate(data[pos:pos + (1 << 20)])
            if hit:
                out, used = hit
                if out[:3] in (b"GFX", b"CFX"):
                    whole = bytearray(out); nxt = (pos + used + 15) & ~15; last = len(out)
                    while last == CHUNK:                      # a big entry continues in the next stream
                        more = inflate(data[nxt:nxt + (1 << 20)])
                        if not more: break
                        whole += more[0]; last = len(more[0]); nxt = (nxt + more[1] + 15) & ~15
                    movie = bytes(whole)
                    if movie[:3] == b"CFX":
                        try: movie = movie[:8] + zlib.decompress(movie[8:])
                        except zlib.error: pass
                    yield pos, movie
                    pos = nxt; continue
                pos = (pos + used + 15) & ~15; continue
        pos += 16


def name_of(movie):
    """The movie's name from its exporter-info tag (code 1000), e.g. "hud"."""
    m = re.search(rb"\x00\x00([\x01-\x40])([A-Za-z0-9_]+)\x00", movie[:200])
    return m.group(2).decode() if m and m.group(1)[0] == len(m.group(2)) else "?"


def main():
    for path in sys.argv[1:]:
        found = list(movies(path))
        defines = [(o, m) for o, m in found if re.search(rb"platform_\w+_input_icons_Main", m)]
        users = [(o, m) for o, m in found if b"InputIconV2" in m]
        hits = {}
        for o, m in found:
            for w in set(t for t in IDENT.findall(m) if KBM.search(t)):
                hits.setdefault(w.decode(errors="replace"), []).append(name_of(m))
        prefixes = Counter(p.decode() for o, m in found for p in set(re.findall(rb"platform_[a-z0-9]+", m)))
        symbols = sorted({int(x) for o, m in defines for x in re.findall(rb"\.input_icons\.Main_Button_(\d+)", m)})
        print(f"{path}: {len(found)} Scaleform movies")
        print(f"  icon libraries (define input_icons): {len(defines)} -> " +
              ", ".join(f"{name_of(m)} at {o:#x}, {len(m)} bytes" for o, m in defines))
        print(f"  symbols exported: {len(symbols)}: Main_Button_{symbols}" if symbols else "  symbols exported: none")
        print(f"  movies that place icons (InputIconV2): {len(users)} -> " + ", ".join(name_of(m) for o, m in users))
        print(f"  platform prefixes seen: {dict(prefixes) or 'none'}")
        print("  identifiers naming a keyboard, mouse or key: " + ("none" if not hits else ""))
        for w, where in sorted(hits.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            print(f"    {w}: {len(where)} movie(s)" + ("" if len(where) > 3 else " (" + ", ".join(where) + ")"))


if __name__ == "__main__":
    main()
