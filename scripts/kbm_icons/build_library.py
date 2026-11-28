r"""Build the keyboard/mouse icon library for the retail title and write it into a COPY of its data.rpf.

    python scripts/kbm_icons/build_library.py --game D:\Game --out %LOCALAPPDATA%\XCL\<title>\overlay\data.rpf

XCL reads a game file from the title's overlay folder when one is there (crates/xcl/src/vfs.rs), so the
patched copy replaces the original without touching the game folder. Delete the copy to undo.

What it does (details and measurements: docs/kbm-icons/README.md):
1. Finds the title's `platform_xbsx.input_icons` movie in data.rpf. Each archive entry is a plain zlib
   stream; nothing is decrypted. The entry is at a known offset in the version we have, checked by content,
   with a scan of the archive as a fallback.
2. Rebuilds that movie: the 38 `Main_Button_N` sprites and their shapes come from icons-master.svg; every
   other tag (the movie's own script, exports, symbol classes) is copied from the user's file as it is.
3. Packs it to the original entry's exact sizes (unpacked and zlib), so the archive's table of contents
   still describes it, and writes data.rpf with only that entry changed.

Supported SVG in the master file: <g transform="translate(x y)">, <rect rx>, <polygon>, <path> with absolute
M L H V Q A Z, <text> (outlined from Windows' Bahnschrift Bold Condensed), fill / stroke / stroke-width as
attributes, #rgb or #rrggbb colours. Units are the library's (1 unit = 20 twips; a symbol is 64 tall).
Needs Python 3.10+ and fontTools (MIT).
"""
import math, re, struct, sys, os, zlib
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkstemp


TW = 20

# ── font ─────────────────────────────────────────────────────────────────────
_font = None


def font():
    global _font
    if _font is None:
        from fontTools.ttLib import TTFont
        from fontTools.varLib import instancer
        f = TTFont(r"C:\Windows\Fonts\bahnschrift.ttf")
        _font = instancer.instantiateVariableFont(f, {"wght": 700, "wdth": 75})
    return _font


def glyph_contours(ch):
    """Contours of one glyph in font units, y-up, as lists of segments ('L', p0, p1) / ('Q', p0, c, p1)."""
    from fontTools.pens.basePen import BasePen
    f = font(); gs = f.getGlyphSet(); name = f.getBestCmap()[ord(ch)]

    class P(BasePen):
        def __init__(s): super().__init__(gs); s.cs = []; s.cur = None; s.start = None
        def _moveTo(s, p): s.cs.append([]); s.cur = s.start = p
        def _lineTo(s, p): s.cs[-1].append(("L", s.cur, p)); s.cur = p
        def _qCurveToOne(s, c, p): s.cs[-1].append(("Q", s.cur, c, p)); s.cur = p
        def _curveToOne(s, c1, c2, p):  # cubic → two quads (Bahnschrift is quadratic; kept for safety)
            m = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
            s._qCurveToOne(((s.cur[0] + 3 * c1[0]) / 4, (s.cur[1] + 3 * c1[1]) / 4), m)
            s._qCurveToOne(((3 * c2[0] + p[0]) / 4, (3 * c2[1] + p[1]) / 4), p)
        def _closePath(s):
            if s.cur != s.start: s.cs[-1].append(("L", s.cur, s.start)); s.cur = s.start
        _endPath = _closePath
    pen = P(); gs[name].draw(pen)
    return pen.cs, gs[name].width


def text_contours(s, x, y, fs, anchor, text_length):
    """SVG text → contours in symbol units (y-down). The cap height matches the SVG design (0.7 em)."""
    f = font(); cap_font = f["OS/2"].sCapHeight
    k = (fs * 0.7) / cap_font
    adv = 0; glyphs = []
    for ch in s:
        cs, w = glyph_contours(ch); glyphs.append((adv, cs)); adv += w
    width = adv * k
    sx = k
    if text_length:
        sx = text_length / adv; width = text_length
    x0 = x - width / 2 if anchor == "middle" else x
    out = []
    for gx, cs in glyphs:
        for c in cs:
            out.append([tuple([seg[0]] + [(x0 + (gx + px) * sx, y - py * k) for px, py in seg[1:]]) for seg in c])
    return out


# ── SVG geometry ─────────────────────────────────────────────────────────────

def arc_to_quads(p0, rx, ry, rot, large, sweep, p1):
    """SVG elliptical arc → quadratic segments (each ≤ 45°)."""
    x1, y1 = p0; x2, y2 = p1
    if rx == 0 or ry == 0: return [("L", p0, p1)]
    phi = math.radians(rot); cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p, y1p = cp * dx + sp * dy, -sp * dx + cp * dy
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1: rx *= math.sqrt(lam); ry *= math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    co = math.sqrt(max(0, num / den)) if den else 0
    if large == sweep: co = -co
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2; cy = sp * cxp + cp * cyp + (y1 + y2) / 2
    ang = lambda ux, uy, vx, vy: math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dt > 0: dt -= 2 * math.pi
    if sweep and dt < 0: dt += 2 * math.pi
    n = max(1, math.ceil(abs(dt) / (math.pi / 2 if max(rx, ry) <= 14 else math.pi / 3) - 1e-9)); h = dt / n  # 90 deg pieces for corners, 60 for big rings
    pt = lambda t: (cx + rx * math.cos(t) * cp - ry * math.sin(t) * sp, cy + rx * math.cos(t) * sp + ry * math.sin(t) * cp)
    segs = []; cur = p0
    for i in range(n):
        ta, tb = t1 + i * h, t1 + (i + 1) * h; tm = (ta + tb) / 2
        e = 1 / math.cos(h / 2)
        c = (cx + rx * e * math.cos(tm) * cp - ry * e * math.sin(tm) * sp, cy + rx * e * math.cos(tm) * sp + ry * e * math.sin(tm) * cp)
        nxt = p1 if i == n - 1 else pt(tb)
        segs.append(("Q", cur, c, nxt)); cur = nxt
    return segs


def path_contours(d):
    toks = re.findall(r"[MLHVAQZ]|-?\d*\.?\d+(?:e-?\d+)?", d)
    i = 0; cs = []; cur = start = (0, 0); cmd = None
    def num():
        nonlocal i; v = float(toks[i]); i += 1; return v
    while i < len(toks):
        if re.match(r"[A-Z]", toks[i]): cmd = toks[i]; i += 1
        if cmd == "M": cur = start = (num(), num()); cs.append([]); cmd = "L"; continue
        if cmd == "Z":
            if cs and cur != start: cs[-1].append(("L", cur, start))
            cur = start; continue
        if cmd == "L": p = (num(), num()); cs[-1].append(("L", cur, p)); cur = p
        elif cmd == "H": p = (num(), cur[1]); cs[-1].append(("L", cur, p)); cur = p
        elif cmd == "V": p = (cur[0], num()); cs[-1].append(("L", cur, p)); cur = p
        elif cmd == "Q": c = (num(), num()); p = (num(), num()); cs[-1].append(("Q", cur, c, p)); cur = p
        elif cmd == "A":
            rx, ry, rot, la, sw = num(), num(), num(), num(), num(); p = (num(), num())
            cs[-1].extend(arc_to_quads(cur, rx, ry, rot, int(la), int(sw), p)); cur = p
        else: raise ValueError(cmd)
    return [c for c in cs if c]


def rect_contours(x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    if r <= 0:
        return path_contours(f"M{x} {y} H{x + w} V{y + h} H{x} Z")
    return path_contours(f"M{x + r} {y} H{x + w - r} A{r} {r} 0 0 1 {x + w} {y + r} V{y + h - r} A{r} {r} 0 0 1 {x + w - r} {y + h} "
                         f"H{x + r} A{r} {r} 0 0 1 {x} {y + h - r} V{y + r} A{r} {r} 0 0 1 {x + r} {y} Z")


def color(v):
    if not v or v == "none": return None
    v = v.lstrip("#")
    if len(v) == 3: v = "".join(c * 2 for c in v)
    return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16), 255)


def elements(svg):
    """Yield (contours, fill, stroke, stroke_width) for each drawable element, in paint order.
    `svg` is an SVG fragment (text) or an element whose children are drawn (a master-file <symbol>)."""
    root = ET.fromstring(f"<g>{svg}</g>") if isinstance(svg, str) else svg

    def walk(el, dx, dy):
        tag = el.tag.split("}")[-1]
        if tag == "g":
            m = re.match(r"translate\(([-\d.e]+)\s+([-\d.e]+)\)", el.get("transform", "translate(0 0)"))
            for c in el: yield from walk(c, dx + float(m.group(1)), dy + float(m.group(2)))
            return
        fill = color(el.get("fill", "#000")); stroke = color(el.get("stroke")); sw = float(el.get("stroke-width", "1"))
        if tag == "rect":
            cs = rect_contours(float(el.get("x")), float(el.get("y")), float(el.get("width")), float(el.get("height")), float(el.get("rx", 0)))
        elif tag == "path":
            cs = path_contours(el.get("d"))
        elif tag == "polygon":
            pts = [tuple(map(float, p.split(","))) for p in el.get("points").split()]
            cs = [[("L", pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]]
        elif tag == "text":
            tl = el.get("textLength")
            cs = text_contours(el.text, float(el.get("x")), float(el.get("y")), float(el.get("font-size")),
                               el.get("text-anchor", "start"), float(tl) if tl else None)
            stroke = None
        else:
            raise ValueError(tag)
        cs = [[tuple([s[0]] + [(px + dx, py + dy) for px, py in s[1:]]) for s in c] for c in cs]
        if tag == "text":
            # Variable-font glyphs overlap (an E is a stem plus bars). Give every solid contour its own
            # shape, with the holes it contains, so overlaps add up instead of cancelling (even-odd).
            areas = [shoelace(c) for c in cs]
            solid_sign = 1 if areas[max(range(len(cs)), key=lambda i: abs(areas[i]))] > 0 else -1
            solids = [i for i in range(len(cs)) if areas[i] * solid_sign > 0]
            groups = {i: [cs[i]] for i in solids}
            for i in range(len(cs)):
                if i in groups: continue
                p = cs[i][0][1]
                owners = [j for j in solids if point_in([s[1] for s in cs[j]], p)]
                if owners: groups[min(owners, key=lambda j: abs(areas[j]))].append(cs[i])
            for i in solids: yield groups[i], fill, stroke, sw
            return
        yield cs, fill, stroke, sw

    for el in root: yield from walk(el, 0, 0)


# ── SWF writing ──────────────────────────────────────────────────────────────

class BW:
    def __init__(s): s.out = bytearray(); s.acc = 0; s.n = 0
    def ub(s, v, n):
        for i in range(n - 1, -1, -1):
            s.acc = (s.acc << 1) | ((v >> i) & 1); s.n += 1
            if s.n == 8: s.out.append(s.acc); s.acc = 0; s.n = 0
    def sb(s, v, n): s.ub(v & ((1 << n) - 1), n)
    def align(s):
        if s.n: s.ub(0, 8 - s.n)
    def bytes(s, b): s.align(); s.out += b


def nbits_s(*vals):
    n = 1
    for v in vals:
        while not (-(1 << (n - 1)) <= v < (1 << (n - 1))): n += 1
    return n


def write_rect(w, x0, x1, y0, y1):
    n = nbits_s(x0, x1, y0, y1); w.ub(n, 5)
    for v in (x0, x1, y0, y1): w.sb(v, n)
    w.align()


def shoelace(c):
    a = 0
    for s in c:
        (x0, y0), (x1, y1) = s[1], s[-1]
        a += x0 * y1 - x1 * y0
    return a


def point_in(poly, p):
    x, y = p; inside = False; n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0): inside = not inside
    return inside


def define_shape3(sid, contours, fill, stroke, sw):
    # integer twips, closing each contour exactly
    q = lambda p: (round(p[0] * TW), round(p[1] * TW))
    cs = [[tuple([s[0]] + [q(p) for p in s[1:]]) for s in c] for c in contours]
    cs = [[s for s in c if not (s[0] == "L" and s[1] == s[2])] for c in cs]
    cs = [c for c in cs if c]
    closed = [c[0][1] == c[-1][-1] for c in cs]
    polys = [[s[1] for s in c] for c in cs]
    # nesting depth for even-odd fills
    depth = []
    for i, c in enumerate(cs):
        t = c[0][1]; depth.append(sum(1 for j, pg in enumerate(polys) if j != i and closed[j] and point_in(pg, t)))
    pad = round(sw * TW / 2) if stroke else 0
    xs = [p[0] for c in cs for s in c for p in s[1:]]; ys = [p[1] for c in cs for s in c for p in s[1:]]
    w = BW()
    w.bytes(struct.pack("<H", sid))
    write_rect(w, min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
    fills = [fill] if fill else []; lines = [stroke] if stroke else []
    w.bytes(bytes([len(fills)]))
    for f in fills: w.bytes(bytes([0x00, *f]))
    w.bytes(bytes([len(lines)]))
    for l in lines: w.bytes(struct.pack("<H", round(sw * TW)) + bytes(l))
    nf, nl = (1 if fills else 0), (1 if lines else 0)
    w.ub(nf, 4); w.ub(nl, 4)
    x = y = 0
    for i, c in enumerate(cs):
        f0 = f1 = 0
        if fills and closed[i]:
            interior_on_f0 = shoelace(c) > 0
            filled_inside = depth[i] % 2 == 0
            if interior_on_f0 == filled_inside: f0 = 1
            else: f1 = 1
        sx, sy = c[0][1]
        # style change: moveTo + fill0 + fill1 (+ line)
        flags = 0x01 | (0x02 if nf else 0) | (0x04 if nf else 0) | (0x08 if nl else 0)
        w.ub(0, 1); w.ub(flags, 5)
        n = nbits_s(sx, sy); w.ub(n, 5); w.sb(sx, n); w.sb(sy, n)
        if nf: w.ub(f0, nf); w.ub(f1, nf)
        if nl: w.ub(1, nl)
        x, y = sx, sy
        for s in c:
            if s[0] == "L":
                (ex, ey) = s[2]; dx, dy = ex - x, ey - y
                n = max(nbits_s(dx, dy), 2)
                w.ub(1, 1); w.ub(1, 1); w.ub(n - 2, 4); w.ub(1, 1); w.sb(dx, n); w.sb(dy, n)
                x, y = ex, ey
            else:
                (cx, cy), (ex, ey) = s[2], s[3]
                a, b, c2, d = cx - x, cy - y, ex - cx, ey - cy
                n = max(nbits_s(a, b, c2, d), 2)
                w.ub(1, 1); w.ub(0, 1); w.ub(n - 2, 4)
                for v in (a, b, c2, d): w.sb(v, n)
                x, y = ex, ey
    w.ub(0, 6); w.align()
    return bytes(w.out)


def tag(code, body):
    if len(body) < 63: return struct.pack("<H", code << 6 | len(body)) + body
    return struct.pack("<HI", code << 6 | 63, len(body)) + body


def read_tags(d, p, end):
    while p < end:
        h = struct.unpack_from("<H", d, p); hdr = p; p += 2; c, l = h[0] >> 6, h[0] & 63
        if l == 63: l = struct.unpack_from("<I", d, p)[0]; p += 4
        yield c, d[hdr:p + l], d[p:p + l]; p += l
        if c == 0: break


def build(d, symbols, target_len=None):
    """d: the title's input_icons movie. symbols: {'Main_Button_N': SVG fragment or <symbol> element}.
    Returns (new movie, its length before padding, number of shapes)."""
    nb = d[8] >> 3; hdr_end = 8 + (5 + 4 * nb + 7) // 8 + 4
    tags = list(read_tags(d, hdr_end, len(d)))
    names = {}
    for c, raw, body in tags:
        if c == 76:
            n = struct.unpack_from("<H", body)[0]; q = 2
            for _ in range(n):
                i = struct.unpack_from("<H", body, q)[0]; q += 2; e = body.index(b"\0", q); names[body[q:e].decode().split(".")[-1]] = i; q = e + 1
    shapes = {}; shape_tags = []; sprite_tags = {}; next_id = 1
    used = {struct.unpack_from("<H", body)[0] for c, _, body in tags if c == 39}
    for nm, sid in sorted(((n, i) for n, i in names.items() if n.startswith("Main_Button_")), key=lambda kv: kv[1]):
        if nm not in symbols: raise KeyError(f"no design for {nm}")
        places = []
        for cs, fill, stroke, sw in elements(symbols[nm]):
            key = (repr(cs), fill, stroke, sw)
            if key not in shapes:
                while next_id in used: next_id += 1
                shapes[key] = next_id; used.add(next_id)
                shape_tags.append(tag(32, define_shape3(next_id, cs, fill, stroke, sw)))
            places.append(shapes[key])
        body = struct.pack("<HH", sid, 1)
        for depth, ch in enumerate(places, 1):
            body += tag(26, bytes([0x02]) + struct.pack("<HH", depth, ch))
        body += tag(1, b"") + tag(0, b"")
        sprite_tags[sid] = tag(39, body)
    out = bytearray(d[:hdr_end]); inserted = False
    for c, raw, body in tags:
        if c == 32:
            if not inserted: out += b"".join(shape_tags); inserted = True
            continue
        if c == 39:
            sid = struct.unpack_from("<H", body)[0]
            if not inserted: out += b"".join(shape_tags); inserted = True
            out += sprite_tags[sid]; continue
        out += raw
    real = len(out)
    if target_len:
        if real > target_len: raise ValueError(f"movie is {real} bytes, over {target_len}")
        out += bytes(target_len - real)
    struct.pack_into("<I", out, 4, len(out))
    return bytes(out), real, len(shape_tags)


# ── data.rpf ────────────────────────────────────────────────────────────────

KNOWN_OFFSET = 0x117d7000   # the entry in the data.rpf we have (481,544,304 bytes)
MARK = b"platform_xbsx.input_icons"


def read_stream(f, off, limit=1 << 20):
    """Inflate the zlib stream at `off`. Returns (data, compressed size) or None."""
    f.seek(off); buf = f.read(limit)
    if len(buf) < 2 or buf[0] != 0x78: return None
    d = zlib.decompressobj()
    try: out = d.decompress(buf)
    except zlib.error: return None
    if not d.eof: return None
    return out, len(buf) - len(d.unused_data)


def is_icon_movie(data):
    return data[:3] in (b"GFX", b"CFX") and MARK in data and b"Main_Button_" in data


def find_entry(path, offset):
    with open(path, "rb") as f:
        hit = read_stream(f, offset)
        if hit and is_icon_movie(hit[0]): return offset, hit[0], hit[1]
        print(f"not at {offset:#x}; scanning {path} for the icon library...")
        size = os.fstat(f.fileno()).st_size; base = 0; step = 64 << 20
        while base < size:
            f.seek(base); chunk = f.read(step + 16)
            i = chunk.find(b"\x78\xda")
            while 0 <= i < step:
                off = base + i
                if off % 16 == 0:
                    hit = read_stream(f, off)
                    if hit and is_icon_movie(hit[0]): return off, hit[0], hit[1]
                i = chunk.find(b"\x78\xda", i + 1)
            base += step
    raise SystemExit("platform_xbsx.input_icons was not found in data.rpf")


def pack_exact(movie, real, csize):
    """A zlib stream of exactly `csize` bytes for `movie` (whose bytes after `real` are padding): the tail of the
    padding gets seeded pseudo-random bytes until the compressed size lands on `csize`."""
    import random
    def comp(b):
        c = zlib.compressobj(9, zlib.DEFLATED, 15, 9); return c.compress(b) + c.flush()
    room = len(movie) - real
    base = len(comp(movie))
    if base > csize:
        raise SystemExit(f"the new library packs to {base} bytes, over the original's {csize}: simplify the icons")
    est = csize - base
    for seed in range(400):
        rnd = random.Random(seed); tail = bytes(rnd.getrandbits(8) for _ in range(room))
        for n in range(max(0, est - 40), min(room, est + 40) + 1):
            data = movie[:real] + bytes(room - n) + tail[:n]
            c = comp(data)
            if len(c) == csize: return c, data
    raise SystemExit("could not pack the library to the original's exact compressed size")


def load_master(path):
    syms = {}
    for el in ET.parse(path).getroot().iter():
        if el.tag.split("}")[-1] == "symbol" and el.get("id", "").startswith("Main_Button_"):
            syms[el.get("id")] = el
    return syms


@contextmanager
def staged_output(destination):
    """Publish a completed output by replacing its directory entry."""
    fd, name = mkstemp(suffix=".tmp", dir=destination.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    import argparse, shutil
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Build the keyboard/mouse icon library into a copy of data.rpf.")
    ap.add_argument("--game", required=True, help="the title's game folder (holds data.rpf); only read")
    ap.add_argument("--out", help="where to write the patched copy of data.rpf (the title's XCL overlay folder)")
    ap.add_argument("--master", default=os.path.join(here, "..", "..", "docs", "kbm-icons", "icons-master.svg"))
    ap.add_argument("--offset", type=lambda v: int(v, 0), default=KNOWN_OFFSET)
    ap.add_argument("--movie-out", help="also write the rebuilt movie here (made from game data: keep it local)")
    a = ap.parse_args()
    game = Path(a.game).resolve()
    src = (game / "data.rpf").resolve()
    out = Path(a.out).resolve() if a.out else None
    movie_out = Path(a.movie_out).resolve() if a.movie_out else None
    for flag, destination in (("--out", out), ("--movie-out", movie_out)):
        if destination is not None and (destination.is_relative_to(game) or destination == src):
            ap.error(f"{flag} must be outside the game folder and must not overwrite data.rpf")

    off, orig, csize = find_entry(src, a.offset)
    movie, real, nshapes = build(orig, load_master(a.master), target_len=len(orig))
    c, data = pack_exact(movie, real, csize)
    assert zlib.decompress(c) == data and len(data) == len(orig) and len(c) == csize
    print(f"input_icons at {off:#x}: {len(orig)} bytes ({csize} packed); new library {real} bytes, "
          f"{nshapes} shapes, packed to {len(c)} bytes")
    if movie_out is not None:
        with staged_output(movie_out) as tmp:
            tmp.write_bytes(data)
    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with staged_output(out) as tmp:
        shutil.copyfile(src, tmp)
        with tmp.open("r+b") as f:
            f.seek(off)
            f.write(c)
        with tmp.open("rb") as f:
            back = read_stream(f, off)
        if not back or back[0] != data:
            raise SystemExit("verification of the written copy failed")
    print(f"wrote {out}: only the input_icons entry differs from the game's data.rpf")


if __name__ == "__main__":
    main()
