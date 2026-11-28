"""Keyboard/mouse keycap designs for the retail title's button-icon library (`platform_xbsx.input_icons`).

One design per `Main_Button_N` symbol the title exports, drawn on the library's own 64-unit grid: white
shapes, a 3-unit black outline, black marks, unlit parts #676767. Only solid fills and strokes, so every
symbol maps onto a Scaleform DefineShape3. Design notes: docs/kbm-icons/README.md.

    python scripts/kbm_icons/keycaps.py            # writes docs/kbm-icons/icons-master.svg and the preview

`icons-master.svg` is the master copy that `build_library.py` reads. Edit the designs here and regenerate,
or edit the master file directly (keep to the SVG subset build_library.py documents).
"""
import sys

INK, PLATE, OFF = "#000", "#fff", "#676767"
FONT = "'Barlow Condensed','Arial Narrow',sans-serif"
SW = 3


def rr(x, y, w, h, r, fill, sw=SW):
    return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{r}' fill='{fill}' stroke='{INK}' stroke-width='{sw}' stroke-linejoin='round'/>"


def keycap(x, y, w, h, r=12, lit=True, sw=SW, lip=None):
    """A key: face on top, a lip along the bottom edge. Returns (svg, face centre y)."""
    lip = lip if lip is not None else round(h * 0.13, 2)
    face = PLATE if lit else OFF
    side = OFF if lit else INK
    s = rr(x, y, w, h, r, side, sw) + rr(x, y, w, h - lip, r, face, sw)
    return s, y + (h - lip) / 2


def text(s, cx, cy, cap, fill=INK):
    fs = cap / 0.7  # Barlow Condensed cap height is 0.7 em
    fit = {4: 46}.get(len(s))
    tl = f" textLength='{fit}' lengthAdjust='spacingAndGlyphs'" if fit else ""
    return (f"<text x='{cx}' y='{cy + cap / 2:.2f}' font-family=\"{FONT}\" font-weight='700' font-size='{fs:.2f}' "
            f"text-anchor='middle' fill='{fill}'{tl}>{s}</text>")


def tri_pts(cx, cy, size, d):
    h = size / 2
    return {"up": [(cx, cy - h), (cx + h, cy + h * .75), (cx - h, cy + h * .75)],
            "down": [(cx, cy + h), (cx - h, cy - h * .75), (cx + h, cy - h * .75)],
            "left": [(cx - h, cy), (cx + h * .75, cy - h), (cx + h * .75, cy + h)],
            "right": [(cx + h, cy), (cx - h * .75, cy + h), (cx - h * .75, cy - h)]}[d]


def tri(cx, cy, size, d, fill=INK, stroke=None):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in tri_pts(cx, cy, size, d))
    st = f" stroke='{INK}' stroke-width='{stroke}' stroke-linejoin='round'" if stroke else ""
    return f"<polygon points='{pts}' fill='{fill}'{st}/>"


def haloed(d, w=3.5):
    """An open stroke drawn white with a black outline, so it reads on any ground."""
    return (f"<path d='{d}' fill='none' stroke='{INK}' stroke-width='{w + 2 * SW}' stroke-linecap='round' stroke-linejoin='round'/>"
            f"<path d='{d}' fill='none' stroke='{PLATE}' stroke-width='{w}' stroke-linecap='round' stroke-linejoin='round'/>")


# ── full-size keys ───────────────────────────────────────────────────────────

def single(legend):
    s, cy = keycap(1.5, 1.5, 61, 61)
    cap = {1: 24, 3: 18, 4: 16}[len(legend)]
    return s + text(legend, 32, cy, cap)


def arrow_key(d):
    s, cy = keycap(1.5, 1.5, 61, 61)
    return s + tri(32, cy, 22, d)


# Wide keys keep the keyboard's proportions: a 1u key is 64 wide; Tab and Ctrl are 1.4u, Shift and
# Enter 1.75u. Each pairs its symbol with the word, the way the keys are printed.
WIDE = {"TAB": 88, "CTRL": 88, "SHIFT": 88, "ENTER": 88}
# The title gives every icon a fixed 64-unit slot (it ignores the symbol's width): measured in game, a label
# ends 37 units left of the slot and the next prompt's label starts 51 units right of it. A wide key therefore
# runs from x = -6 to 82: 31 units clear of its own label, 33 clear of the next prompt.
WIDE_X0 = -6


def return_arrow(x, cy, s=1.0):
    stem = f"M{x + 13 * s} {cy - 9 * s} V{cy + 3 * s} H{x + 4 * s}"
    return (f"<path d='{stem}' fill='none' stroke='{INK}' stroke-width='{4 * s}' stroke-linecap='square' stroke-linejoin='miter'/>"
            + tri(x, cy + 3 * s, 12 * s, "left"))


def shift_arrow(cx, cy, s=1.0):
    a = f"M{cx} {cy - 11 * s} L{cx + 11 * s} {cy + 1 * s} H{cx + 5 * s} V{cy + 10 * s} H{cx - 5 * s} V{cy + 1 * s} H{cx - 11 * s} Z"
    return f"<path d='{a}' fill='none' stroke='{INK}' stroke-width='{3 * s}' stroke-linejoin='round'/>"


def tab_arrow(x, cy, s=1.0):
    return (f"<path d='M{x} {cy} H{x + 13 * s} M{x + 17 * s} {cy - 8 * s} V{cy + 8 * s}' fill='none' stroke='{INK}' "
            f"stroke-width='{3.5 * s}' stroke-linecap='square'/>" + tri(x + 12 * s, cy, 11 * s, "right"))


def wide_key(word):
    w = WIDE[word]
    s, cy = keycap(1.5, 1.5, w - 3, 61)
    icon = {"ENTER": lambda x: return_arrow(x + 6, cy), "SHIFT": lambda x: shift_arrow(x + 11, cy),
            "TAB": lambda x: tab_arrow(x, cy), "CTRL": None}[word]
    cap = 16 if len(word) <= 4 else 14
    # The title scales and shakes every icon about (32, 32) (ICON_OFFSET -32, PRESSED_SCALE 0.8), so a
    # wide key is centred on x = 32 and runs into negative x, not from 0 to w.
    g = f"<g transform='translate({WIDE_X0} 0)'>"
    if icon is None:
        return g + s + text(word, w / 2, cy, cap) + "</g>"
    # symbol on the left, word on the right, as one centred group
    tw = 10.2 * len(word) * cap / 16  # measured: Barlow Condensed Bold caps at a 16-unit cap height
    iw = {"ENTER": 21, "SHIFT": 22, "TAB": 19}[word]  # drawn width of the symbol (the return arrow head reaches 6 left of its x)
    gw = iw + 8 + tw
    x0 = (w - gw) / 2
    return g + s + icon(x0) + text(word, x0 + iw + 8 + tw / 2, cy, cap) + "</g>"


def enter_key():
    return wide_key("ENTER")


def shift_key():
    return wide_key("SHIFT")


# ── clusters: inverted T of small keys, lit keys white, the rest grey ────────

# 17x22 keys, 5 apart, the block centred: 61 wide, 49 tall.
CL = {"up": (23.5, 7.5), "left": (1.5, 34.5), "down": (23.5, 34.5), "right": (45.5, 34.5)}
WASD = {"up": "W", "left": "A", "down": "S", "right": "D"}


def cluster(lit, legends=None):
    out = []
    for d, (x, y) in CL.items():
        on = d in lit
        s, cy = keycap(x, y, 17, 22, r=4.5, lit=on, sw=2.5, lip=4)
        out.append(s)
        out.append(text(legends[d], x + 8.5, cy, 9.5) if legends else tri(x + 8.5, cy, 8.5, d))
    return "".join(out)


# ── mouse ────────────────────────────────────────────────────────────────────

def mouse_body(x, y, w, h, pressed=None, wheel=False, sw=SW):
    r = w / 2
    cx = x + w / 2
    split = y + h * 0.44
    body = (f"M{x} {y + r} A{r} {r} 0 0 1 {x + w} {y + r} V{y + h - r * 0.9} "
            f"A{r} {r * 0.9} 0 0 1 {x} {y + h - r * 0.9} Z")
    out = [f"<path d='{body}' fill='{PLATE}' stroke='{INK}' stroke-width='{sw}' stroke-linejoin='round'/>"]
    if pressed in ("left", "right"):
        px, ex = (x, cx) if pressed == "left" else (cx, x + w)
        arc = (f"M{x} {split} V{y + r} A{r} {r} 0 0 1 {cx} {y} V{split} Z" if pressed == "left"
               else f"M{cx} {split} V{y} A{r} {r} 0 0 1 {x + w} {y + r} V{split} Z")
        out.append(f"<path d='{arc}' fill='{INK}'/>")
    out.append(f"<path d='M{x} {split} H{x + w} M{cx} {y} V{split}' stroke='{INK}' stroke-width='{sw * 0.8}' fill='none'/>")
    ww, wh = w * 0.2, h * 0.24
    out.append(f"<rect x='{cx - ww / 2:.2f}' y='{y + h * 0.1:.2f}' width='{ww:.2f}' height='{wh:.2f}' rx='{ww / 2:.2f}' "
               f"fill='{INK if (wheel or pressed == 'middle') else PLATE}' stroke='{INK}' stroke-width='{sw * 0.7}'/>")
    return "".join(out)


def mouse_button(side):
    # body leans away from the burst so the click lines have room in the corner
    if side == "left":
        body = mouse_body(22, 8, 36, 54, pressed="left")
        burst = "M9 8 L14 13 M4 21 H11 M18 2 V8"
    else:
        body = mouse_body(6, 8, 36, 54, pressed="right")
        burst = "M55 8 L50 13 M60 21 H53 M46 2 V8"
    return haloed(burst, 3) + body


def chevron(cx, cy, d, w=11, h=6):
    s = -1 if d == "up" else 1
    return f"M{cx - w} {cy - s * h / 2} L{cx} {cy + s * h / 2} L{cx + w} {cy - s * h / 2}"


def mouse_wheel(d):
    if d == "up":
        body = mouse_body(14, 17, 36, 45, wheel=True)
        ch = chevron(32, 5.5, "up") + " " + chevron(32, 12.5, "up")
    else:
        body = mouse_body(14, 2, 36, 45, wheel=True)
        ch = chevron(32, 51.5, "down") + " " + chevron(32, 58.5, "down")
    return body + haloed(ch, 3)


POS = {"up": (32, 8), "down": (32, 56), "left": (8, 32), "right": (56, 32)}


def mouse_move(dirs):
    out = [mouse_body(23, 19, 18, 26)]
    for d, (x, y) in POS.items():
        on = d in dirs
        out.append(tri(x, y, 12, d, PLATE if on else OFF, 2.5))
    return "".join(out)


def mouse_rotate(cw):
    # three-quarter ring around the mouse, open at the top, with a large arrow head at the opening
    out = [mouse_body(23.5, 21, 17, 24)]
    if cw:
        ring = "M13 21 A23 23 0 1 0 30 9.5"
        head = "<polygon points='28,1.5 41,9.5 28,17.5' fill='#fff' stroke='#000' stroke-width='2.5' stroke-linejoin='round'/>"
    else:
        ring = "M51 21 A23 23 0 1 1 34 9.5"
        head = "<polygon points='36,1.5 23,9.5 36,17.5' fill='#fff' stroke='#000' stroke-width='2.5' stroke-linejoin='round'/>"
    return haloed(ring, 3.5) + head + "".join(out)


ALL = ["up", "down", "left", "right"]
S = {
    4: ("D-pad up", "▲ key", arrow_key("up")),
    5: ("D-pad down", "▼ key", arrow_key("down")),
    6: ("D-pad left", "◀ key", arrow_key("left")),
    7: ("D-pad right", "▶ key", arrow_key("right")),
    8: ("D-pad", "arrow keys", cluster([])),
    9: ("D-pad, any", "arrow keys", cluster(ALL)),
    10: ("D-pad up / down", "▲ ▼ keys", cluster(["up", "down"])),
    11: ("D-pad left / right", "◀ ▶ keys", cluster(["left", "right"])),
    12: ("L stick up", "W", single("W")),
    13: ("L stick down", "S", single("S")),
    14: ("L stick left", "A", single("A")),
    15: ("L stick right", "D", single("D")),
    16: ("L stick click", "SHIFT", shift_key()),
    17: ("L stick", "WASD", cluster(ALL, WASD)),
    18: ("L stick up / down", "W S", cluster(["up", "down"], WASD)),
    19: ("L stick left / right", "A D", cluster(["left", "right"], WASD)),
    20: ("L stick rotate ↺", "WASD", cluster(ALL, WASD)),
    21: ("R stick up", "mouse up", mouse_move(["up"])),
    22: ("R stick down", "mouse down", mouse_move(["down"])),
    23: ("R stick left", "mouse left", mouse_move(["left"])),
    24: ("R stick right", "mouse right", mouse_move(["right"])),
    25: ("R stick click", "CTRL", wide_key("CTRL")),
    26: ("R stick", "mouse move", mouse_move(ALL)),
    27: ("R stick up / down", "mouse ▲ ▼", mouse_move(["up", "down"])),
    28: ("R stick left / right", "mouse ◀ ▶", mouse_move(["left", "right"])),
    29: ("R stick rotate ↺", "mouse circle", mouse_rotate(False)),
    30: ("A", "ENTER", enter_key()),
    31: ("B", "ESC", single("ESC")),
    32: ("X", "Q", single("Q")),
    33: ("Y", "E", single("E")),
    34: ("LB", "wheel down", mouse_wheel("down")),
    35: ("LT", "right button", mouse_button("right")),
    36: ("RB", "wheel up", mouse_wheel("up")),
    37: ("RT", "left button", mouse_button("left")),
    38: ("Menu", "P", single("P")),
    39: ("View", "TAB", wide_key("TAB")),
    58: ("L stick rotate ↻", "WASD", cluster(ALL, WASD)),
    59: ("R stick rotate ↻", "mouse circle", mouse_rotate(True)),
}


WIDTH = {30: WIDE['ENTER'], 16: WIDE['SHIFT'], 25: WIDE['CTRL'], 39: WIDE['TAB']}


def svg(n, size):
    w = WIDTH.get(n, 64)
    return f"<svg viewBox='{WIDE_X0 if w > 64 else 0} 0 {w} 64' width='{size * w / 64:g}' height='{size}' role='img' aria-label='Main_Button_{n}'>{S[n][2]}</svg>"


GROUPS = [("Face, menu and shoulder buttons", [30, 31, 32, 33, 38, 39, 37, 35, 36, 34]),
          ("D-pad", [4, 5, 6, 7, 8, 9, 10, 11]),
          ("Left stick", [12, 13, 14, 15, 17, 18, 19, 20, 58, 16]),
          ("Right stick", [21, 22, 23, 24, 26, 27, 28, 29, 59, 25])]


def master_svg():
    """All designs as <symbol id="Main_Button_N">, laid out as a labelled sheet so the file opens as a picture."""
    cell_w, cell_h, cols = 128, 112, 10
    rows = sum((len(ns) + cols - 1) // cols for _, ns in GROUPS)
    height = 40 + rows * cell_h + len(GROUPS) * 28
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' "
           f"width='{cols * cell_w + 32}' height='{height}' viewBox='0 0 {cols * cell_w + 32} {height}'>",
           "<!-- Master copy of the XCL keyboard/mouse icons for platform_xbsx.input_icons (Main_Button_N).",
           "     Generated by scripts/kbm_icons/keycaps.py. build_library.py reads the <symbol> elements; the rest",
           "     is only the contact sheet. Changed an icon? Post your updated master file in the PR (see",
           "     docs/kbm-icons/README.md). -->",
           "<defs>"]
    for n in sorted(S):
        w = WIDTH.get(n, 64); x0 = WIDE_X0 if w > 64 else 0
        pad, key, body = S[n]
        out.append(f"<symbol id='Main_Button_{n}' viewBox='{x0:g} 0 {w} 64' data-pad='{pad}' data-key='{key}'>{body}</symbol>")
    out.append("</defs>")
    out.append(f"<rect width='100%' height='100%' fill='#20202c'/>")
    out.append("<text x='16' y='26' fill='#f9f9f9' font-family='Segoe UI, sans-serif' font-size='16' font-weight='600'>"
               "XCL keyboard/mouse icons: master copy (platform_xbsx.input_icons)</text>")
    y = 40
    for title, ns in GROUPS:
        out.append(f"<text x='16' y='{y + 18}' fill='#e6e6e6' font-family='Segoe UI, sans-serif' font-size='12' "
                   f"letter-spacing='1'>{title.upper()}</text>")
        y += 28
        for i, n in enumerate(ns):
            if i and i % cols == 0: y += cell_h
            w = WIDTH.get(n, 64); x0 = WIDE_X0 if w > 64 else 0
            cx = 16 + (i % cols) * cell_w
            out.append(f"<use xlink:href='#Main_Button_{n}' href='#Main_Button_{n}' x='{cx + (cell_w - 16 - w) / 2:g}' y='{y}' "
                       f"width='{w}' height='64'/>")
            out.append(f"<text x='{cx + (cell_w - 16) / 2:g}' y='{y + 84}' fill='#f9f9f9' text-anchor='middle' "
                       f"font-family='Segoe UI, sans-serif' font-size='11'>#{n} {S[n][0]}</text>")
            out.append(f"<text x='{cx + (cell_w - 16) / 2:g}' y='{y + 98}' fill='#e6e6e6' text-anchor='middle' "
                       f"font-family='Segoe UI, sans-serif' font-size='11'>{S[n][1]}</text>")
        y += cell_h
    out.append("</svg>")
    return "\n".join(out) + "\n"


def preview_html():
    css = ("body{margin:0}.g{padding:16px 24px 4px}.g h4{margin:0 0 10px;font:600 12px/1 var(--font-hint);letter-spacing:.06em;"
           "text-transform:uppercase;color:var(--hint-ink)}.row{display:flex;flex-wrap:wrap;gap:14px 18px}"
           ".t{display:flex;flex-direction:column;align-items:center;gap:6px;min-width:88px}.t svg{display:block}"
           ".t b{font:600 12px/1.2 var(--font-hint);color:var(--prompt-ink)}.t small{font:12px/1.2 var(--font-hint);color:var(--hint-ink);text-align:center}"
           ".live{display:flex;justify-content:flex-end;gap:21px;padding:18px 40px 22px;align-items:center}"
           ".live .xp-item svg{height:27px;flex:none}")
    body = []
    for title, ns in GROUPS:
        body.append(f"<section class='g'><h4>{title}</h4><div class='row'>")
        for n in ns:
            body.append(f"<div class='t'>{svg(n, 54)}<b>#{n} · {S[n][0]}</b><small>{S[n][1]}</small></div>")
        body.append("</div></section>")
    live = ("<div class='live'><span class='xp-item'><span class='xp-label'>Restore default</span>" + svg(32, 27) +
            "</span><span class='xp-item'><span class='xp-label'>Aim</span>" + svg(35, 27) +
            "</span><span class='xp-item'><span class='xp-label'>Continue</span>" + svg(30, 27) +
            "</span><span class='xp-item'><span class='xp-label'>Back</span>" + svg(31, 27) + "</span></div>")
    return ("<!-- Icon library: keycap replacements for platform_xbsx.input_icons. Generated by scripts/kbm_icons/keycaps.py -->\n"
            "<!doctype html><html><head><link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">"
            "<link href=\"https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700&family=Figtree:wght@400;600&display=swap\" rel=\"stylesheet\">"
            "<link rel=\"stylesheet\" href=\"../../tokens.css\"><link rel=\"stylesheet\" href=\"../bundle.css\">"
            f"<style>{css}</style></head><body><div class='xp-surface-account'>" + "".join(body) + live + "</div></body></html>\n")


if __name__ == "__main__":
    import os
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    docs = os.path.join(root, "docs", "kbm-icons")
    with open(os.path.join(docs, "icons-master.svg"), "w", encoding="utf-8", newline="\n") as f: f.write(master_svg())
    with open(os.path.join(docs, "design-system", "components", "IconLibrary", "preview.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(preview_html())
    print(f"wrote {len(S)} designs: docs/kbm-icons/icons-master.svg, design-system/components/IconLibrary/preview.html")
