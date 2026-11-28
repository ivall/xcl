"""Prompt motion preview: the title's own prompt-icon animations (class InputIconV2), frame for frame at
30 fps, played on the keycaps. The tables are the title's; docs/kbm-icons/README.md says where they come from.

    python scripts/kbm_icons/motion.py   # writes docs/kbm-icons/design-system/components/PromptMotion/preview.html
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keycaps as k

FPS = 30
PRESSED_SCALE = 0.8
PRESSED_EASE = [0.407, 0.788, 0.918, 0.981, 1]
ENABLED_EASE = [0.223, 0.506, 0.758, 0.933, 1]
MASH_EASE = [1, 0.9554, 0.8484, 0.8, 0.8506, 0.9374, 0.9744, 0.99, 0.9976]
HOLD_EASE = [1] * 14 + [0.996, 0.9646, 0.919, 0.885, 0.8626, 0.8424, 0.8314, 0.823, 0.816, 0.81, 0.806, 0.804, 0.802] + [0.8] * 31
HINT = [1] * 30 + [0.9554, 0.8484, 0.8, 0.8354, 0.9234, 0.9616, 0.9816, 0.9926, 0.998, 1]
WIGGLE = [1.52, 3, -1.55, -6, -3.32, 3.1, 7, 5.24, 2.96, 0.85, 0]

SIZE = 54  # px for 64 units: the 2160p size


def press_tint(rest=15, held=10):
    return [0.0] * rest + [1.0] * held + [1 - e for e in PRESSED_EASE]


def press_cycle(rest=15, held=10):
    return [1.0] * rest + [PRESSED_SCALE] * held + [PRESSED_SCALE + (1 - PRESSED_SCALE) * e for e in PRESSED_EASE]


def keyframes(name, frames, prop):
    n = len(frames); out = []
    for i, v in enumerate(frames):
        out.append(f"{i * 100 / n:.3f}%{{transform:{prop(v)}}}")
    out.append(f"100%{{transform:{prop(frames[0])}}}")
    return f"@keyframes {name}{{{''.join(out)}}}", n / FPS


def curve(frames, lo, hi, w=176, h=44, label_lo="", label_hi=""):
    n = len(frames)
    pts = " ".join(f"{i * (w - 2) / max(n - 1, 1) + 1:.1f},{h - 4 - (v - lo) / (hi - lo) * (h - 8):.1f}" for i, v in enumerate(frames))
    return (f"<svg class='cv' viewBox='0 0 {w} {h}' width='{w}' height='{h}' aria-hidden='true'>"
            f"<line x1='0' x2='{w}' y1='{h - 4 - (1 - lo) / (hi - lo) * (h - 8):.1f}' y2='{h - 4 - (1 - lo) / (hi - lo) * (h - 8):.1f}' class='base'/>"
            f"<polyline points='{pts}' class='ln'/></svg>")


def disabled_q():
    s, cy = k.keycap(1.5, 1.5, 61, 61, lit=False)
    return f"<svg viewBox='0 0 64 64' width='{SIZE}' height='{SIZE}' role='img' aria-label='Q, disabled'>{s}{k.text('Q', 32, cy, 24)}</svg>"


demos = []


def demo(key, title, what, timing, frames, icon_svg, prop, lo=0.75, hi=1.05, tint=None):
    kf, dur = keyframes(f"m-{key}", frames, prop)
    if tint:
        # the title's colour transform: rest = white x COLOR_PROMPT_NEUTRAL (#F9F9F9); pressed = + COLOR_PROMPT_PRESSED (#ABABAB)
        n = len(tint)
        kf += f"@keyframes t-{key}{{" + "".join(f"{i * 100 / n:.3f}%{{opacity:{v:.3f}}}" for i, v in enumerate(tint)) + f"100%{{opacity:{tint[0]}}}}}"
        icon_svg = (f"<div class='lay' style='filter:url(#neutral)'>{icon_svg}</div>"
                    f"<div class='lay' style='filter:url(#pressed);animation:t-{key} {dur:.3f}s linear infinite'>{icon_svg}</div>")
    demos.append((kf, f"<figure class='d'><div class='stage'><div class='ic' style='animation:m-{key} {dur:.3f}s linear infinite'>{icon_svg}</div></div>"
                      f"<figcaption><b>{title}</b><span>{what}</span><code>{timing}</code>{curve(frames, lo, hi)}</figcaption></figure>"))


scale = lambda v: f"scale({v:.4f})"
demo("press", "Press", "Snaps to 80 % and turns its black to #ABABAB on the frame the key goes down; the lip vanishes into the face. On release both spring back.",
     "press: 0 ms · release: 5 frames, 167 ms", press_cycle(), k.svg(30, SIZE), scale, tint=press_tint())
demo("wide", "Press, wide key", "Wide keys run from -6 to 82 so they clear their label; they shrink about the same point as every icon.",
     "same as Press", press_cycle(), k.svg(16, SIZE), scale, tint=press_tint())
demo("hold", "Hold", "Waits 14 frames, then sinks to 80 % over the next 13 and stays there. Repeats while held.",
     "58-frame loop, 1.93 s", HOLD_EASE, k.svg(33, SIZE), scale)
demo("mash", "Mash", "A quick dip to 80 % and back, once per press. Loops while the prompt asks for tapping.",
     "9-frame loop, 300 ms", MASH_EASE, k.svg(32, SIZE), scale)
demo("hint", "Hint", "Rests for one second, then dips to 80 % and back, to draw the eye.",
     "40-frame loop, 1.33 s", HINT, k.svg(31, SIZE), scale)
demo("wiggle", "Press while disabled", "Doesn't shrink. It shakes left and right, and the shake dies away.",
     "11 frames, 367 ms", WIGGLE + [0] * 19, disabled_q(),
     lambda v: f"translateX({v * SIZE / 64:.2f}px)", lo=-8, hi=8)

css = ("body{margin:0}.wrap{padding:20px 24px 24px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px 28px}"
       ".d{margin:0;display:flex;gap:16px;align-items:flex-start}"
       ".stage{flex:none;width:104px;height:72px;display:flex;align-items:center;justify-content:center}"
       ".ic{display:grid;transform-origin:50% 50%;will-change:transform}.ic svg{display:block}.lay{grid-area:1/1}"
       "figcaption{display:flex;flex-direction:column;gap:5px;min-width:0}"
       "figcaption b{font:600 14px/1.2 var(--font-hint);color:var(--prompt-ink)}"
       "figcaption span{font:13px/1.35 var(--font-hint);color:var(--hint-ink)}"
       "figcaption code{font:12px/1.2 ui-monospace,Consolas,monospace;color:var(--hint-ink)}"
       ".cv{margin-top:2px}.cv .base{stroke:#676767;stroke-width:1;stroke-dasharray:3 3}"
       ".cv .ln{fill:none;stroke:var(--prompt-ink);stroke-width:1.75;stroke-linejoin:round}"
       "@media (prefers-reduced-motion:reduce){.ic{animation:none!important}}")
html = ("<!-- Prompt motion: how the title animates a prompt icon, on the keycaps. Generated by scripts/kbm_icons/motion.py -->\n"
        "<!doctype html><html><head><link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">"
        "<link href=\"https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700&family=Figtree:wght@400;600&display=swap\" rel=\"stylesheet\">"
        "<link rel=\"stylesheet\" href=\"../../tokens.css\"><link rel=\"stylesheet\" href=\"../bundle.css\">"
        f"<style>{css}{''.join(kf for kf, _ in demos)}</style></head><body>"
        "<svg width='0' height='0' style='position:absolute' aria-hidden='true'><defs>"
        "<filter id='neutral' color-interpolation-filters='sRGB'><feComponentTransfer>"
        "<feFuncR type='linear' slope='0.9765'/><feFuncG type='linear' slope='0.9765'/><feFuncB type='linear' slope='0.9765'/></feComponentTransfer></filter>"
        "<filter id='pressed' color-interpolation-filters='sRGB'><feComponentTransfer>"
        "<feFuncR type='linear' intercept='0.6706'/><feFuncG type='linear' intercept='0.6706'/><feFuncB type='linear' intercept='0.6706'/></feComponentTransfer></filter>"
        "</defs></svg><div class='xp-surface-account'><div class='wrap'>"
        + "".join(h for _, h in demos) + "</div></div></body></html>\n")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "kbm-icons", "design-system",
                   "components", "PromptMotion", "preview.html")
with open(out, "w", encoding="utf-8", newline="\n") as f: f.write(html)
print("wrote", os.path.normpath(out))
