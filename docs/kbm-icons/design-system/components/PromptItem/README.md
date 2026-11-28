One action prompt: a capitals label followed by its glyph, centred on one line.

Provide the label text and one glyph (a pad disc, a Keycap or a MouseGlyph). Markup: `.xp-item` > `.xp-label` + glyph; add `xp-item--disabled` for an unavailable action.

- The label always comes first and the glyph second, `label-glyph-gap` apart.
- Keep the title's label text; the glyph names the key.
- A disabled prompt stays in place with `disabled-fill` and a black outline. Never hide it.
