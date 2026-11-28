A keyboard key drawn as a rounded square, the size of the title's pad disc.

Provide the legend as text: one character (`Q`, `E`, `P`), a word in capitals (`ENTER`, `ESC`, `TAB`, `SHIFT`, `CTRL`) or an arrow triangle with `xp-key--arrow`. Markup: `<span class="xp-glyph xp-key">ENTER</span>`.

- Use the primary key from the README's mapping table: `ENTER` for A, `ESC` for B, `Q` for X, `E` for Y.
- For Q, E and WASD, the legend is the character the user's layout has at that position, because XCL reads those keys by scancode.
- Group keys that act together (WASD) in `.xp-keys`, 4px apart.
- Don't round a keycap into a disc, outline it, or add a shadow.
