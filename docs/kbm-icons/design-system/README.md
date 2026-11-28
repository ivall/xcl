Keyboard-and-mouse button prompts for the retail title running under XCL. The title draws Xbox face-button discs (A, B, X, Y) next to its action labels. This system gives each pad input the key or mouse button that XCL actually maps to it, drawn to the same grid as the title's prompts, so a key prompt can replace a pad prompt in place.

Every value here was measured in the title's own 3840×2160 frames: the account prompt, Navigation Experience, Subtitles, Brightness and Legal Notices screens. Lengths are written at 1080p; multiply by screen height ÷ 1080.

## Which key a prompt shows

The key is the one XCL presses the pad button with (`crates/xcl/src/hle/gameinput.rs` `keyboard()` and `crates/xcl/src/hle/kbm.rs` `apply()`, upstream `main` 930e8b4). Show the **primary** key; the alternate also works but is never drawn.

| Pad input | Primary prompt | Also works | How XCL reads it |
|---|---|---|---|
| A | `ENTER` keycap | Space | virtual key |
| B | `ESC` keycap | Backspace | virtual key |
| X | `Q` keycap | – | physical scancode 0x10 |
| Y | `E` keycap | – | physical scancode 0x12 |
| View | `TAB` keycap | – | virtual key |
| Menu | `P` keycap | – | virtual key |
| D-pad up / down / left / right | `▲` `▼` `◀` `▶` arrow keycaps | – | virtual key |
| Left bumper (LB) | mouse wheel down | – | one press per notch |
| Right bumper (RB) | mouse wheel up | – | one press per notch |
| Left trigger (LT) | right mouse button | – | full pull |
| Right trigger (RT) | left mouse button | – | full pull |
| Left stick click | `SHIFT` keycap (left) | – | virtual key |
| Right stick click | `CTRL` keycap (left) | middle mouse button | virtual key |
| Left stick | `W` `A` `S` `D` keycaps | – | physical scancodes |
| Right stick | mouse move glyph | – | raw mouse counts |

- **Physical keys follow the layout.** Q, E and WASD are read by position. Label those keycaps with the character the user's layout puts there: on AZERTY the X prompt shows `A` and the left stick shows `Z Q S D`.
- **Triggers are crossed on purpose.** RT (fire) is the left mouse button and LT (aim) is the right one. Never draw RT as the right mouse button.

### The startup screens, translated

| Screen | Title shows | Show instead |
|---|---|---|
| Account prompt | ACCOUNT PICKER (Y) · SCREEN NARRATION (X) · PLAYER1 (A) | ACCOUNT PICKER `E` · SCREEN NARRATION (OFF) `Q` · PLAYER1 `ENTER` |
| Navigation Experience | CONTINUE (A) | CONTINUE `ENTER` |
| Subtitles | CONTINUE (A) · BACK (B) | CONTINUE `ENTER` · BACK `ESC` |
| Brightness | RESTORE DEFAULT (X), disabled · CONTINUE (A) · BACK (B), stacked | RESTORE DEFAULT `Q`, disabled · CONTINUE `ENTER` · BACK `ESC`, stacked |
| Legal Notices | BACK (B) | BACK `ESC` |

## Content

- Labels are the title's own words, in capitals: CONTINUE, BACK, RESTORE DEFAULT, PLAYER1. Never reword a label to name the key ("PRESS ENTER"): the glyph says which key.
- Keycap legends are capitals: `ENTER`, `ESC`, `TAB`, `SHIFT`, `CTRL`, `BKSP`, `SPACE`. Single letters stay single letters. Arrows are solid triangles, not the ↑ arrow characters.
- The footer hint is sentence case and ends with a full stop ("More options available in the Settings menu.").

## Visual foundations

- **Glyphs.** Every glyph is `glyph-size` (27px) tall, filled `prompt-ink`, with its legend in `glyph-legend`. The shape tells the device apart:
  - pad buttons: a disc (`radius-disc`);
  - keys: a rounded square (`radius-keycap`), 27px wide for one character and wider for a word, padded by `keycap-pad`;
  - mouse: a 20×27 body (`radius-mouse`) with the pressed zone filled `glyph-legend` (assets in the Mouse group).
  
  Never draw a key as a disc: a round `E` reads as a pad button.
- **Prompt item.** The label comes first and the glyph second, separated by `label-glyph-gap`. Label and glyph share a centre line: the label's 18.5px caps sit centred on the 27px glyph.
- **Footer bar.** Items are right-aligned, `item-gap` apart. The last glyph is `safe-right` from the screen edge, and glyphs sit `safe-bottom` above the bottom edge. A hint sentence, if any, sits at `safe-left` on the same line in `hint`.
- **Stack.** When the screen's own panel is in the middle (Brightness), the prompts stack right-aligned at `stack-pitch`. The order is top to bottom: secondary (X / `Q`), confirm, back.
- **Disabled.** A prompt that does nothing yet stays visible: label and glyph fill `disabled-fill`, with a 1.5px `glyph-legend` outline. This pair is the title's own and reaches only 1.9:1 on `surface-calibration`. Keep it exact; the outline carries the shape.
- **Grounds.** The prompts sit on the title's own screens: `surface-account`, the `surface-settings` diagonal, and `surface-calibration`. `prompt-ink` holds at least 8.5:1 on each.
- **Type.** Set labels in `prompt-label`, legends in `keycap-legend` and hints in `hint`. Barlow Condensed and Figtree stand in for the title's own faces, which are not copied into this system. In the game itself, use the title's label face so key prompts match its pad prompts.
- No shadows, no gradients on glyphs, no animation: the title's prompts have none.

## Iconography

- Pad discs and keycaps are CSS shapes (`.xp-disc`, `.xp-key` in `components/bundle.css`) holding text legends; there are no icon files for them.
- Mouse glyphs are the six SVGs in the Mouse group, drawn for this system: left, right, middle, wheel up, wheel down, move.
- The title's own Xbox glyph art is not included here.
- **In the game**, prompts come from the title's icon library, `platform_xbsx.input_icons`: 38 vector symbols named `Main_Button_N`. The Icon library card draws a keycap replacement for each one. It follows the library's 64-unit grid (white plate, 3-unit black outline, black legend), not the 27px CSS glyphs above.
