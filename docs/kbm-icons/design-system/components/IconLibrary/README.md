The keycap replacement for the title's own button-icon library, `platform_xbsx.input_icons`: one 64×64 symbol for each `Main_Button_N` the title exports. XCL serves this library in place of the original, so every prompt in every menu shows the key instead of the pad glyph.

The library is a Scaleform movie holding 38 symbols. Each symbol is a few solid-fill vector shapes, with no bitmaps and no text. The replacement keeps each symbol's name and speaks the original's measured language:

- 64×64 units, origin top left. The title scales the symbol to `glyph-size` (54px at 2160p).
- White shapes (`#FFFFFF`) with a 3-unit black outline. Black marks: legends, arrows, a pressed button.
- Unlit parts are `#676767`, the grey of the title's own unlit d-pad arms.

## The parts

- **Keycap.** A rounded square (corner 12) with a lip: the white face sits on a grey band 8 units deep along the bottom edge, so a key reads as a key, never as a pad disc. Legends centre on the face, not on the whole symbol. One letter has a 24-unit cap height; `ESC` has 18.
- **Wide keys.** `ENTER`, `SHIFT`, `TAB` and `CTRL` are 88 wide (1.4 keys) and 64 tall, and run from x = −6 to 82. Each carries its printed symbol before the word (↵ ENTER, ⇧ SHIFT, ⇥ TAB; `CTRL` has none), as one centred group with an 8-unit gap. Four-letter words have a 16-unit cap height, five-letter ones 14.
- **Why 88 and −6.** In the game every icon gets a fixed 64-unit slot, whatever the symbol's width. Measured in the title: a prompt's label ends 37 units left of the slot, and the next prompt's label starts 51 units right of it. A key from −6 to 82 keeps 31 units clear of its own label and 33 clear of the next prompt, so each label still reads with its own key. On the account prompt, PLAYER1 to ↵ ENTER is 16 px at 1440p, against 20 px for the title's A disc.
- **Cluster.** An inverted T of small keys (17×22, lip 4) with 5 units between them, the block centred in the 64 square. Keys you press are white, the others are grey with a black lip: the same lit/unlit rule as the title's d-pad.
- **Mouse.** A capsule with the button seam and wheel. A pressed button is filled black and gets three click lines in its corner. The wheel symbols fill the wheel and stack two chevrons on the side it turns to.
- **Mouse move.** A small mouse (18×26) with four triangles at the edges, about 9 units clear of the body, lit white in the directions that count and grey otherwise. Rotate has a ring arrow open at the top.
- **Halo.** Anything drawn outside a key or mouse body (click lines, chevrons, the ring) is a white stroke with a 3-unit black outline, so it holds on light and dark screens alike.

## Which symbol is which

| N | Pad input | Draw |
|---|---|---|
| 30 | A | ↵ `ENTER` wide key |
| 31 | B | `ESC` key |
| 32 | X | `Q` key |
| 33 | Y | `E` key |
| 38 | Menu | `P` key |
| 39 | View | ⇥ `TAB` wide key |
| 37 | RT | mouse, left button pressed |
| 35 | LT | mouse, right button pressed |
| 36 / 34 | RB / LB | mouse wheel, chevrons up / down |
| 4–7 | D-pad up, down, left, right | arrow key |
| 8 / 9 / 10 / 11 | D-pad none / any / up-down / left-right | arrow-key cluster |
| 12–15 | L stick up, down, left, right | `W` `S` `A` `D` key |
| 16 | L stick click | ⇧ `SHIFT` wide key |
| 17, 20, 58 | L stick, rotate ↺, rotate ↻ | `WASD` cluster, all lit |
| 18 / 19 | L stick up-down / left-right | `WASD` cluster, `W S` / `A D` lit |
| 21–24, 26–28 | R stick directions | mouse move, the matching triangles lit |
| 29 / 59 | R stick rotate ↺ / ↻ | mouse with a ring arrow |
| 25 | R stick click | `CTRL` wide key |

## Rules

- A symbol keeps its own `Main_Button_N` name and is 64 tall. All but the four wide keys stay 64×64. Nothing may be wider than 88 or start left of −6: the title doesn't make room for a wider icon, so it would run into the labels. A 112-wide `ENTER` was tried in the game and ran into PLAYER1.
- Tested in the title under XCL (account prompt): the keycaps draw, and pressing E or Enter shrinks and greys them as in the Prompt motion card. Other prompt screens aren't checked yet.
- Never use a disc: in this title a disc means a pad button.
- In the movie, every legend is a filled outline, not text: Scaleform draws the library with no font. Convert the Barlow Condensed letters to shapes when building it.
- For Q, E and WASD, build one library per keyboard layout. On AZERTY the X symbol is `A` and the cluster is `Z Q S D`.
