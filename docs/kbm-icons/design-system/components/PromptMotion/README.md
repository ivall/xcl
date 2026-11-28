How the title animates a prompt icon when its button is pressed, held, tapped or disabled, and how the keycaps take the same motion.

The motion is not in the icons. The title's `InputIconV2` class (in every menu and HUD movie that shows prompts) places a `Main_Button_N` symbol in a container and animates the container: its scale, its x position and its colour. The keycap library replaces only the symbols, so each keycap gets exactly the motion below with nothing added to it. Every value comes from that class's code. It runs at 30 frames a second and steps through a lookup table, one entry per frame.

## States

| State | What the icon does | Frames at 30 fps |
|---|---|---|
| Press | Snaps to 80 % scale (`PRESSED_SCALE` 0.8) on the frame the button goes down, with no easing, and stays there while it is held. | 0 |
| Release | Springs back to 100 % over `PRESSED_EASE` (0.407, 0.788, 0.918, 0.981, 1 of the way). | 5 (167 ms) |
| Hold prompt | Rests at 100 % for 14 frames, sinks to 80 % by frame 27 and holds there. The loop restarts after 58 frames. A ring (the "gutter") round the icon fills while the button is held. | 58 (1.93 s) |
| Mash prompt | Dips to 80 % and recovers: 1, 0.955, 0.848, 0.8, 0.851, 0.937, 0.974, 0.99, 0.998. Once per tap. | 9 (300 ms) |
| Hint | Still for 30 frames, then the mash dip, so a waiting prompt pulses about once a second. | 40 (1.33 s) |
| Press while disabled | Doesn't shrink: the icon shakes sideways by 1.5, 3, −1.6, −6, −3.3, 3.1, 7, 5.2, 3.0, 0.9, 0 units of the 64-unit grid. | 11 (367 ms) |
| Enable / disable | The ring fades in or out over `ENABLED_EASE` (0.223, 0.506, 0.758, 0.933, 1). | 5 (167 ms) |
| Complete | A glow appears behind the icon at 50 % opacity, sized 0.68× without a ring or 1× with one. | – |

## Colour while pressed

At rest the title multiplies the icon by `COLOR_PROMPT_NEUTRAL`, so the white plate shows as #F9F9F9 (`prompt-ink`). On press it drops that tint and adds `COLOR_PROMPT_PRESSED` (`prompt-pressed`, #ABABAB) as a colour offset. The plate goes to #FFFFFF, and the black legend and outline go to #ABABAB. On release both ease back over the same 5 frames. The ring takes `COLOR_THEME_ACCENT_PRIMARY`, then fades back over 16 frames.

The game engine supplies these colours at run time (`fwuiColorStore.GetRGB`); they are not in `data.rpf`. #F9F9F9 and #ABABAB were measured under XCL on the account prompt, with Y held, in a 2560×1440 window. Held, the icon's plate covers 0.62× its resting area, which matches a scale of 0.8. Its colours were back to rest 640 ms after release. The accent colour of the ring isn't measured.

On a keycap the offset does something useful: the grey lip (#676767 + #ABABAB) clips to white, so a pressed key loses its lip and looks pushed down. The press and wide-key demos show this.

## What this means for the keycaps

- **Pivot.** The container scales and shakes about the symbol's point (32, 32) (`ICON_OFFSET` −32, −32). A 64×64 symbol shrinks about its centre. A wide keycap runs from −6 to 82 to keep clear of its label (Icon library card), so its centre (38) is 6 units right of the pivot. When pressed it moves about 1 unit to the left, which isn't visible in the game.
- **Black must be black, white must be white.** The press adds #ABABAB to every colour, so pure black reads as the pressed grey and pure white stays white. Keep legends, outlines and click marks #000000 and faces #FFFFFF. Any other grey ends up between the two, and the lip grey (#676767) turns white on purpose.
- **No motion in the symbols.** The library's symbols are single-frame. Don't add timeline animation to a keycap: it would play on top of the title's own.
- **Ring around wide keys.** The hold ring is drawn for a 64-unit disc. Around a wide key it circles only the middle. That's untested in the game.
- **Reduced motion.** The preview stops looping when the system asks for reduced motion. The title itself has no such switch for these animations.
