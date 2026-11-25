# xcl-gui – a window for XCL

`xcl-gui` runs `xcl.exe` for you, so a game can be started, watched and configured without a
command line. It is a program of its own: it starts `xcl run <game>` as you would, reads what it
prints and the `xcl.log` it writes, and edits the title's `xcl.toml`. Nothing of XCL's runtime is
linked into it or changed by it.

```powershell
cargo build --release -p xcl-gui      # target\release\xcl-gui.exe, next to xcl.exe
```

## What it does

- **Home:** the game's folder, its exe or its `MicrosoftGame.config`, chosen once and kept.
  *Play* runs it; *Stop* ends `xcl` and the game process it started. Under it, what is known of
  the title's last run and of this PC (the graphics adapter `xcl run` named).
- **Console:** what `xcl` prints, as it prints it, and how the run ended (the exit code and, for
  the known ones, what it means: device removed, device hung, out of memory, …).
- **Log (xcl.log):** the title's log, followed while the game runs.
- **Errors and warnings:** the `WARN` / `ERROR` lines of the log and the console lines that
  report a failure, on a tab of their own with a count.
- **Settings**
  - *Every run:* offline (`XCL_NETWORK=0`), `XCL_PIPELINE_THREADS`, `XCL_RUN_SECONDS`, any other
    variable as `NAME=value` lines, and arguments for the game.
  - *This title:* the settings of its `xcl.toml` that matter most – Series X / S, title memory
    (and `title_memory_fit`, shown only where the file has that key),
    GPU-visible memory, ray tracing, adapter, log level, window size, keyboard and mouse. The
    file's other keys, its comments and its order are kept; a key it does not have is added only
    when its value is not XCL's default. XCL writes a title's `xcl.toml` at its first run, so
    the section is empty until then.
- **Graphics and shaders (every run):** the shader translator (`XCL_DIRECT_DXIL`), the shader
  compiler's threads, the two disk caches, background pipelines and `XCL_DRED`. XCL has no
  resolution scale, V-Sync or frame limit to set.
- **XCL:** which `xcl.exe` and which source folder are used (found from where `xcl-gui.exe` is,
  or set by hand). *Update and build* runs `git pull --rebase --autostash` and then
  `cargo build --release -p xcl` in that folder, with a progress bar that follows cargo's own
  count; each step has a button of its own too. Only `xcl` is built: `xcl-gui.exe` is running
  and cannot be written over.
- **Shaders (on Home, under *Play*; its choices are on the XCL page):** *Precompile shaders* runs
  `xcl shaders <the game's shader package> --precompile --pipelines --used-only`: every shader
  of the package is compiled into XCL's cache (vertex and pixel too; `--only cs` is added when
  that is unticked) and the compute pipelines an earlier run used are built (all of them when
  `--used-only` is unticked), so a title does not compile shaders itself on a black screen or
  wait for the graphics driver's compiler
  while its world loads. The pass's own reports become a progress bar with the pipelines built
  and the time left, taken from the speed between its last reports, beside the button, with
  *Stop* next to it. It can be stopped and goes
  on later from where it was.
  What is precompiled belongs to one build of XCL, one graphics card and one driver. An update
  notes the commit before `git pull` and compares the shader translators' files
  (`crates/xcl/src/gpu`, `crates/xcl/build.rs`) after it: when they changed, the window says
  that the pass has to be run again, until one has finished.

Every view has a filter, *Follow* (stay at the end), *Copy* and *Clear*. Lines are selected with
a click, Shift+click or by dragging (the view scrolls when the pointer leaves it), Ctrl+A takes
all that is shown, and Ctrl+C or *Copy selection* copies them; the selected lines are also shown
in a box under the view, where part of a line can be selected.

## Where it keeps things

Under XCL's data root (`%LOCALAPPDATA%\XCL`, or `%XCL_DATA_ROOT%`):

- `gui\settings.toml` – the window's own settings;
- `gui\sessions\<date>_console.log`, `<date>_xcl.log`, `<date>_report.json` – what a run printed,
  and copies of its log and report (XCL itself keeps the last two runs). The newest 20 runs are
  kept. *Save what each run prints* turns it off.

The game folder is not written to, as with `xcl run`.

## Languages

English and Spanish; the button at the foot of the side column switches. Next to it, the theme:
the project's own colours (those of the pictures in `assets/readme`, the default), black, or light.

## Tested

Its own tests pass on Windows and Linux. The window was tried on Linux with a stand-in for
`xcl.exe`: the pages in the three themes, the settings, stop, saved sessions, update and build,
the precompile pass with its bar, and selecting and copying lines (click, Shift+click, drag with
scrolling, Ctrl+A, Ctrl+C).

On Windows (RTX 3080 Ti) it ran a real title, stopped it, followed its log and saved its
`xcl.toml`, and ran the precompile pass on the title's shader package, in the project's theme.

## Not tested

On Windows: selecting and copying lines, and the black and light themes. A precompile pass
followed to its end from the window.
