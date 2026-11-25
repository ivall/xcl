---
name: xcl-xgameruntime
description: >-
  Implement XCL's offline XGameRuntime: GDK thunks, XUser, XGameSave, XPackage,
  task queues, and XAsync. Use when report.json xgameruntime lists a missing
  slot, or when editing hle/xgameruntime/.
---

# XGameRuntime

`hle/xgameruntime/`. The title calls through the GDK thunk table (`gdk_table.rs`, `api.rs`), not only named exports. `report.json` `xgameruntime` records which class it asked for and which XCL slot returned unimplemented.

Implemented offline path: XUser, XGameSave (`gamesave.rs`: `<data>\gamesave\<configId>\<container>\b\<blob>`, `%XX` names, temp-file replace, 256 MB cap), PLS, XPackage, XGame title id, XSystem (the title sees Series X or S), task queues (`taskqueue.rs`), XAsync (`xasync.rs`). A blob enumeration is the listing taken at the start of that call, so an update submitted while it runs is not mixed into the result. `XCL_MUTE_TTS=1` answers XAccessibility from XCL and does not speak. Networking that needs the PC runtime goes to the system `xgameruntime.dll` where the code already does. Live, achievements, and store stay unavailable: dialogs return `E_ABORT` (`stubdb.toml`). Do not return a purchased license.

`XGameGetXboxTitleId` reads `title_id` or `<TitleId>` in `MicrosoftGame.Config`.

`gdk_table.rs` is generated (`scripts/gen_gdk_table.py` from the public PC GDK thunk library). Do not edit it. New behavior goes in `api.rs`, which walks that table. Add a test that calls the thunk the way `tests/testtitle` does. Return `E_NOTIMPL` only when the feature is intentionally offline.

`xsign.rs` builds offline tokens for HTTP the title signs locally. It is not an Xbox Live login.
