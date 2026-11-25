---
name: xcl-vfs
description: >-
  Change XCL path translation, the copy-on-write overlay, or console
  DirectStorage (xcrdapi). Use when a title cannot open archives, report.json
  vfs.unmapped_paths is set, or when editing vfs.rs, vfs_hooks.rs, rpf.rs, or
  hle/dstorage.rs.
---

# Files

`vfs.rs` decides; `vfs_hooks.rs` patches KERNELBASE so CRT and Win32 opens hit that decision.

Mounts (`xcl.toml`, `{game}` and `{data}`): `GAME:`, `G:`, `D:` map into the game folder (`Target::Game`). `S:\` is persistent local storage (`pls\`), `T:\` is `temp\`. `Target::Pass` is a host path left alone.

Writes to a game file copy that file into `<data>\overlay\` and modify the copy. Deletes write a whiteout. `remove_dir` of a directory that still has a visible child (game folder or overlay, ignoring whiteouts) returns `ERROR_DIR_NOT_EMPTY` and does not whiteout the directory. `Decision::UseWithDisposition` rewrites `TRUNCATE_EXISTING` so the overlay file is not published before the open succeeds. A test that leaves the title directory changed is a failure.

The shared shader cache is an `own_folder`. A console mount must not rewrite it: with the cache on the PC drive that the console calls `D:`, the title would write DXIL into the overlay and miss what `xcl shaders --precompile` stored.

`xcl rpf` lists or extracts readable RPF9 records (`rpf.rs`, `rpf_probe.rs`). Archives over 1 GB use a bounded header probe; `--probe` is summary-only (no `--find`, `--extract`, `--out`, `--full`). Encrypted TOC or Oodle payloads stay unread. The title unpacks its own archives; do not add a decryptor.

## DirectStorage

`hle/dstorage.rs` is `xcrdapi.dll` (`DStorageGetFactory`). Layouts in that file header were measured on a shipping title: factory, queue, request (`0x48`), file, status array. The worker reads files through the VFS, copies memory, inflates zlib (`miniz_oxide`), then unpacks BCPack where `hle/bcpack` can (`decode`: BC1, BC3, BC4, BC5, BC6H, BC7). File-backed zlib may be inflated off the queue thread (`hle/dstorage/parallel.rs`) inside a staging budget. The queue thread still owns the reads, the cancellation checks, and retirement. BC2, mode 0, modes above 7, and the hardware swizzle are refused (`E_DECOMPRESSION`). Refusals are `dstorage.bcpack_refused`; streams the decoder ran are `dstorage.bcpack_unpacked`. `tables.rs` is generated. A destination page array entry that is 0 at `EnqueueRequest` is read again when the request runs, because titles fill that entry after enqueue. An array that already had its pages is kept as snapshotted. That later read uses the same path as the snapshot: memory the title has since unmapped fails the request with `E_ACCESS_VIOLATION` and does not fault the worker. An entry that is still 0 at that later read is `E_ACCESS_VIOLATION`. A new status array's slots start complete with `S_OK`; `EnqueueStatus` makes a slot `E_PENDING`. A request refused at enqueue is still numbered, so `CancelRequestsWithTag` covers it. A cancelled request, refused or not, does not change the status or the error record. The first failure's error record keeps that request's file offset, source size, and cancellation tag. `XCL_DSTORAGE_BCPACK_FILL=1` writes solid BCn blocks into a refused destination so the title's wait shows up, and the request still fails. Methods not listed return `E_NOTIMPL`.

Retail BCPack checks use `XCL_SAMPLE_BCPACK` (`<name>.bc<mode>` plus the reference decoder's `<name>.vic<mode>`). A unit test feeds a synthetic buffer through the queue.

A cutscene blocked on asset load is usually a refused request or an unmapped path, not a shader bug. Fix the request the report names. BC1, BC3, BC4, BC5, BC6H, and BC7 already unpack; do not treat those modes as unsupported.
