# XCL triage dashboard

RepoPulse is a browser-only dashboard for the public issues, pull requests, commits, and the
maintained triage digest in [xcl-dev/xcl#1607](https://github.com/xcl-dev/xcl/issues/1607).
It stores personal pins, notes, and triage status only in the visitor's browser. It never asks for
or embeds a GitHub token.

## Run Locally

**Prerequisites:**  Node.js


1. `npm ci`
2. `npm test`
3. `npm run lint`
4. `npm run dev`

Production builds use `npm run build`. Vite emits `dist/` with the `/xcl/triage/` base path required
by the project Pages URL. The repository workflow publishes that artifact after a push to `main`.
