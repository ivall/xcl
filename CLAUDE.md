# Claude Code

The instructions for agents in this repository are in `AGENTS.md`, and the rules for every
contributor are in `CONTRIBUTING.md`. Both apply to you. `AGENTS.md` is their short version and is
loaded here; read `CONTRIBUTING.md` before you commit or write a PR description:

@AGENTS.md

Specific to Claude Code:

- Plan a task that spans more than one subsystem or more than ~400 lines (plan mode), and propose
  how to split it into PRs before you write code.
- Run the tests and report the real output; do not stop at "the tests should pass".
- Fill in the **AI use** section of the PR description with Claude Code and your exact model name.
- For a runtime change, run `scripts/smoke.ps1` on main and on the PR's head commit and paste both
  `xcl-smoke` blocks; do not write or edit them by hand.
