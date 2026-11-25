import type { IssueTemplate } from '../types/github';

export const DEFAULT_REPO_SLUG = 'xcl-dev/xcl';

export const ISSUE_TEMPLATES: IssueTemplate[] = [
  {
    id: 'bug',
    label: 'Bug Report',
    defaultLabels: ['bug'],
    titlePrefix: 'Bug: ',
    body: `### Summary\nDescribe what went wrong clearly.\n\n### Steps to Reproduce\n1. Build / commit SHA:\n2. Title or test case:\n3. Observed behavior:\n\n### Expected Behavior\nDescribe what should happen instead.`,
  },
  {
    id: 'recompiler',
    label: 'Recompiler / Shader',
    defaultLabels: ['recompiler'],
    titlePrefix: 'Recompiler: ',
    body: `### Subsystem\nRecompiler / RDNA2 -> DXIL lowering\n\n### Instruction or Pass\nMention the opcode, predicate, or pass involved.\n\n### Regression Commit (if known)\nCommit SHA or PR #`,
  },
  {
    id: 'feature',
    label: 'Enhancement / Tooling',
    defaultLabels: ['enhancement'],
    titlePrefix: 'Scripts: ',
    body: `### Goal\nDescribe the workflow or feature improvement.\n\n### Proposed Change\n- Key details\n- Edge cases to cover`,
  },
  {
    id: 'docs',
    label: 'Documentation',
    defaultLabels: ['documentation'],
    titlePrefix: 'Docs: ',
    body: `### Page or Section\nWhich doc file or guide needs updating?\n\n### Proposed Clarification\nSummarize what should be added or corrected.`,
  },
];

export const QUICK_REPLY_SNIPPETS: Array<{ label: string; text: string }> = [
  {
    label: 'Request Repro / SHA',
    text: 'Thanks for reporting! Could you share the exact commit SHA, GPU/driver version, and minimal reproduction steps so we can bisect this?',
  },
  {
    label: 'Needs Rebase',
    text: 'Changes look solid overall—please rebase onto latest `main` so CI runs cleanly against recent merges.',
  },
  {
    label: 'LGTM / Ready',
    text: 'Reviewed and verified locally. Looks good to merge once checks pass.',
  },
  {
    label: 'Link Related Issue',
    text: 'Tracking alongside related work in # — let us know if this resolves the root cause or needs a follow-up.',
  },
];
