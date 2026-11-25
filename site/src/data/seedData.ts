import type { GitHubCommit, GitHubItem, IssueTemplate, RepoBundle, RepoInfo } from '../types/github';

export const DEFAULT_REPO_SLUG = 'xcl-dev/xcl';

export const XCL_FALLBACK_REPO: RepoInfo = {
  full_name: 'xcl-dev/xcl',
  owner: 'xcl-dev',
  name: 'xcl',
  description:
    'Xbox Series compatibility layer for Windows - runs console game code natively, reimplements the console system libraries and recompiles RDNA2 shaders to DXIL',
  html_url: 'https://github.com/xcl-dev/xcl',
  stargazers_count: 7833,
  forks_count: 1039,
  open_issues_count: 1337,
  default_branch: 'main',
  language: 'Rust',
  pushed_at: '2026-11-25T00:26:37Z',
};

export const XCL_FALLBACK_ISSUES: GitHubItem[] = [
  {
    id: 4001489,
    number: 1489,
    kind: 'issue',
    title: 'Recompiler: float compare-and-swap atomics fail IEEE-754 negative zero equality',
    body: '### Summary\nDuring shader execution, `fcmpswap` treats `-0.0` and `+0.0` inconsistently when lowered to DXIL.\n\n### Expected\nMatch IEEE-754 equality rules for float compare-and-swap.\n\n### Linked Fix\nPR #2011 addresses this check in the recompiler pass.',
    state: 'open',
    html_url: 'https://github.com/xcl-dev/xcl/issues/1489',
    user: { login: 'Decodere', avatar_url: '', html_url: 'https://github.com/Decodere' },
    labels: [
      { id: 1, name: 'bug', color: 'd73a4a' },
      { id: 2, name: 'recompiler', color: '1d76db' },
    ],
    assignees: [],
    comments: 4,
    created_at: '2026-11-24T18:10:00Z',
    updated_at: '2026-11-25T00:41:12Z',
    closed_at: null,
  },
  {
    id: 4001988,
    number: 1988,
    kind: 'issue',
    title: 'Shaders: DXIL validation error on root signature descriptor heap overflow',
    body: '### Environment\n- Windows 11 24H2\n- DirectX Shader Compiler latest\n\n### Details\nTitles binding >64 SRVs in a single compute dispatch hit `D3D12 ERROR` during PSO creation.',
    state: 'open',
    html_url: 'https://github.com/xcl-dev/xcl/issues/1988',
    user: { login: 'luckyteagirl', avatar_url: '', html_url: 'https://github.com/luckyteagirl' },
    labels: [
      { id: 3, name: 'shaders', color: '0e8a16' },
      { id: 4, name: 'dxil', color: '5319e7' },
    ],
    assignees: [],
    comments: 7,
    created_at: '2026-11-24T14:22:00Z',
    updated_at: '2026-11-24T23:50:00Z',
    closed_at: null,
  },
  {
    id: 4001975,
    number: 1975,
    kind: 'issue',
    title: 'Kernel: NtQueryVirtualMemory returns incomplete allocation base for title heaps',
    body: 'Memory-mapped title buffers expect `AllocationBase` to match the reserved region boundary rather than the committed page offset.',
    state: 'open',
    html_url: 'https://github.com/xcl-dev/xcl/issues/1975',
    user: { login: 'Kalmarv', avatar_url: '', html_url: 'https://github.com/Kalmarv' },
    labels: [{ id: 5, name: 'kernel', color: 'fbca04' }],
    assignees: [],
    comments: 2,
    created_at: '2026-11-24T09:15:00Z',
    updated_at: '2026-11-24T21:10:00Z',
    closed_at: null,
  },
  {
    id: 4001960,
    number: 1960,
    kind: 'issue',
    title: 'Scripts: smoke test runner needs GPU memory & pipeline counter summary',
    body: 'Track draw call count, committed video memory, and PSO cache hits across smoke runs so regressions are visible in PR checks.',
    state: 'open',
    html_url: 'https://github.com/xcl-dev/xcl/issues/1960',
    user: { login: 'brandostrong', avatar_url: '', html_url: 'https://github.com/brandostrong' },
    labels: [{ id: 6, name: 'tooling', color: '006b75' }],
    assignees: [],
    comments: 3,
    created_at: '2026-11-23T22:05:00Z',
    updated_at: '2026-11-24T19:30:00Z',
    closed_at: null,
  },
  {
    id: 4001942,
    number: 1942,
    kind: 'issue',
    title: 'Docs: add CODEOWNERS for recompiler, DXIL, and kernel subsystems',
    body: 'With PR volume accelerating, adding `CODEOWNERS` ensures subsystem maintainers get automatic review requests.',
    state: 'open',
    html_url: 'https://github.com/xcl-dev/xcl/issues/1942',
    user: { login: 'theaustinayers', avatar_url: '', html_url: 'https://github.com/theaustinayers' },
    labels: [{ id: 7, name: 'documentation', color: '0075ca' }],
    assignees: [],
    comments: 1,
    created_at: '2026-11-23T16:40:00Z',
    updated_at: '2026-11-24T18:00:00Z',
    closed_at: null,
  },
];

export const XCL_FALLBACK_PULLS: GitHubItem[] = [
  {
    id: 5002014,
    number: 2014,
    kind: 'pr',
    title: 'Recompiler: fix per-lane whole-wave predicate and SRC_SCC checks',
    body: 'Fixes per-lane mask propagation when whole-wave predicates combine with `SRC_SCC` scalar condition checks.',
    state: 'open',
    draft: false,
    merged_at: null,
    head_ref: 'fix/per-lane-whole-wave-predicates',
    base_ref: 'main',
    html_url: 'https://github.com/xcl-dev/xcl/pull/2014',
    user: { login: 'AlpPesen', avatar_url: '', html_url: 'https://github.com/AlpPesen' },
    labels: [{ id: 2, name: 'recompiler', color: '1d76db' }],
    assignees: [],
    comments: 0,
    created_at: '2026-11-25T00:42:57Z',
    updated_at: '2026-11-25T00:42:57Z',
    closed_at: null,
  },
  {
    id: 5002013,
    number: 2013,
    kind: 'pr',
    title: 'Shaders: bypass caches when DXIL optimization is disabled',
    body: 'Ensures unoptimized debug shader dumps bypass disk and memory caches cleanly.',
    state: 'open',
    draft: true,
    merged_at: null,
    head_ref: 'codex/dxil-no-opt-cache',
    base_ref: 'main',
    html_url: 'https://github.com/xcl-dev/xcl/pull/2013',
    user: { login: 'luckyteagirl', avatar_url: '', html_url: 'https://github.com/luckyteagirl' },
    labels: [{ id: 3, name: 'shaders', color: '0e8a16' }],
    assignees: [],
    comments: 1,
    created_at: '2026-11-25T00:42:10Z',
    updated_at: '2026-11-25T00:42:10Z',
    closed_at: null,
  },
  {
    id: 5002012,
    number: 2012,
    kind: 'pr',
    title: 'Docs: CODEOWNERS names the reviewers a merge into main needs',
    body: 'Defines subsystem code owners for faster automated review assignment.',
    state: 'open',
    draft: false,
    merged_at: null,
    head_ref: 'docs/codeowners',
    base_ref: 'main',
    html_url: 'https://github.com/xcl-dev/xcl/pull/2012',
    user: { login: 'theaustinayers', avatar_url: '', html_url: 'https://github.com/theaustinayers' },
    labels: [{ id: 7, name: 'documentation', color: '0075ca' }],
    assignees: [],
    comments: 2,
    created_at: '2026-11-25T00:41:14Z',
    updated_at: '2026-11-25T00:41:14Z',
    closed_at: null,
  },
  {
    id: 5002011,
    number: 2011,
    kind: 'pr',
    title: 'Recompiler: use IEEE-754 equality for float compare-and-swap atomics',
    body: 'Resolves #1489 by preserving bit-exact float equality semantics in `fcmpswap`.',
    state: 'open',
    draft: false,
    merged_at: null,
    head_ref: 'fix-1489-fcmpswap-ieee-eq',
    base_ref: 'main',
    html_url: 'https://github.com/xcl-dev/xcl/pull/2011',
    user: { login: 'Decodere', avatar_url: '', html_url: 'https://github.com/Decodere' },
    labels: [
      { id: 1, name: 'bug', color: 'd73a4a' },
      { id: 2, name: 'recompiler', color: '1d76db' },
    ],
    assignees: [],
    comments: 3,
    created_at: '2026-11-25T00:41:12Z',
    updated_at: '2026-11-25T00:41:12Z',
    closed_at: null,
  },
  {
    id: 5002010,
    number: 2010,
    kind: 'pr',
    title: 'Scripts: smoke block v2 adds draws, video memory and pipeline counters',
    body: 'Implements draw call, VRAM, and pipeline telemetry counters in the smoke harness (#1960).',
    state: 'open',
    draft: false,
    merged_at: null,
    head_ref: 'smoke-block-v2',
    base_ref: 'main',
    html_url: 'https://github.com/xcl-dev/xcl/pull/2010',
    user: { login: 'brandostrong', avatar_url: '', html_url: 'https://github.com/brandostrong' },
    labels: [{ id: 6, name: 'tooling', color: '006b75' }],
    assignees: [],
    comments: 2,
    created_at: '2026-11-25T00:40:53Z',
    updated_at: '2026-11-25T00:40:53Z',
    closed_at: null,
  },
];

export const XCL_FALLBACK_COMMITS: GitHubCommit[] = [
  {
    sha: '6567c5f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7',
    shortSha: '6567c5f',
    messageHeadline: 'Merge pull request #1910 from Kalmarv/whole-wave-branches',
    messageBody: 'Recompiler: handle whole-wave branch divergence cleanly',
    authorName: 'Ray',
    authorLogin: 'Ray',
    authorAvatarUrl: null,
    date: '2026-11-25T00:26:37Z',
    html_url: 'https://github.com/xcl-dev/xcl/commit/6567c5f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7',
    linkedPrNumber: 1910,
  },
  {
    sha: '8c4a46a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7',
    shortSha: '8c4a46a',
    messageHeadline: 'Merge pull request #1772 from AlpPesen/fix/stream-pipeline-depth-default',
    messageBody: 'Fix stream pipeline depth default state',
    authorName: 'Ray',
    authorLogin: 'Ray',
    authorAvatarUrl: null,
    date: '2026-11-25T00:21:18Z',
    html_url: 'https://github.com/xcl-dev/xcl/commit/8c4a46a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7',
    linkedPrNumber: 1772,
  },
  {
    sha: '2c95f011a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7',
    shortSha: '2c95f01',
    messageHeadline: 'Merge pull request #1937 from ZekkJJ/upstream/two-pass-triangle',
    messageBody: 'Support two-pass triangle topology lowering',
    authorName: 'Ray',
    authorLogin: 'Ray',
    authorAvatarUrl: null,
    date: '2026-11-25T00:20:58Z',
    html_url: 'https://github.com/xcl-dev/xcl/commit/2c95f011a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7',
    linkedPrNumber: 1937,
  },
  {
    sha: '80f48191a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7',
    shortSha: '80f4819',
    messageHeadline: 'Merge pull request #1836 from raylitebull/perf/pso-direct-dxil-speedup',
    messageBody: 'Speed up direct DXIL PSO creation path',
    authorName: 'Ray',
    authorLogin: 'Ray',
    authorAvatarUrl: null,
    date: '2026-11-25T00:04:11Z',
    html_url: 'https://github.com/xcl-dev/xcl/commit/80f48191a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7',
    linkedPrNumber: 1836,
  },
  {
    sha: '8e87fc71a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7',
    shortSha: '8e87fc7',
    messageHeadline: 'Merge pull request #1898 from emirduran707/dxil-round-denorm-inf-nan',
    messageBody: 'Preserve denorm/inf/nan behavior on DXIL round instructions',
    authorName: 'Ray',
    authorLogin: 'Ray',
    authorAvatarUrl: null,
    date: '2026-11-24T23:55:08Z',
    html_url: 'https://github.com/xcl-dev/xcl/commit/8e87fc71a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7',
    linkedPrNumber: 1898,
  },
];

export function getXclFallbackBundle(notice?: string): RepoBundle {
  return {
    repo: XCL_FALLBACK_REPO,
    issues: XCL_FALLBACK_ISSUES,
    issuesTotalCount: XCL_FALLBACK_ISSUES.length,
    pulls: XCL_FALLBACK_PULLS,
    commits: XCL_FALLBACK_COMMITS,
    source: 'snapshot',
    fetchedAt: new Date().toISOString(),
    notice,
  };
}

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

export const XCL_1607_FALLBACK_BODY = `<!-- xcl-digest:begin -->
### Checked by two contributors (same finding)
| unit | issue | PRs | finding | agreeing checks | checked by |
|---|---|---|---|---|---|
| D20 | #105 | #297 #313 #1371 | different approaches: packed xclin members vs clamp | 2 | [alice](https://github.com/alice) |

### Checks disagree (a third look would help)
| unit | issue | PRs | finding | checks |
|---|---|---|---|---|

### Waiting for a second check
| unit | issue | kind | PRs | note | checks on current heads | outdated checks |
|---|---|---|---|---|---|---|
| D51 | #175 | dup | #867 #1395 | vkd3d-proton's vtable assertions on adopted objects under Wine | [humsterbumster](https://github.com/humsterbumster) (nv-linux) | 2 |
| D72 | #1909 | dup | #1945 #1967 | Both clean allocator-pool state on final queue Release. | [humsterbumster](https://github.com/humsterbumster) (nv-linux) | 0 |
| B15 | – | batch on 1d027e5 | #640 #1364 #1831 | – | [CalvyPZ](https://github.com/CalvyPZ) (nv-win) | 0 |

<!-- xcl-ids:{"dup|297,313,1371|":"D20","dup|867,1395|":"D51","dup|1945,1967|":"D72","batch|640,1364,1831|1d027e5":"B15"} -->
<!-- xcl-digest:end -->
<!-- xcl-reports:begin -->
### Independent test reports per PR
_Counted 2026-11-25 01:24 UTC by a script over all 1112 open PRs._
Ready PRs (not draft, \`PR description\` check green): **538**. With 2+ independent reports on their current head: **155**; with 1: **196**; with none on the current head: **187** (82 of them have reports on an earlier head).

### Most likely ready to merge
| PR | linked issue | reports on current head | by | title |
|---|---|---|---|---|
| #587 | yes | 3 | CryingLaughing, MesshM, zhouzhidong3-hash | Recompiler: v_frexp_mant_f32 / _f16 keep the input's sign (#580) |
| #632 | yes | 3 | JoaoPauluu, MesshM, eizispark | Memory: a whole coherent window in a title's first run (#284) |
| #1595 | yes | 3 | MiharaxD, eizispark, futurepro62 | D3D12.X: warn when the title's memory budget is larger than the GPU's |

### One more test report would make these ready
_Same rules, with one independent report on the current head so far._
#543 #1050 #1691 #1846 #1925 #346 #464 #609 #665 #939

<details><summary>Ready PRs with no report on their current head (oldest 40)</summary>#264 #271 #272 #274 #275 #276 #277 #279 #282 #286</details>
<!-- xcl-reports:end -->`;
