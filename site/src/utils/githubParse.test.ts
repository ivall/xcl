import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildNewIssueUrl,
  extractLabelCounts,
  extractLinkedPrNumber,
  extractSubsystemCounts,
  filterAndSortItems,
  formatCompactNumber,
  formatRelativeTime,
  parseRepoInput,
  parseSubsystemPrefix,
} from './githubParse';
import type { GitHubItem, TriageRecord } from '../types/github';

function makeItem(overrides: Partial<GitHubItem> & { number: number; title: string }): GitHubItem {
  return {
    id: overrides.number,
    number: overrides.number,
    kind: overrides.kind ?? 'issue',
    title: overrides.title,
    body: overrides.body ?? '',
    state: overrides.state ?? 'open',
    html_url: `https://github.com/xcl-dev/xcl/issues/${overrides.number}`,
    user: overrides.user ?? {
      login: 'devuser',
      avatar_url: '',
      html_url: 'https://github.com/devuser',
    },
    labels: overrides.labels ?? [],
    assignees: overrides.assignees ?? [],
    comments: overrides.comments ?? 0,
    created_at: overrides.created_at ?? '2026-11-24T12:00:00Z',
    updated_at: overrides.updated_at ?? '2026-11-24T12:00:00Z',
    closed_at: overrides.closed_at ?? null,
    draft: overrides.draft,
    head_ref: overrides.head_ref,
    base_ref: overrides.base_ref,
  };
}

test('parseRepoInput accepts owner/repo and full GitHub URLs', () => {
  assert.deepEqual(parseRepoInput('xcl-dev/xcl'), {
    ok: true,
    owner: 'xcl-dev',
    repo: 'xcl',
    slug: 'xcl-dev/xcl',
  });
  assert.deepEqual(parseRepoInput('https://github.com/xcl-dev/xcl/issues'), {
    ok: true,
    owner: 'xcl-dev',
    repo: 'xcl',
    slug: 'xcl-dev/xcl',
  });
  assert.deepEqual(parseRepoInput('git@github.com:xcl-dev/xcl.git'), {
    ok: true,
    owner: 'xcl-dev',
    repo: 'xcl',
    slug: 'xcl-dev/xcl',
  });
});

test('parseRepoInput rejects empty and malformed inputs without fallback', () => {
  const empty = parseRepoInput('   ');
  assert.equal(empty.ok, false);
  const single = parseRepoInput('justowner');
  assert.equal(single.ok, false);
  const badChars = parseRepoInput('bad owner/repo!');
  assert.equal(badChars.ok, false);
});

test('parseSubsystemPrefix extracts colon and bracket prefixes', () => {
  assert.equal(
    parseSubsystemPrefix('Recompiler: fix per-lane whole-wave predicate checks'),
    'Recompiler'
  );
  assert.equal(parseSubsystemPrefix('[Shaders] bypass DXIL cache'), 'Shaders');
  assert.equal(parseSubsystemPrefix('Fix memory leak on exit'), null);
});

test('extractLinkedPrNumber finds merge and squash PR references', () => {
  assert.equal(
    extractLinkedPrNumber('Merge pull request #1910 from Kalmarv/whole-wave-branches\n\nBody'),
    1910
  );
  assert.equal(extractLinkedPrNumber('Shaders: fix resource binding (#1844)'), 1844);
  assert.equal(extractLinkedPrNumber('Docs: update README'), null);
});

test('formatRelativeTime and formatCompactNumber format clearly', () => {
  const now = Date.parse('2026-11-25T01:00:00Z');
  assert.equal(formatRelativeTime('2026-11-25T00:59:45Z', now), 'just now');
  assert.equal(formatRelativeTime('2026-11-25T00:45:00Z', now), '15m ago');
  assert.equal(formatRelativeTime('2026-11-24T22:00:00Z', now), '3h ago');
  assert.equal(formatRelativeTime('2026-11-22T01:00:00Z', now), '3d ago');
  assert.equal(formatCompactNumber(7833), '7,833');
  assert.equal(formatCompactNumber(14200), '14.2k');
});

test('buildNewIssueUrl constructs pre-filled GitHub issue URL', () => {
  const url = buildNewIssueUrl('xcl-dev', 'xcl', 'Recompiler: crash on wave op', 'Steps to repro', [
    'bug',
    'recompiler',
  ]);
  const parsed = new URL(url);
  assert.equal(parsed.origin + parsed.pathname, 'https://github.com/xcl-dev/xcl/issues/new');
  assert.equal(parsed.searchParams.get('title'), 'Recompiler: crash on wave op');
  assert.equal(parsed.searchParams.get('body'), 'Steps to repro');
  assert.equal(parsed.searchParams.get('labels'), 'bug,recompiler');
});

test('filterAndSortItems filters by search, subsystem, triage, and sorts accurately', () => {
  const items: GitHubItem[] = [
    makeItem({
      number: 2014,
      kind: 'pr',
      title: 'Recompiler: fix per-lane checks',
      state: 'open',
      comments: 1,
      updated_at: '2026-11-25T00:42:57Z',
      head_ref: 'fix/per-lane',
      base_ref: 'main',
    }),
    makeItem({
      number: 2013,
      kind: 'pr',
      title: 'Shaders: bypass cache when disabled',
      state: 'open',
      draft: true,
      comments: 5,
      updated_at: '2026-11-25T00:40:00Z',
    }),
    makeItem({
      number: 1990,
      kind: 'pr',
      title: 'Docs: CODEOWNERS update',
      state: 'closed',
      comments: 0,
      updated_at: '2026-11-24T20:00:00Z',
    }),
  ];

  const triageMap: Record<string, TriageRecord> = {
    'xcl-dev/xcl#2014': {
      key: 'xcl-dev/xcl#2014',
      repoSlug: 'xcl-dev/xcl',
      number: 2014,
      kind: 'pr',
      title: 'Recompiler: fix per-lane checks',
      html_url: 'https://github.com/xcl-dev/xcl/pull/2014',
      status: 'in_progress',
      pinned: true,
      note: 'Reviewing wave predicate logic',
      updatedAt: '2026-11-25T00:43:00Z',
    },
  };

  const recompiler = filterAndSortItems(items, {
    search: 'per-lane',
    stateFilter: 'open',
    subsystem: 'all',
    label: 'all',
    triageFilter: 'all',
    sortMode: 'updated',
    triageMap,
    repoSlug: 'xcl-dev/xcl',
  });
  assert.deepEqual(
    recompiler.map((i) => i.number),
    [2014]
  );

  const drafts = filterAndSortItems(items, {
    search: '',
    stateFilter: 'draft',
    subsystem: 'all',
    label: 'all',
    triageFilter: 'all',
    sortMode: 'updated',
    triageMap,
    repoSlug: 'xcl-dev/xcl',
  });
  assert.deepEqual(
    drafts.map((i) => i.number),
    [2013]
  );

  const untriaged = filterAndSortItems(items, {
    search: '',
    stateFilter: 'all',
    subsystem: 'all',
    label: 'all',
    triageFilter: 'untriaged',
    sortMode: 'comments',
    triageMap,
    repoSlug: 'xcl-dev/xcl',
  });
  assert.deepEqual(
    untriaged.map((i) => i.number),
    [2013, 1990]
  );
});

test('extractSubsystemCounts and extractLabelCounts aggregate frequency order', () => {
  const items: GitHubItem[] = [
    makeItem({
      number: 1,
      title: 'Recompiler: a',
      labels: [{ id: 1, name: 'bug', color: 'd73a4a' }],
    }),
    makeItem({
      number: 2,
      title: 'Recompiler: b',
      labels: [
        { id: 1, name: 'bug', color: 'd73a4a' },
        { id: 2, name: 'dxil', color: '0075ca' },
      ],
    }),
    makeItem({
      number: 3,
      title: 'Shaders: c',
      labels: [{ id: 2, name: 'dxil', color: '0075ca' }],
    }),
  ];

  assert.deepEqual(extractSubsystemCounts(items), [
    { name: 'Recompiler', count: 2 },
    { name: 'Shaders', count: 1 },
  ]);
  assert.deepEqual(extractLabelCounts(items).map((l) => l.name), ['bug', 'dxil']);
});
