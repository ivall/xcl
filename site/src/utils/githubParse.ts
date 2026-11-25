import type {
  GitHubItem,
  ItemSortMode,
  ItemStateFilter,
  TriageRecord,
  TriageStatus,
} from '../types/github';

export type ParseRepoResult =
  | { ok: true; owner: string; repo: string; slug: string }
  | { ok: false; error: string };

const SEGMENT_RE = /^[A-Za-z0-9_.-]+$/;

export function parseRepoInput(raw: string): ParseRepoResult {
  const trimmed = raw.trim();
  if (!trimmed) {
    return {
      ok: false,
      error: 'Enter a GitHub repository like xcl-dev/xcl or paste a GitHub URL.',
    };
  }

  let pathPart = trimmed;

  // Handle SSH clone URLs like git@github.com:xcl-dev/xcl.git
  const sshMatch = trimmed.match(/^git@github\.com:([^/]+)\/([^/\s]+?)(?:\.git)?$/i);
  if (sshMatch) {
    const [, owner, repo] = sshMatch;
    if (SEGMENT_RE.test(owner) && SEGMENT_RE.test(repo)) {
      return { ok: true, owner, repo, slug: `${owner}/${repo}` };
    }
  }

  // Handle full or protocol-less GitHub URLs
  if (/^(https?:\/\/)?(www\.)?github\.com\//i.test(trimmed)) {
    const normalized = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
    try {
      const url = new URL(normalized);
      pathPart = url.pathname;
    } catch {
      return {
        ok: false,
        error: 'That GitHub URL could not be parsed. Try owner/repo (e.g. xcl-dev/xcl).',
      };
    }
  }

  const segments = pathPart
    .replace(/^\/+|\/+$/g, '')
    .replace(/\.git$/i, '')
    .split('/')
    .filter(Boolean);

  if (segments.length < 2) {
    return {
      ok: false,
      error: 'Use the format owner/repo (for example xcl-dev/xcl).',
    };
  }

  const [owner, repo] = segments;
  if (!SEGMENT_RE.test(owner) || !SEGMENT_RE.test(repo)) {
    return {
      ok: false,
      error: 'Repository owner and name may only contain letters, numbers, ., _, and -.',
    };
  }

  return {
    ok: true,
    owner,
    repo,
    slug: `${owner}/${repo}`,
  };
}

export function parseSubsystemPrefix(title: string): string | null {
  const trimmed = title.trim();
  if (!trimmed) return null;

  const colonMatch = trimmed.match(/^([A-Za-z][A-Za-z0-9_-]{1,22}):\s+/);
  if (colonMatch) {
    return colonMatch[1];
  }

  const bracketMatch = trimmed.match(/^\[([A-Za-z][A-Za-z0-9 _/-]{1,22})\]\s+/);
  if (bracketMatch) {
    return bracketMatch[1].trim();
  }

  return null;
}

export function extractLinkedPrNumber(commitMessage: string): number | null {
  const headline = commitMessage.split('\n')[0] ?? '';
  const mergeMatch = headline.match(/Merge pull request #(\d+)/i);
  if (mergeMatch) {
    return Number(mergeMatch[1]);
  }
  const squashMatch = headline.match(/\(#(\d+)\)\s*$/);
  if (squashMatch) {
    return Number(squashMatch[1]);
  }
  return null;
}

export function formatRelativeTime(isoDate: string, nowMs: number = Date.now()): string {
  const targetMs = Date.parse(isoDate);
  if (Number.isNaN(targetMs)) return 'unknown';

  const diffSeconds = Math.max(0, Math.floor((nowMs - targetMs) / 1000));
  if (diffSeconds < 60) return 'just now';

  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes}m ago`;

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 30) return `${diffDays}d ago`;

  const date = new Date(targetMs);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export function formatCompactNumber(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`;
  }
  if (value >= 10_000) {
    return `${(value / 1_000).toFixed(1).replace(/\.0$/, '')}k`;
  }
  return value.toLocaleString('en-US');
}

export function buildNewIssueUrl(
  owner: string,
  repo: string,
  title: string,
  body: string,
  labels: string[] = []
): string {
  const url = new URL(`https://github.com/${owner}/${repo}/issues/new`);
  if (title.trim()) {
    url.searchParams.set('title', title.trim());
  }
  if (body.trim()) {
    url.searchParams.set('body', body.trim());
  }
  if (labels.length > 0) {
    url.searchParams.set('labels', labels.join(','));
  }
  return url.toString();
}

export interface FilterItemsOptions {
  search: string;
  stateFilter: ItemStateFilter;
  subsystem: string;
  label: string;
  triageFilter: 'all' | 'untriaged' | 'pinned' | TriageStatus;
  sortMode: ItemSortMode;
  triageMap: Record<string, TriageRecord>;
  repoSlug: string;
}

export function makeTriageKey(repoSlug: string, itemNumber: number): string {
  return `${repoSlug.toLowerCase()}#${itemNumber}`;
}

export function filterAndSortItems(
  items: GitHubItem[],
  options: FilterItemsOptions
): GitHubItem[] {
  const q = options.search.trim().toLowerCase();
  const normalizedQueryNumber = q.startsWith('#') ? q.slice(1) : q;

  const filtered = items.filter((item) => {
    if (options.stateFilter === 'open' && item.state !== 'open') return false;
    if (options.stateFilter === 'closed' && item.state !== 'closed') return false;
    if (options.stateFilter === 'draft' && !(item.kind === 'pr' && item.draft)) {
      return false;
    }

    if (options.subsystem !== 'all') {
      const prefix = parseSubsystemPrefix(item.title);
      if (!prefix || prefix.toLowerCase() !== options.subsystem.toLowerCase()) {
        return false;
      }
    }

    if (options.label !== 'all') {
      const hasLabel = item.labels.some(
        (l) => l.name.toLowerCase() === options.label.toLowerCase()
      );
      if (!hasLabel) return false;
    }

    const triage = options.triageMap[makeTriageKey(options.repoSlug, item.number)];
    if (options.triageFilter === 'untriaged') {
      if (triage?.status || triage?.pinned) return false;
    } else if (options.triageFilter === 'pinned') {
      if (!triage?.pinned) return false;
    } else if (options.triageFilter !== 'all') {
      if (triage?.status !== options.triageFilter) return false;
    }

    if (q) {
      const matchesNumber =
        normalizedQueryNumber.length > 0 &&
        String(item.number).includes(normalizedQueryNumber);
      const matchesTitle = item.title.toLowerCase().includes(q);
      const matchesAuthor = item.user.login.toLowerCase().includes(q);
      const matchesLabel = item.labels.some((l) => l.name.toLowerCase().includes(q));
      const matchesBranch =
        (item.head_ref && item.head_ref.toLowerCase().includes(q)) ||
        (item.base_ref && item.base_ref.toLowerCase().includes(q));

      if (!matchesNumber && !matchesTitle && !matchesAuthor && !matchesLabel && !matchesBranch) {
        return false;
      }
    }

    return true;
  });

  return [...filtered].sort((a, b) => {
    if (options.sortMode === 'created') {
      return Date.parse(b.created_at) - Date.parse(a.created_at);
    }
    if (options.sortMode === 'comments') {
      if (b.comments !== a.comments) return b.comments - a.comments;
      return Date.parse(b.updated_at) - Date.parse(a.updated_at);
    }
    if (options.sortMode === 'needs_triage') {
      const aTriaged = Boolean(
        options.triageMap[makeTriageKey(options.repoSlug, a.number)]?.status
      );
      const bTriaged = Boolean(
        options.triageMap[makeTriageKey(options.repoSlug, b.number)]?.status
      );
      if (aTriaged !== bTriaged) return aTriaged ? 1 : -1;
      if (a.labels.length !== b.labels.length) return a.labels.length - b.labels.length;
      return Date.parse(b.updated_at) - Date.parse(a.updated_at);
    }
    return Date.parse(b.updated_at) - Date.parse(a.updated_at);
  });
}

export function extractSubsystemCounts(items: GitHubItem[]): Array<{ name: string; count: number }> {
  const counts = new Map<string, number>();
  for (const item of items) {
    const prefix = parseSubsystemPrefix(item.title);
    if (prefix) {
      counts.set(prefix, (counts.get(prefix) ?? 0) + 1);
    }
  }
  return Array.from(counts.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

export function extractLabelCounts(items: GitHubItem[]): Array<{ name: string; color: string; count: number }> {
  const map = new Map<string, { name: string; color: string; count: number }>();
  for (const item of items) {
    for (const label of item.labels) {
      const existing = map.get(label.name);
      if (existing) {
        existing.count += 1;
      } else {
        map.set(label.name, { name: label.name, color: label.color, count: 1 });
      }
    }
  }
  return Array.from(map.values()).sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}
