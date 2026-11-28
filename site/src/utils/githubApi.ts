import type {
  GitHubComment,
  GitHubCommit,
  GitHubItem,
  GitHubUser,
  RepoBundle,
  RepoInfo,
} from '../types/github';
import { extractLinkedPrNumber } from './githubParse';

const CACHE_PREFIX = 'repopulse:cache:v1:';
const CACHE_TTL_MS = 60 * 60 * 1000; // Complete pagination can consume much of the anonymous quota.

const HEADERS: HeadersInit = {
  Accept: 'application/vnd.github+json',
};

function mapUser(raw: Record<string, unknown> | null | undefined): GitHubUser {
  return {
    login: String(raw?.login ?? 'unknown'),
    avatar_url: String(raw?.avatar_url ?? ''),
    html_url: String(raw?.html_url ?? 'https://github.com'),
  };
}

function mapIssueOrPr(
  raw: Record<string, unknown>,
  forcedKind?: 'issue' | 'pr'
): GitHubItem {
  const isPr = forcedKind === 'pr' || Boolean(raw.pull_request) || Boolean(raw.head);
  const head = raw.head as Record<string, unknown> | undefined;
  const base = raw.base as Record<string, unknown> | undefined;

  return {
    id: Number(raw.id ?? raw.number),
    number: Number(raw.number),
    kind: isPr ? 'pr' : 'issue',
    title: String(raw.title ?? '(untitled)'),
    body: String(raw.body ?? ''),
    state: raw.state === 'closed' ? 'closed' : 'open',
    html_url: String(raw.html_url ?? ''),
    user: mapUser(raw.user as Record<string, unknown> | undefined),
    labels: Array.isArray(raw.labels)
      ? raw.labels.map((l: Record<string, unknown>) => ({
          id: Number(l.id ?? 0),
          name: String(l.name ?? ''),
          color: String(l.color ?? '64748b'),
          description: l.description ? String(l.description) : null,
        }))
      : [],
    assignees: Array.isArray(raw.assignees)
      ? raw.assignees.map((a) => mapUser(a as Record<string, unknown>))
      : [],
    comments: Number(raw.comments ?? 0),
    created_at: String(raw.created_at ?? ''),
    updated_at: String(raw.updated_at ?? ''),
    closed_at: raw.closed_at ? String(raw.closed_at) : null,
    draft: typeof raw.draft === 'boolean' ? raw.draft : undefined,
    merged_at: raw.merged_at ? String(raw.merged_at) : null,
    head_ref: head?.ref ? String(head.ref) : undefined,
    head_sha: head?.sha ? String(head.sha) : undefined,
    base_ref: base?.ref ? String(base.ref) : undefined,
    additions: typeof raw.additions === 'number' ? raw.additions : undefined,
    deletions: typeof raw.deletions === 'number' ? raw.deletions : undefined,
    changed_files: typeof raw.changed_files === 'number' ? raw.changed_files : undefined,
  };
}

function mapCommit(raw: Record<string, unknown>): GitHubCommit {
  const commitObj = (raw.commit as Record<string, unknown>) ?? {};
  const authorMeta = (commitObj.author as Record<string, unknown>) ?? {};
  const ghAuthor = raw.author as Record<string, unknown> | null | undefined;
  const fullMessage = String(commitObj.message ?? '');
  const [headline, ...rest] = fullMessage.split('\n');
  const sha = String(raw.sha ?? '');

  return {
    sha,
    shortSha: sha.slice(0, 7),
    messageHeadline: headline || '(no commit message)',
    messageBody: rest.join('\n').trim(),
    authorName: String(authorMeta.name ?? ghAuthor?.login ?? 'unknown'),
    authorLogin: ghAuthor?.login ? String(ghAuthor.login) : null,
    authorAvatarUrl: ghAuthor?.avatar_url ? String(ghAuthor.avatar_url) : null,
    date: String(authorMeta.date ?? ''),
    html_url: String(raw.html_url ?? ''),
    linkedPrNumber: extractLinkedPrNumber(fullMessage),
  };
}

function readCachedBundle(repoSlug: string): RepoBundle | null {
  try {
    const raw = localStorage.getItem(`${CACHE_PREFIX}${repoSlug.toLowerCase()}`);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as RepoBundle;
    const age = Date.now() - Date.parse(parsed.fetchedAt);
    if (Number.isNaN(age) || age > CACHE_TTL_MS) {
      return null;
    }
    return { ...parsed, source: 'cache' };
  } catch {
    return null;
  }
}

function writeCachedBundle(repoSlug: string, bundle: RepoBundle): void {
  try {
    localStorage.setItem(`${CACHE_PREFIX}${repoSlug.toLowerCase()}`, JSON.stringify(bundle));
  } catch {
    // Ignore storage quota errors
  }
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { headers: HEADERS });
  if (!response.ok) {
    const remaining = response.headers.get('x-ratelimit-remaining');
    if (response.status === 403 && remaining === '0') {
      throw new Error('RATE_LIMIT');
    }
    if (response.status === 404) {
      throw new Error('NOT_FOUND');
    }
    throw new Error(`HTTP_${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchAllPages<T>(url: string): Promise<T[]> {
  const items: T[] = [];
  const separator = url.includes('?') ? '&' : '?';

  for (let page = 1; ; page += 1) {
    const batch = await fetchJson<T[]>(`${url}${separator}per_page=100&page=${page}`);
    items.push(...batch);
    if (batch.length < 100) return items;
  }
}

export async function fetchRepoBundle(
  owner: string,
  repo: string,
  options: { forceRefresh?: boolean } = {}
): Promise<RepoBundle> {
  const slug = `${owner}/${repo}`;

  if (!options.forceRefresh) {
    const cached = readCachedBundle(slug);
    if (cached) {
      return cached;
    }
  }

  try {
    const [repoRaw, issuesAndPullsRaw, pullsRaw, commitsRaw] = await Promise.all([
      fetchJson<Record<string, unknown>>(`https://api.github.com/repos/${owner}/${repo}`),
      fetchAllPages<Record<string, unknown>>(
        `https://api.github.com/repos/${owner}/${repo}/issues?state=all&sort=updated&direction=desc`
      ),
      fetchAllPages<Record<string, unknown>>(
        `https://api.github.com/repos/${owner}/${repo}/pulls?state=all&sort=updated&direction=desc`
      ),
      fetchJson<Array<Record<string, unknown>>>(
        `https://api.github.com/repos/${owner}/${repo}/commits?per_page=40`
      ),
    ]);

    const ownerObj = (repoRaw.owner as Record<string, unknown>) ?? {};
    const repoInfo: RepoInfo = {
      full_name: String(repoRaw.full_name ?? slug),
      owner: String(ownerObj.login ?? owner),
      name: String(repoRaw.name ?? repo),
      description: repoRaw.description ? String(repoRaw.description) : null,
      html_url: String(repoRaw.html_url ?? `https://github.com/${slug}`),
      stargazers_count: Number(repoRaw.stargazers_count ?? 0),
      forks_count: Number(repoRaw.forks_count ?? 0),
      open_issues_count: Number(repoRaw.open_issues_count ?? 0),
      default_branch: String(repoRaw.default_branch ?? 'main'),
      language: repoRaw.language ? String(repoRaw.language) : null,
      pushed_at: String(repoRaw.pushed_at ?? new Date().toISOString()),
    };

    const issues = issuesAndPullsRaw
      .filter((item) => !item.pull_request)
      .map((item) => mapIssueOrPr(item, 'issue'));
    const pulls = pullsRaw.map((pr) => mapIssueOrPr(pr, 'pr'));
    const commits = commitsRaw.map(mapCommit);

    const bundle: RepoBundle = {
      repo: repoInfo,
      issues,
      issuesTotalCount: issues.length,
      pulls,
      commits,
      source: 'live',
      fetchedAt: new Date().toISOString(),
    };

    writeCachedBundle(slug, bundle);
    return bundle;
  } catch (err) {
    const code = err instanceof Error ? err.message : 'UNKNOWN';
    if (code === 'NOT_FOUND') {
      throw new Error(`Repository "${slug}" was not found or is private.`);
    }

    try {
      const raw = localStorage.getItem(`${CACHE_PREFIX}${slug.toLowerCase()}`);
      if (raw) {
        const parsed = JSON.parse(raw) as RepoBundle;
        return {
          ...parsed,
          source: 'cache',
          notice:
            code === 'RATE_LIMIT'
              ? 'GitHub unauthenticated API rate limit reached — showing previously cached GitHub data.'
              : 'GitHub API unreachable — showing previously cached GitHub data.',
        };
      }
    } catch {
      // Ignore
    }

    throw new Error(
      code === 'RATE_LIMIT'
        ? 'GitHub unauthenticated API rate limit reached (60 req/hr), and no cached data is available. Wait for the limit to reset and retry.'
        : `Failed to load ${slug} from GitHub API.`
    );
  }
}

export async function fetchItemComments(
  owner: string,
  repo: string,
  number: number
): Promise<GitHubComment[]> {
  const data = await fetchJson<Array<Record<string, unknown>>>(
    `https://api.github.com/repos/${owner}/${repo}/issues/${number}/comments?per_page=30`
  );
  return data.map((c) => ({
    id: Number(c.id),
    user: mapUser(c.user as Record<string, unknown> | undefined),
    body: String(c.body ?? ''),
    created_at: String(c.created_at ?? ''),
    updated_at: String(c.updated_at ?? ''),
    html_url: String(c.html_url ?? ''),
  }));
}

export async function fetchPullDetail(
  owner: string,
  repo: string,
  number: number
): Promise<Partial<GitHubItem>> {
  const data = await fetchJson<Record<string, unknown>>(
    `https://api.github.com/repos/${owner}/${repo}/pulls/${number}`
  );
  const head = data.head as Record<string, unknown> | undefined;
  return {
    additions: typeof data.additions === 'number' ? data.additions : undefined,
    deletions: typeof data.deletions === 'number' ? data.deletions : undefined,
    changed_files: typeof data.changed_files === 'number' ? data.changed_files : undefined,
    merged_at: data.merged_at ? String(data.merged_at) : null,
    head_sha: head?.sha ? String(head.sha) : undefined,
  };
}

export async function fetchPullByNumber(
  owner: string,
  repo: string,
  number: number
): Promise<GitHubItem> {
  const cacheKey = `${CACHE_PREFIX}pr:${owner.toLowerCase()}/${repo.toLowerCase()}#${number}`;
  try {
    const cachedRaw = localStorage.getItem(cacheKey);
    if (cachedRaw) {
      const parsed = JSON.parse(cachedRaw) as { fetchedAt: number; item: GitHubItem };
      if (Date.now() - parsed.fetchedAt < CACHE_TTL_MS) {
        return parsed.item;
      }
    }
  } catch {
    // Ignore
  }

  const data = await fetchJson<Record<string, unknown>>(
    `https://api.github.com/repos/${owner}/${repo}/pulls/${number}`
  );
  const item = mapIssueOrPr(data, 'pr');
  try {
    localStorage.setItem(cacheKey, JSON.stringify({ fetchedAt: Date.now(), item }));
  } catch {
    // Ignore
  }
  return item;
}

export async function fetchXcl1607Body(forceRefresh = false): Promise<{
  body: string;
  source: 'live' | 'cache';
}> {
  const cacheKey = `${CACHE_PREFIX}xcl1607`;
  if (!forceRefresh) {
    try {
      const cachedRaw = localStorage.getItem(cacheKey);
      if (cachedRaw) {
        const parsed = JSON.parse(cachedRaw) as { fetchedAt: number; body: string };
        if (Date.now() - parsed.fetchedAt < CACHE_TTL_MS) {
          return { body: parsed.body, source: 'cache' };
        }
      }
    } catch {
      // Ignore
    }
  }

  let failure = 'EMPTY_BODY';
  try {
    const issue = await fetchJson<Record<string, unknown>>(
      'https://api.github.com/repos/xcl-dev/xcl/issues/1607'
    );
    const body = String(issue.body ?? '');
    if (body) {
      try {
        localStorage.setItem(cacheKey, JSON.stringify({ fetchedAt: Date.now(), body }));
      } catch {
        // Ignore
      }
      return { body, source: 'live' };
    }
  } catch (err) {
    failure = err instanceof Error ? err.message : 'UNKNOWN';
    // A stale, previously fetched copy is preferable to invented content.
    try {
      const cachedRaw = localStorage.getItem(cacheKey);
      if (cachedRaw) {
        const parsed = JSON.parse(cachedRaw) as { body: string };
        if (parsed.body) return { body: parsed.body, source: 'cache' };
      }
    } catch {
      // Ignore
    }
  }

  throw new Error(
    failure === 'RATE_LIMIT'
      ? 'GitHub rate limit reached while loading #1607, and no cached digest is available.'
      : 'Unable to load the #1607 triage digest, and no cached digest is available.'
  );
}
