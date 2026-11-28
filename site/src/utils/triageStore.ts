import type { GitHubItem, TriageRecord, TriageStatus } from '../types/github';
import { makeTriageKey } from './githubParse';

const TRIAGE_STORAGE_KEY = 'repopulse:triage:v1';
const RECENT_REPOS_KEY = 'repopulse:recent_repos:v1';
const THEME_STORAGE_KEY = 'repopulse:theme:v1';

export const TRIAGE_STATUS_META: Record<
  TriageStatus,
  { label: string; shortLabel: string; textClass: string; dotClass: string }
> = {
  needs_review: {
    label: 'Needs Review',
    shortLabel: 'Review',
    textClass: 'text-amber-700 dark:text-amber-400',
    dotClass: 'bg-amber-600 dark:bg-amber-400',
  },
  in_progress: {
    label: 'In Progress',
    shortLabel: 'Active',
    textClass: 'text-blue-700 dark:text-blue-400',
    dotClass: 'bg-blue-600 dark:bg-blue-400',
  },
  ready_to_merge: {
    label: 'Ready to Merge',
    shortLabel: 'Ready',
    textClass: 'text-emerald-700 dark:text-emerald-400',
    dotClass: 'bg-emerald-600 dark:bg-emerald-400',
  },
  blocked: {
    label: 'Blocked',
    shortLabel: 'Blocked',
    textClass: 'text-rose-700 dark:text-rose-400',
    dotClass: 'bg-rose-600 dark:bg-rose-400',
  },
  snoozed: {
    label: 'Snoozed',
    shortLabel: 'Snoozed',
    textClass: 'text-slate-600 dark:text-slate-400',
    dotClass: 'bg-slate-400 dark:bg-slate-500',
  },
};

export function loadTriageMap(): Record<string, TriageRecord> {
  try {
    const raw = localStorage.getItem(TRIAGE_STORAGE_KEY);
    if (!raw) return {};
    return JSON.parse(raw) as Record<string, TriageRecord>;
  } catch {
    return {};
  }
}

export function saveTriageMap(map: Record<string, TriageRecord>): void {
  try {
    localStorage.setItem(TRIAGE_STORAGE_KEY, JSON.stringify(map));
  } catch {
    // Ignore storage quota errors
  }
}

export function upsertTriageRecord(
  currentMap: Record<string, TriageRecord>,
  repoSlug: string,
  item: GitHubItem,
  patch: Partial<Pick<TriageRecord, 'status' | 'pinned' | 'note'>>
): Record<string, TriageRecord> {
  const key = makeTriageKey(repoSlug, item.number);
  const existing = currentMap[key];
  const next: TriageRecord = {
    key,
    repoSlug,
    number: item.number,
    kind: item.kind,
    title: item.title,
    html_url: item.html_url,
    status: patch.status !== undefined ? patch.status : existing?.status ?? null,
    pinned: patch.pinned !== undefined ? patch.pinned : existing?.pinned ?? false,
    note: patch.note !== undefined ? patch.note : existing?.note ?? '',
    updatedAt: new Date().toISOString(),
  };

  const updatedMap = { ...currentMap };
  if (!next.status && !next.pinned && !next.note.trim()) {
    delete updatedMap[key];
  } else {
    updatedMap[key] = next;
  }

  saveTriageMap(updatedMap);
  return updatedMap;
}

export function loadRecentRepos(): string[] {
  try {
    const raw = localStorage.getItem(RECENT_REPOS_KEY);
    if (!raw) return ['xcl-dev/xcl'];
    const parsed = JSON.parse(raw) as string[];
    return parsed.length > 0 ? parsed : ['xcl-dev/xcl'];
  } catch {
    return ['xcl-dev/xcl'];
  }
}

export function recordRecentRepo(slug: string): string[] {
  const current = loadRecentRepos().filter((s) => s.toLowerCase() !== slug.toLowerCase());
  const next = [slug, ...current].slice(0, 6);
  try {
    localStorage.setItem(RECENT_REPOS_KEY, JSON.stringify(next));
  } catch {
    // Ignore
  }
  return next;
}

export function loadThemePreference(): 'light' | 'dark' {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'dark' || saved === 'light') return saved;
  } catch {
    // Ignore
  }
  return 'light';
}

export function saveThemePreference(theme: 'light' | 'dark'): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Ignore
  }
}
