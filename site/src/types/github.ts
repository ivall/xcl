export type WorkspaceTab = 'issues' | 'pulls' | 'commits' | 'xcl1607' | 'triage';

export type TriageStatus =
  | 'needs_review'
  | 'in_progress'
  | 'ready_to_merge'
  | 'blocked'
  | 'snoozed';

export type ItemStateFilter = 'open' | 'closed' | 'all' | 'draft';
export type ItemSortMode = 'updated' | 'created' | 'comments' | 'needs_triage';

export interface GitHubUser {
  login: string;
  avatar_url: string;
  html_url: string;
}

export interface GitHubLabel {
  id: number;
  name: string;
  color: string;
  description?: string | null;
}

export interface RepoInfo {
  full_name: string;
  owner: string;
  name: string;
  description: string | null;
  html_url: string;
  stargazers_count: number;
  forks_count: number;
  open_issues_count: number;
  default_branch: string;
  language: string | null;
  pushed_at: string;
}

export interface GitHubItem {
  id: number;
  number: number;
  kind: 'issue' | 'pr';
  title: string;
  body: string;
  state: 'open' | 'closed';
  html_url: string;
  user: GitHubUser;
  labels: GitHubLabel[];
  assignees: GitHubUser[];
  comments: number;
  created_at: string;
  updated_at: string;
  closed_at: string | null;
  draft?: boolean;
  merged_at?: string | null;
  head_ref?: string;
  head_sha?: string;
  base_ref?: string;
  additions?: number;
  deletions?: number;
  changed_files?: number;
}

export interface GitHubComment {
  id: number;
  user: GitHubUser;
  body: string;
  created_at: string;
  updated_at: string;
  html_url: string;
}

export interface GitHubCommit {
  sha: string;
  shortSha: string;
  messageHeadline: string;
  messageBody: string;
  authorName: string;
  authorLogin: string | null;
  authorAvatarUrl: string | null;
  date: string;
  html_url: string;
  linkedPrNumber: number | null;
}

export interface TriageRecord {
  key: string; // `${repoSlug}#${number}`
  repoSlug: string;
  number: number;
  kind: 'issue' | 'pr';
  title: string;
  html_url: string;
  status: TriageStatus | null;
  pinned: boolean;
  note: string;
  updatedAt: string;
}

export interface RepoBundle {
  repo: RepoInfo;
  issues: GitHubItem[];
  issuesTotalCount: number;
  pulls: GitHubItem[];
  commits: GitHubCommit[];
  source: 'live' | 'cache';
  fetchedAt: string;
  notice?: string;
}

export interface IssueTemplate {
  id: string;
  label: string;
  defaultLabels: string[];
  titlePrefix: string;
  body: string;
}
