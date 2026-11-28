import { useMemo, useState } from 'react';
import { ExternalLink, GitCommitHorizontal, Search } from 'lucide-react';
import type { GitHubCommit } from '../types/github';
import { formatRelativeTime } from '../utils/githubParse';
import { Avatar } from './Avatar';

interface CommitFeedProps {
  commits: GitHubCommit[];
  defaultBranch: string;
  repoSlug: string;
  isLoading: boolean;
  onJumpToPr: (prNumber: number) => void;
}

export function CommitFeed({
  commits,
  defaultBranch,
  repoSlug,
  isLoading,
  onJumpToPr,
}: CommitFeedProps) {
  const [search, setSearch] = useState('');
  const [onlyPrMerges, setOnlyPrMerges] = useState(false);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return commits.filter((c) => {
      if (onlyPrMerges && c.linkedPrNumber === null) return false;
      if (!q) return true;
      return (
        c.shortSha.toLowerCase().includes(q) ||
        c.messageHeadline.toLowerCase().includes(q) ||
        c.authorName.toLowerCase().includes(q) ||
        (c.linkedPrNumber !== null && String(c.linkedPrNumber).includes(q.replace(/^#/, '')))
      );
    });
  }, [commits, search, onlyPrMerges]);

  return (
    <section
      aria-label="Recent commits"
      className="overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950"
    >
      <div className="flex flex-col gap-3 border-b border-slate-200 p-4 sm:flex-row sm:items-center sm:justify-between dark:border-slate-800">
        <div>
          <h1 className="text-base font-bold text-slate-900 dark:text-slate-50">
            Commit Pulse on <span className="font-mono">{defaultBranch}</span>
          </h1>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
            Latest {commits.length} commits in {repoSlug}. Click any linked PR number to inspect it.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <Search
              className="pointer-events-none absolute top-1/2 left-3 h-3.5 w-3.5 -translate-y-1/2 text-slate-400"
              aria-hidden="true"
            />
            <label htmlFor="commit-search-input" className="sr-only">
              Search commits
            </label>
            <input
              id="commit-search-input"
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter commits, SHA, author..."
              className="h-9 w-56 rounded-lg border border-slate-200 bg-slate-50 pr-3 pl-8 text-xs text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:bg-white focus:outline-none dark:border-slate-800 dark:bg-slate-900 dark:text-slate-100"
            />
          </div>

          <button
            type="button"
            onClick={() => setOnlyPrMerges((prev) => !prev)}
            aria-pressed={onlyPrMerges}
            className={`h-9 shrink-0 whitespace-nowrap rounded-lg border px-3 text-xs font-medium transition-colors ${
              onlyPrMerges
                ? 'border-blue-600 bg-blue-50 text-blue-800 dark:border-blue-500 dark:bg-blue-950/50 dark:text-blue-200'
                : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800'
            }`}
          >
            Merged PRs Only
          </button>
        </div>
      </div>

      {isLoading && commits.length === 0 ? (
        <div className="divide-y divide-slate-200 dark:divide-slate-800">
          {Array.from({ length: 6 }).map((_, idx) => (
            <div key={idx} className="flex items-center gap-4 px-4 py-3">
              <div className="h-4 w-16 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
              <div className="h-4 flex-1 animate-pulse rounded bg-slate-100 dark:bg-slate-900" />
            </div>
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="px-6 py-12 text-center">
          <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
            No commits match that filter.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/70 text-xs font-semibold text-slate-600 dark:border-slate-800 dark:bg-slate-900/50 dark:text-slate-400">
                <th className="px-4 py-2.5">SHA</th>
                <th className="px-4 py-2.5">Message</th>
                <th className="px-4 py-2.5">Author</th>
                <th className="px-4 py-2.5 text-right">Time</th>
                <th className="px-4 py-2.5 text-right">Link</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {filtered.map((commit) => (
                <tr
                  key={commit.sha}
                  className="hover:bg-slate-50/80 dark:hover:bg-slate-900/50"
                >
                  <td className="px-4 py-3 align-top font-mono text-xs font-semibold text-slate-700 tabular-nums dark:text-slate-300">
                    <span className="inline-flex items-center gap-1.5">
                      <GitCommitHorizontal className="h-3.5 w-3.5 text-slate-400" aria-hidden="true" />
                      {commit.shortSha}
                    </span>
                  </td>
                  <td className="px-4 py-3 align-top">
                    <div className="font-medium text-slate-900 dark:text-slate-100">
                      {commit.messageHeadline}
                    </div>
                    {commit.linkedPrNumber !== null && (
                      <div className="mt-1">
                        <button
                          type="button"
                          onClick={() => onJumpToPr(commit.linkedPrNumber!)}
                          className="font-mono text-xs font-semibold text-blue-600 hover:underline dark:text-blue-400"
                        >
                          Inspect PR #{commit.linkedPrNumber}
                        </button>
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 align-top text-xs text-slate-700 dark:text-slate-300">
                    <span className="inline-flex items-center gap-2 whitespace-nowrap">
                      <Avatar
                        src={commit.authorAvatarUrl}
                        name={commit.authorLogin ?? commit.authorName}
                        size="sm"
                      />
                      <span>{commit.authorLogin ?? commit.authorName}</span>
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right align-top font-mono text-xs whitespace-nowrap text-slate-500 tabular-nums dark:text-slate-400">
                    {formatRelativeTime(commit.date)}
                  </td>
                  <td className="px-4 py-3 text-right align-top">
                    <a
                      href={commit.html_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label={`Open commit ${commit.shortSha} on GitHub`}
                      className="inline-flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
                    >
                      <span>Diff</span>
                      <ExternalLink className="h-3 w-3" aria-hidden="true" />
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
