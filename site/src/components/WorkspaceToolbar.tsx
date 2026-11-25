import React, { useState, useEffect } from 'react';
import { RefreshCw, ExternalLink, Moon, Sun, Globe } from 'lucide-react';
import type { RepoBundle } from '../types/github';
import { formatCompactNumber, formatRelativeTime, parseRepoInput } from '../utils/githubParse';

interface WorkspaceToolbarProps {
  activeRepoSlug: string;
  bundle: RepoBundle | null;
  recentRepos: string[];
  isLoading: boolean;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  onSwitchRepo: (owner: string, repo: string) => void;
  onRefresh: () => void;
  onOpenDeployGuide: () => void;
}

export function WorkspaceToolbar({
  activeRepoSlug,
  bundle,
  recentRepos,
  isLoading,
  theme,
  onToggleTheme,
  onSwitchRepo,
  onRefresh,
  onOpenDeployGuide,
}: WorkspaceToolbarProps) {
  const [repoInput, setRepoInput] = useState(activeRepoSlug);
  const [validationError, setValidationError] = useState<string | null>(null);

  useEffect(() => {
    setRepoInput(activeRepoSlug);
    setValidationError(null);
  }, [activeRepoSlug]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const parsed = parseRepoInput(repoInput);
    if (!parsed.ok) {
      setValidationError(parsed.error);
      return;
    }
    setValidationError(null);
    onSwitchRepo(parsed.owner, parsed.repo);
  };

  return (
    <section
      aria-label="Repository controls"
      className="border-b border-slate-200 bg-slate-50/70 dark:border-slate-800 dark:bg-slate-900/50"
    >
      <div className="mx-auto flex max-w-[1440px] flex-col gap-3 px-4 py-3 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-col gap-1.5">
          <form onSubmit={handleSubmit} className="flex items-center gap-2" noValidate>
            <label htmlFor="repo-switcher-input" className="sr-only">
              GitHub repository owner and name or URL
            </label>
            <input
              id="repo-switcher-input"
              type="text"
              value={repoInput}
              onChange={(e) => {
                setRepoInput(e.target.value);
                if (validationError) setValidationError(null);
              }}
              placeholder="owner/repo or GitHub URL"
              aria-invalid={Boolean(validationError)}
              aria-describedby={validationError ? 'repo-input-error' : undefined}
              className="h-9 min-w-0 flex-1 rounded-lg border border-slate-300 bg-white px-3 font-mono text-xs text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none sm:w-72 sm:flex-none dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:focus:border-blue-500"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="h-9 shrink-0 whitespace-nowrap rounded-lg border border-slate-300 bg-white px-3.5 text-xs font-semibold text-slate-800 transition-colors hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              Load Repo
            </button>

            {recentRepos.length > 1 && (
              <div className="hidden items-center gap-2 pl-2 text-xs text-slate-500 xl:flex dark:text-slate-400">
                <span>Recent:</span>
                {recentRepos
                  .filter((slug) => slug.toLowerCase() !== activeRepoSlug.toLowerCase())
                  .slice(0, 3)
                  .map((slug) => (
                    <button
                      key={slug}
                      type="button"
                      onClick={() => {
                        const parsed = parseRepoInput(slug);
                        if (parsed.ok) onSwitchRepo(parsed.owner, parsed.repo);
                      }}
                      className="font-mono text-slate-600 underline decoration-slate-300 underline-offset-2 hover:text-slate-900 dark:text-slate-300 dark:decoration-slate-700 dark:hover:text-white"
                    >
                      {slug}
                    </button>
                  ))}
              </div>
            )}
          </form>

          {validationError && (
            <p id="repo-input-error" role="alert" className="text-xs font-medium text-rose-600 dark:text-rose-400">
              {validationError}
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 lg:justify-end">
          {bundle && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-600 dark:text-slate-400">
              <a
                href={bundle.repo.html_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 font-mono font-semibold text-slate-900 hover:text-blue-600 dark:text-slate-100 dark:hover:text-blue-400"
              >
                {bundle.repo.full_name}
                <ExternalLink className="h-3 w-3" aria-hidden="true" />
              </a>
              <span className="font-mono tabular-nums">
                {formatCompactNumber(bundle.repo.stargazers_count)} stars
              </span>
              <span className="font-mono tabular-nums">
                {formatCompactNumber(bundle.repo.open_issues_count)} open
              </span>
              <span>
                {bundle.source === 'live'
                  ? 'Live sync'
                  : bundle.source === 'cache'
                    ? 'Cached'
                    : 'Cached GitHub data'}{' '}
                ({formatRelativeTime(bundle.fetchedAt)})
              </span>
            </div>
          )}

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onRefresh}
              disabled={isLoading}
              aria-label="Sync repository data"
              className="inline-flex h-9 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg border border-slate-300 bg-white px-3 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} aria-hidden="true" />
              <span>Sync</span>
            </button>

            <button
              type="button"
              onClick={onOpenDeployGuide}
              className="inline-flex h-9 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg border border-slate-300 bg-white px-3 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              <Globe className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Host on GitHub</span>
            </button>

            <button
              type="button"
              onClick={onToggleTheme}
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-300 bg-white text-slate-700 transition-colors hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              {theme === 'dark' ? (
                <Sun className="h-4 w-4" aria-hidden="true" />
              ) : (
                <Moon className="h-4 w-4" aria-hidden="true" />
              )}
            </button>
          </div>
        </div>
      </div>

      {bundle?.notice && (
        <div className="border-t border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-900 sm:px-6 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-200">
          <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-4">
            <span>{bundle.notice}</span>
            <button
              type="button"
              onClick={onRefresh}
              className="shrink-0 font-semibold underline underline-offset-2 hover:no-underline"
            >
              Retry live sync
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
