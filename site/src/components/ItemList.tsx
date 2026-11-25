import React from 'react';
import {
  CircleDot,
  CheckCircle2,
  GitPullRequest,
  GitMerge,
  GitPullRequestDraft,
  Pin,
  Search,
  MessageSquare,
} from 'lucide-react';
import type {
  GitHubItem,
  ItemSortMode,
  ItemStateFilter,
  TriageRecord,
  TriageStatus,
} from '../types/github';
import {
  extractLabelCounts,
  extractSubsystemCounts,
  formatRelativeTime,
  makeTriageKey,
  parseSubsystemPrefix,
} from '../utils/githubParse';
import { TRIAGE_STATUS_META } from '../utils/triageStore';
import { Avatar } from './Avatar';

interface ItemListProps {
  mode: 'issue' | 'pr';
  items: GitHubItem[];
  allItemsForFacets: GitHubItem[];
  totalCountHint: number;
  selectedNumber: number | null;
  search: string;
  stateFilter: ItemStateFilter;
  subsystem: string;
  labelFilter: string;
  triageFilter: 'all' | 'untriaged' | 'pinned' | TriageStatus;
  sortMode: ItemSortMode;
  triageMap: Record<string, TriageRecord>;
  repoSlug: string;
  isLoading: boolean;
  searchInputRef: React.RefObject<HTMLInputElement | null>;
  onSearchChange: (val: string) => void;
  onStateFilterChange: (val: ItemStateFilter) => void;
  onSubsystemChange: (val: string) => void;
  onLabelFilterChange: (val: string) => void;
  onTriageFilterChange: (val: 'all' | 'untriaged' | 'pinned' | TriageStatus) => void;
  onSortModeChange: (val: ItemSortMode) => void;
  onSelectItem: (item: GitHubItem) => void;
  onClearFilters: () => void;
}

function ItemStateLabel({ item }: { item: GitHubItem }) {
  if (item.kind === 'pr') {
    if (item.merged_at) {
      return (
        <span className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-violet-700 dark:text-violet-400">
          <GitMerge className="h-3.5 w-3.5" aria-hidden="true" />
          <span>Merged</span>
        </span>
      );
    }
    if (item.state === 'closed') {
      return (
        <span className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-rose-700 dark:text-rose-400">
          <GitPullRequest className="h-3.5 w-3.5" aria-hidden="true" />
          <span>Closed</span>
        </span>
      );
    }
    if (item.draft) {
      return (
        <span className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-slate-600 dark:text-slate-400">
          <GitPullRequestDraft className="h-3.5 w-3.5" aria-hidden="true" />
          <span>Draft</span>
        </span>
      );
    }
    return (
      <span className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-emerald-700 dark:text-emerald-400">
        <GitPullRequest className="h-3.5 w-3.5" aria-hidden="true" />
        <span>Open</span>
      </span>
    );
  }

  if (item.state === 'closed') {
    return (
      <span className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-violet-700 dark:text-violet-400">
        <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
        <span>Closed</span>
      </span>
    );
  }

  return (
    <span className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-emerald-700 dark:text-emerald-400">
      <CircleDot className="h-3.5 w-3.5" aria-hidden="true" />
      <span>Open</span>
    </span>
  );
}

export function ItemList({
  mode,
  items,
  allItemsForFacets,
  totalCountHint,
  selectedNumber,
  search,
  stateFilter,
  subsystem,
  labelFilter,
  triageFilter,
  sortMode,
  triageMap,
  repoSlug,
  isLoading,
  searchInputRef,
  onSearchChange,
  onStateFilterChange,
  onSubsystemChange,
  onLabelFilterChange,
  onTriageFilterChange,
  onSortModeChange,
  onSelectItem,
  onClearFilters,
}: ItemListProps) {
  const subsystems = extractSubsystemCounts(allItemsForFacets);
  const labels = extractLabelCounts(allItemsForFacets);
  const stateOptions: Array<{ id: ItemStateFilter; label: string }> =
    mode === 'pr'
      ? [
          { id: 'open', label: 'Open' },
          { id: 'draft', label: 'Draft' },
          { id: 'closed', label: 'Closed' },
          { id: 'all', label: 'All' },
        ]
      : [
          { id: 'open', label: 'Open' },
          { id: 'closed', label: 'Closed' },
          { id: 'all', label: 'All' },
        ];

  const hasActiveFilter =
    search.trim().length > 0 ||
    stateFilter !== 'open' ||
    subsystem !== 'all' ||
    labelFilter !== 'all' ||
    triageFilter !== 'all';

  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950">
      {/* Search & Filter Toolbar */}
      <div className="flex flex-col gap-3 border-b border-slate-200 p-4 dark:border-slate-800">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="relative flex-1">
            <Search
              className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-slate-400"
              aria-hidden="true"
            />
            <label htmlFor="item-search-input" className="sr-only">
              Search {mode === 'pr' ? 'pull requests' : 'issues'}
            </label>
            <input
              ref={searchInputRef}
              id="item-search-input"
              type="search"
              value={search}
              onChange={(e) => onSearchChange(e.target.value)}
              placeholder={`Search ${mode === 'pr' ? 'PRs' : 'issues'} by #number, title, author, branch... (/)`}
              className="h-9 w-full rounded-lg border border-slate-200 bg-slate-50 pr-3 pl-9 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:bg-white focus:outline-none dark:border-slate-800 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-blue-500 dark:focus:bg-slate-950"
            />
          </div>

          {/* Segmented state control */}
          <div
            role="group"
            aria-label="Filter by state"
            className="inline-flex shrink-0 items-center gap-1 rounded-lg bg-slate-100 p-1 dark:bg-slate-900"
          >
            {stateOptions.map((opt) => {
              const active = stateFilter === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => onStateFilterChange(opt.id)}
                  aria-pressed={active}
                  className={`shrink-0 whitespace-nowrap rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                    active
                      ? 'bg-white text-slate-900 shadow-xs dark:bg-slate-800 dark:text-slate-50'
                      : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100'
                  }`}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div>
            <label htmlFor="subsystem-select" className="sr-only">
              Subsystem
            </label>
            <select
              id="subsystem-select"
              value={subsystem}
              onChange={(e) => onSubsystemChange(e.target.value)}
              className="h-8 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 focus:border-blue-600 focus:outline-none dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200"
            >
              <option value="all">All Subsystems</option>
              {subsystems.map((sub) => (
                <option key={sub.name} value={sub.name}>
                  {sub.name} ({sub.count})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="label-select" className="sr-only">
              GitHub Label
            </label>
            <select
              id="label-select"
              value={labelFilter}
              onChange={(e) => onLabelFilterChange(e.target.value)}
              className="h-8 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 focus:border-blue-600 focus:outline-none dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200"
            >
              <option value="all">All Labels</option>
              {labels.map((lbl) => (
                <option key={lbl.name} value={lbl.name}>
                  {lbl.name} ({lbl.count})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="triage-filter-select" className="sr-only">
              Filter by Triage Status
            </label>
            <select
              id="triage-filter-select"
              value={triageFilter}
              onChange={(e) =>
                onTriageFilterChange(
                  e.target.value as 'all' | 'untriaged' | 'pinned' | TriageStatus
                )
              }
              className="h-8 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 focus:border-blue-600 focus:outline-none dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200"
            >
              <option value="all">All Triage States</option>
              <option value="untriaged">Untriaged Only</option>
              <option value="pinned">Pinned</option>
              <option value="needs_review">Needs Review</option>
              <option value="in_progress">In Progress</option>
              <option value="ready_to_merge">Ready to Merge</option>
              <option value="blocked">Blocked</option>
              <option value="snoozed">Snoozed</option>
            </select>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <label htmlFor="sort-select" className="sr-only">
              Sort order
            </label>
            <select
              id="sort-select"
              value={sortMode}
              onChange={(e) => onSortModeChange(e.target.value as ItemSortMode)}
              className="h-8 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 focus:border-blue-600 focus:outline-none dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200"
            >
              <option value="updated">Sort: Recently Updated</option>
              <option value="created">Sort: Newest Opened</option>
              <option value="comments">Sort: Most Discussed</option>
              <option value="needs_triage">Sort: Needs Triage First</option>
            </select>
          </div>
        </div>

        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
          <span className="tabular-nums">
            Showing {items.length} of {allItemsForFacets.length} loaded{' '}
            {mode === 'pr' ? 'pull requests' : 'issues'}
            {totalCountHint > allItemsForFacets.length
              ? ` (${totalCountHint.toLocaleString('en-US')} total on GitHub)`
              : ''}
          </span>
          <span className="hidden sm:inline">Keys: J / K to move · Enter to open</span>
        </div>
      </div>

      {/* Rows */}
      {isLoading && allItemsForFacets.length === 0 ? (
        <div aria-label="Loading items" className="divide-y divide-slate-200 dark:divide-slate-800">
          {Array.from({ length: 6 }).map((_, idx) => (
            <div key={idx} className="flex flex-col gap-2 px-4 py-3.5">
              <div className="h-4 w-3/4 animate-pulse rounded bg-slate-200 dark:bg-slate-800" />
              <div className="h-3 w-1/2 animate-pulse rounded bg-slate-100 dark:bg-slate-900" />
            </div>
          ))}
        </div>
      ) : items.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
          <p className="text-sm font-medium text-slate-900 dark:text-slate-100">
            No {mode === 'pr' ? 'pull requests' : 'issues'} match the current filters.
          </p>
          <p className="max-w-sm text-xs text-slate-500 dark:text-slate-400">
            Try switching the state filter to All or clearing your subsystem and text filters.
          </p>
          {hasActiveFilter && (
            <button
              type="button"
              onClick={onClearFilters}
              className="mt-1 rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              Clear Filters
            </button>
          )}
        </div>
      ) : (
        <div
          role="list"
          aria-label={mode === 'pr' ? 'Pull requests list' : 'Issues list'}
          className="divide-y divide-slate-200 dark:divide-slate-800"
        >
          {items.map((item) => {
            const isSelected = selectedNumber === item.number;
            const triage = triageMap[makeTriageKey(repoSlug, item.number)];
            const subsystemPrefix = parseSubsystemPrefix(item.title);
            const triageMeta = triage?.status ? TRIAGE_STATUS_META[triage.status] : null;

            return (
              <div key={item.number} role="listitem">
                <button
                  type="button"
                  onClick={() => onSelectItem(item)}
                  aria-current={isSelected ? 'true' : undefined}
                  className={`flex w-full flex-col gap-1.5 px-4 py-3 text-left transition-colors focus-visible:outline-2 focus-visible:outline-inset focus-visible:outline-blue-600 ${
                    isSelected
                      ? 'bg-blue-50/70 dark:bg-blue-950/30'
                      : 'hover:bg-slate-50 dark:hover:bg-slate-900/60'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline gap-2">
                        <span className="shrink-0 font-mono text-xs font-medium text-slate-500 tabular-nums dark:text-slate-400">
                          #{item.number}
                        </span>
                        <span className="text-sm font-semibold leading-snug text-slate-900 dark:text-slate-100">
                          {item.title}
                        </span>
                      </div>
                    </div>

                    <div className="flex shrink-0 items-center gap-2.5 pt-0.5">
                      {triage?.pinned && (
                        <Pin
                          className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400"
                          aria-label="Pinned"
                        />
                      )}
                      {triageMeta && (
                        <span className={`text-xs font-semibold ${triageMeta.textClass}`}>
                          {triageMeta.shortLabel}
                        </span>
                      )}
                      <ItemStateLabel item={item} />
                    </div>
                  </div>

                  {/* Clean unboxed metadata line */}
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
                    <span className="inline-flex items-center gap-1.5 text-slate-700 dark:text-slate-300">
                      <Avatar src={item.user.avatar_url} name={item.user.login} size="sm" />
                      <span>{item.user.login}</span>
                    </span>
                    <span aria-hidden="true">·</span>
                    <span className="tabular-nums">updated {formatRelativeTime(item.updated_at)}</span>

                    {item.kind === 'pr' && item.head_ref && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="max-w-[220px] truncate font-mono text-[11px] text-slate-600 dark:text-slate-400">
                          {item.head_ref} → {item.base_ref ?? 'main'}
                        </span>
                      </>
                    )}

                    {item.comments > 0 && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="inline-flex items-center gap-1 tabular-nums">
                          <MessageSquare className="h-3 w-3" aria-hidden="true" />
                          {item.comments}
                        </span>
                      </>
                    )}

                    {subsystemPrefix && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="font-medium text-slate-600 dark:text-slate-300">
                          {subsystemPrefix}
                        </span>
                      </>
                    )}

                    {item.labels.slice(0, 3).map((lbl) => (
                      <span
                        key={lbl.id || lbl.name}
                        className="inline-flex items-center gap-1 text-slate-600 dark:text-slate-300"
                      >
                        <span aria-hidden="true">·</span>
                        <span
                          aria-hidden="true"
                          className="inline-block h-2 w-2 rounded-[2px]"
                          style={{ backgroundColor: `#${lbl.color}` }}
                        />
                        <span>{lbl.name}</span>
                      </span>
                    ))}
                  </div>
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
