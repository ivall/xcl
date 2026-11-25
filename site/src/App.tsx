import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CommitFeed } from './components/CommitFeed';
import { DeployGuideModal } from './components/DeployGuideModal';
import { ItemDetailPane } from './components/ItemDetailPane';
import { ItemList } from './components/ItemList';
import { NewIssueModal } from './components/NewIssueModal';
import { TopBar } from './components/TopBar';
import { TriageQueueView } from './components/TriageQueueView';
import { WorkspaceToolbar } from './components/WorkspaceToolbar';
import { XclTriageView } from './components/XclTriageView';
import { DEFAULT_REPO_SLUG } from './data/seedData';
import type {
  GitHubItem,
  ItemSortMode,
  ItemStateFilter,
  RepoBundle,
  TriageRecord,
  TriageStatus,
  WorkspaceTab,
} from './types/github';
import { fetchRepoBundle, fetchXcl1607Body } from './utils/githubApi';
import { filterAndSortItems, makeTriageKey, parseRepoInput } from './utils/githubParse';
import {
  loadRecentRepos,
  loadThemePreference,
  loadTriageMap,
  recordRecentRepo,
  saveThemePreference,
  saveTriageMap,
  upsertTriageRecord,
} from './utils/triageStore';
import { parseXcl1607Digest, type Xcl1607Digest } from './utils/xclTriageParser';

export default function App() {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('issues');
  const [owner, setOwner] = useState('xcl-dev');
  const [repo, setRepo] = useState('xcl');
  const [recentRepos, setRecentRepos] = useState<string[]>(() => loadRecentRepos());
  const [bundle, setBundle] = useState<RepoBundle | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [xclDigest, setXclDigest] = useState<Xcl1607Digest | null>(null);
  const [xclSource, setXclSource] = useState<'live' | 'cache' | null>(null);
  const [loadingXcl, setLoadingXcl] = useState(false);

  const [theme, setTheme] = useState<'light' | 'dark'>(() => loadThemePreference());
  const [triageMap, setTriageMap] = useState<Record<string, TriageRecord>>(() =>
    loadTriageMap()
  );

  // Filters
  const [search, setSearch] = useState('');
  const [stateFilter, setStateFilter] = useState<ItemStateFilter>('open');
  const [subsystem, setSubsystem] = useState('all');
  const [labelFilter, setLabelFilter] = useState('all');
  const [triageFilter, setTriageFilter] = useState<
    'all' | 'untriaged' | 'pinned' | TriageStatus
  >('all');
  const [sortMode, setSortMode] = useState<ItemSortMode>('updated');

  const [selectedNumber, setSelectedNumber] = useState<number | null>(null);
  const [newIssueOpen, setNewIssueOpen] = useState(false);
  const [deployGuideOpen, setDeployGuideOpen] = useState(false);

  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const repoSlug = `${owner}/${repo}`;

  // Apply dark mode class cleanly on root <html>
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    saveThemePreference(theme);
  }, [theme]);

  const loadRepo = useCallback(
    async (targetOwner: string, targetRepo: string, forceRefresh = false) => {
      setIsLoading(true);
      setLoadError(null);
      try {
        const data = await fetchRepoBundle(targetOwner, targetRepo, { forceRefresh });
        setBundle(data);
        setOwner(data.repo.owner);
        setRepo(data.repo.name);
        setRecentRepos(recordRecentRepo(data.repo.full_name));
      } catch (err) {
        setLoadError(
          err instanceof Error ? err.message : `Failed to load ${targetOwner}/${targetRepo}.`
        );
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const loadXcl1607 = useCallback(async (forceRefresh = false) => {
    setLoadingXcl(true);
    try {
      const res = await fetchXcl1607Body(forceRefresh);
      setXclDigest(parseXcl1607Digest(res.body));
      setXclSource(res.source);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Unable to load the #1607 triage digest.');
    } finally {
      setLoadingXcl(false);
    }
  }, []);

  useEffect(() => {
    const parsed = parseRepoInput(DEFAULT_REPO_SLUG);
    if (parsed.ok) {
      loadRepo(parsed.owner, parsed.repo);
    }
    loadXcl1607();
  }, [loadRepo, loadXcl1607]);

  const activeSourceItems = useMemo(() => {
    if (!bundle) return [];
    return activeTab === 'pulls' ? bundle.pulls : bundle.issues;
  }, [bundle, activeTab]);

  const filteredItems = useMemo(() => {
    return filterAndSortItems(activeSourceItems, {
      search,
      stateFilter,
      subsystem,
      label: labelFilter,
      triageFilter,
      sortMode,
      triageMap,
      repoSlug,
    });
  }, [
    activeSourceItems,
    search,
    stateFilter,
    subsystem,
    labelFilter,
    triageFilter,
    sortMode,
    triageMap,
    repoSlug,
  ]);

  const selectedItem = useMemo(() => {
    if (selectedNumber === null) return null;
    const pool = activeTab === 'pulls' ? bundle?.pulls ?? [] : bundle?.issues ?? [];
    return pool.find((i) => i.number === selectedNumber) ?? null;
  }, [selectedNumber, activeTab, bundle]);

  // Auto-select first visible item on desktop when switching tabs/repos if nothing valid is selected
  useEffect(() => {
    if (activeTab !== 'issues' && activeTab !== 'pulls') return;
    if (filteredItems.length === 0) {
      setSelectedNumber(null);
      return;
    }
    const stillExists = filteredItems.some((i) => i.number === selectedNumber);
    if (!stillExists && window.innerWidth >= 1024) {
      setSelectedNumber(filteredItems[0].number);
    }
  }, [activeTab, filteredItems, selectedNumber]);

  // Keyboard shortcuts: / focuses search, J/K navigate items
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isTyping =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable);

      if (e.key === '/' && !isTyping) {
        e.preventDefault();
        searchInputRef.current?.focus();
        return;
      }

      if (isTyping || newIssueOpen || deployGuideOpen) return;
      if (activeTab !== 'issues' && activeTab !== 'pulls') return;
      if (filteredItems.length === 0) return;

      if (e.key === 'j' || e.key === 'J' || e.key === 'ArrowDown') {
        e.preventDefault();
        const idx = filteredItems.findIndex((i) => i.number === selectedNumber);
        const nextIdx = idx < 0 ? 0 : Math.min(filteredItems.length - 1, idx + 1);
        setSelectedNumber(filteredItems[nextIdx].number);
      } else if (e.key === 'k' || e.key === 'K' || e.key === 'ArrowUp') {
        e.preventDefault();
        const idx = filteredItems.findIndex((i) => i.number === selectedNumber);
        const prevIdx = idx <= 0 ? 0 : idx - 1;
        setSelectedNumber(filteredItems[prevIdx].number);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeTab, filteredItems, selectedNumber, newIssueOpen, deployGuideOpen]);

  const handleSwitchRepo = (nextOwner: string, nextRepo: string) => {
    setSelectedNumber(null);
    setSubsystem('all');
    setLabelFilter('all');
    setSearch('');
    loadRepo(nextOwner, nextRepo);
  };

  const handleTabChange = (tab: WorkspaceTab) => {
    setActiveTab(tab);
    setSelectedNumber(null);
    setSubsystem('all');
    setLabelFilter('all');
    if (tab === 'issues' && stateFilter === 'draft') {
      setStateFilter('open');
    }
    if (tab === 'xcl1607' && !xclDigest) {
      loadXcl1607();
    }
  };

  const handleClearFilters = () => {
    setSearch('');
    setStateFilter('open');
    setSubsystem('all');
    setLabelFilter('all');
    setTriageFilter('all');
  };

  const handleUpdateTriage = (
    item: GitHubItem,
    patch: Partial<Pick<TriageRecord, 'status' | 'pinned' | 'note'>>
  ) => {
    setTriageMap((prev) => upsertTriageRecord(prev, repoSlug, item, patch));
  };

  const handleJumpToPr = (prNumber: number) => {
    setActiveTab('pulls');
    setStateFilter('all');
    setSearch('');
    setSubsystem('all');
    setLabelFilter('all');
    setTriageFilter('all');

    const exists = bundle?.pulls.some((p) => p.number === prNumber);
    if (exists) {
      setSelectedNumber(prNumber);
    } else {
      setSearch(`#${prNumber}`);
    }
  };

  const handleOpenFromTriageQueue = (record: TriageRecord) => {
    const parsed = parseRepoInput(record.repoSlug);
    if (parsed.ok && record.repoSlug.toLowerCase() !== repoSlug.toLowerCase()) {
      loadRepo(parsed.owner, parsed.repo);
    }
    setActiveTab(record.kind === 'pr' ? 'pulls' : 'issues');
    setStateFilter('all');
    setSelectedNumber(record.number);
  };

  const triageList = useMemo(() => Object.values(triageMap), [triageMap]);

  const summaryStats = useMemo(() => {
    const issuesOpen = bundle?.issues.filter((i) => i.state === 'open').length ?? 0;
    const pullsOpen = bundle?.pulls.filter((p) => p.state === 'open').length ?? 0;
    const pullsDraft = bundle?.pulls.filter((p) => p.state === 'open' && p.draft).length ?? 0;
    const triagedCurrent = triageList.filter(
      (t) => t.repoSlug.toLowerCase() === repoSlug.toLowerCase()
    ).length;

    return {
      issuesOpen,
      pullsOpen,
      pullsDraft,
      commitsRecent: bundle?.commits.length ?? 0,
      triagedCurrent,
    };
  }, [bundle, triageList, repoSlug]);

  return (
    <div id="top" className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <TopBar
        activeTab={activeTab}
        onSelectTab={handleTabChange}
        onOpenNewIssue={() => setNewIssueOpen(true)}
      />

      <WorkspaceToolbar
        activeRepoSlug={repoSlug}
        bundle={bundle}
        recentRepos={recentRepos}
        isLoading={isLoading}
        theme={theme}
        onToggleTheme={() => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))}
        onSwitchRepo={handleSwitchRepo}
        onRefresh={() => {
          loadRepo(owner, repo, true);
          loadXcl1607(true);
        }}
        onOpenDeployGuide={() => setDeployGuideOpen(true)}
      />

      <main className="mx-auto max-w-[1440px] px-4 py-5 sm:px-6">
        {/* Compact Repo Summary Strip */}
        <section
          aria-label="Repository overview metrics"
          className="mb-5 grid grid-cols-2 divide-x divide-y divide-slate-200 overflow-hidden rounded-xl border border-slate-200 bg-white sm:grid-cols-5 sm:divide-y-0 dark:divide-slate-800 dark:border-slate-800 dark:bg-slate-950"
        >
          <button
            type="button"
            onClick={() => handleTabChange('issues')}
            className="flex flex-col gap-0.5 px-4 py-3 text-left transition-colors hover:bg-slate-50 dark:hover:bg-slate-900/50"
          >
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Loaded Open Issues
            </span>
            <span className="font-mono text-lg font-bold text-slate-900 tabular-nums dark:text-slate-50">
              {summaryStats.issuesOpen}
            </span>
          </button>

          <button
            type="button"
            onClick={() => {
              handleTabChange('pulls');
              setStateFilter('open');
            }}
            className="flex flex-col gap-0.5 px-4 py-3 text-left transition-colors hover:bg-slate-50 dark:hover:bg-slate-900/50"
          >
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Loaded Open PRs
            </span>
            <span className="font-mono text-lg font-bold text-slate-900 tabular-nums dark:text-slate-50">
              {summaryStats.pullsOpen}
            </span>
          </button>

          <button
            type="button"
            onClick={() => {
              handleTabChange('pulls');
              setStateFilter('draft');
            }}
            className="flex flex-col gap-0.5 px-4 py-3 text-left transition-colors hover:bg-slate-50 dark:hover:bg-slate-900/50"
          >
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Draft PRs
            </span>
            <span className="font-mono text-lg font-bold text-slate-900 tabular-nums dark:text-slate-50">
              {summaryStats.pullsDraft}
            </span>
          </button>

          <button
            type="button"
            onClick={() => handleTabChange('xcl1607')}
            className="flex flex-col gap-0.5 px-4 py-3 text-left transition-colors hover:bg-slate-50 dark:hover:bg-slate-900/50"
          >
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              #1607 Needs 1 Report
            </span>
            <span className="font-mono text-lg font-bold text-amber-700 tabular-nums dark:text-amber-400">
              {xclDigest ? xclDigest.oneReport : '—'}
            </span>
          </button>

          <button
            type="button"
            onClick={() => handleTabChange('triage')}
            className="col-span-2 flex flex-col gap-0.5 px-4 py-3 text-left transition-colors hover:bg-slate-50 sm:col-span-1 dark:hover:bg-slate-900/50"
          >
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              In My Queue
            </span>
            <span className="font-mono text-lg font-bold text-blue-600 tabular-nums dark:text-blue-400">
              {summaryStats.triagedCurrent}
            </span>
          </button>
        </section>

        {loadError && (
          <div
            role="alert"
            className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs font-medium text-rose-900 dark:border-rose-900/60 dark:bg-rose-950/40 dark:text-rose-200"
          >
            <span>{loadError}</span>
            <button
              type="button"
              onClick={() => loadRepo(owner, repo, true)}
              className="rounded-md border border-rose-300 bg-white px-3 py-1.5 font-semibold text-rose-800 hover:bg-rose-100 dark:border-rose-800 dark:bg-slate-950 dark:text-rose-200"
            >
              Retry
            </button>
          </div>
        )}

        {(activeTab === 'issues' || activeTab === 'pulls') && (
          <div className="grid grid-cols-1 items-start gap-5 lg:grid-cols-12">
            <div
              className={`${
                selectedItem ? 'hidden lg:block' : 'block'
              } lg:col-span-7`}
            >
              <ItemList
                mode={activeTab === 'pulls' ? 'pr' : 'issue'}
                items={filteredItems}
                allItemsForFacets={activeSourceItems}
                totalCountHint={
                  activeTab === 'pulls'
                    ? bundle?.pulls.length ?? 0
                    : bundle?.issuesTotalCount ?? 0
                }
                selectedNumber={selectedNumber}
                search={search}
                stateFilter={stateFilter}
                subsystem={subsystem}
                labelFilter={labelFilter}
                triageFilter={triageFilter}
                sortMode={sortMode}
                triageMap={triageMap}
                repoSlug={repoSlug}
                isLoading={isLoading}
                searchInputRef={searchInputRef}
                onSearchChange={setSearch}
                onStateFilterChange={setStateFilter}
                onSubsystemChange={setSubsystem}
                onLabelFilterChange={setLabelFilter}
                onTriageFilterChange={setTriageFilter}
                onSortModeChange={setSortMode}
                onSelectItem={(item) => setSelectedNumber(item.number)}
                onClearFilters={handleClearFilters}
              />
            </div>

            <div
              className={`${
                selectedItem ? 'block' : 'hidden lg:block'
              } lg:sticky lg:top-4 lg:col-span-5`}
            >
              <ItemDetailPane
                owner={owner}
                repo={repo}
                item={selectedItem}
                triageRecord={
                  selectedItem
                    ? triageMap[makeTriageKey(repoSlug, selectedItem.number)]
                    : undefined
                }
                xclGroupIds={
                  selectedItem?.kind === 'pr'
                    ? xclDigest?.prToGroups[selectedItem.number]
                    : undefined
                }
                onUpdateTriage={handleUpdateTriage}
                onCloseMobile={() => setSelectedNumber(null)}
              />
            </div>
          </div>
        )}

        {activeTab === 'commits' && (
          <CommitFeed
            commits={bundle?.commits ?? []}
            defaultBranch={bundle?.repo.default_branch ?? 'main'}
            repoSlug={repoSlug}
            isLoading={isLoading}
            onJumpToPr={handleJumpToPr}
          />
        )}

        {activeTab === 'xcl1607' && (
          <XclTriageView
            digest={xclDigest}
            source={xclSource}
            isLoading={loadingXcl}
            knownPulls={bundle?.pulls ?? []}
            onRefresh={() => loadXcl1607(true)}
          />
        )}

        {activeTab === 'triage' && (
          <TriageQueueView
            repoSlug={repoSlug}
            records={triageList}
            onChangeStatus={(rec, status) => {
              const updated = {
                ...triageMap,
                [rec.key]: { ...rec, status, updatedAt: new Date().toISOString() },
              };
              if (!status && !rec.pinned && !rec.note.trim()) {
                delete updated[rec.key];
              }
              saveTriageMap(updated);
              setTriageMap(updated);
            }}
            onTogglePin={(rec) => {
              const nextPinned = !rec.pinned;
              const updated = {
                ...triageMap,
                [rec.key]: { ...rec, pinned: nextPinned, updatedAt: new Date().toISOString() },
              };
              if (!rec.status && !nextPinned && !rec.note.trim()) {
                delete updated[rec.key];
              }
              saveTriageMap(updated);
              setTriageMap(updated);
            }}
            onRemoveRecord={(rec) => {
              const updated = { ...triageMap };
              delete updated[rec.key];
              saveTriageMap(updated);
              setTriageMap(updated);
            }}
            onOpenItem={handleOpenFromTriageQueue}
          />
        )}
      </main>

      <NewIssueModal
        isOpen={newIssueOpen}
        owner={owner}
        repo={repo}
        onClose={() => setNewIssueOpen(false)}
      />

      <DeployGuideModal
        isOpen={deployGuideOpen}
        onClose={() => setDeployGuideOpen(false)}
      />
    </div>
  );
}
