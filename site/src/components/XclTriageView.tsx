import { useEffect, useMemo, useState } from 'react';
import {
  Check,
  Copy,
  ExternalLink,
  GitBranch,
  RefreshCw,
  Search,
} from 'lucide-react';
import type { GitHubItem } from '../types/github';
import { fetchPullByNumber } from '../utils/githubApi';
import {
  buildXclCheckBlock,
  buildXclTestReportComment,
  type Xcl1607Digest,
  type XclGroupUnit,
} from '../utils/xclTriageParser';
import { Avatar } from './Avatar';

type XclSubTab = 'needs_one' | 'needs_first' | 'ready_merge' | 'groups';

interface XclTriageViewProps {
  digest: Xcl1607Digest | null;
  source: 'live' | 'cache' | null;
  isLoading: boolean;
  knownPulls: GitHubItem[];
  onRefresh: () => void;
}

const HARDWARE_CLASSES = [
  { id: 'nv-win', label: 'nv-win (NVIDIA / Windows)' },
  { id: 'nv-linux', label: 'nv-linux (NVIDIA / Linux)' },
  { id: 'amd-win', label: 'amd-win (AMD / Windows)' },
  { id: 'amd-radv', label: 'amd-radv (AMD / RADV)' },
  { id: 'intel', label: 'intel' },
  { id: 'other', label: 'other' },
];

export function XclTriageView({
  digest,
  source,
  isLoading,
  knownPulls,
  onRefresh,
}: XclTriageViewProps) {
  const [subTab, setSubTab] = useState<XclSubTab>('needs_one');
  const [search, setSearch] = useState('');
  const [groupStatusFilter, setGroupStatusFilter] = useState<
    'all' | 'waiting_second' | 'disagree' | 'checked' | 'done'
  >('waiting_second');

  // Selected PR inspector state
  const [selectedPrNumber, setSelectedPrNumber] = useState<number | null>(null);
  const [selectedPrData, setSelectedPrData] = useState<GitHubItem | null>(null);
  const [loadingPr, setLoadingPr] = useState(false);
  const [prError, setPrError] = useState<string | null>(null);

  // Selected group for xcl-check drafting
  const [selectedGroup, setSelectedGroup] = useState<XclGroupUnit | null>(null);
  const [groupVerdict, setGroupVerdict] = useState('');
  const [groupHow, setGroupHow] = useState('read all diffs');
  const [groupHeads, setGroupHeads] = useState<Record<string, string>>({});
  const [loadingGroupHeads, setLoadingGroupHeads] = useState(false);

  // Test report builder state
  const [passedCount, setPassedCount] = useState('812');
  const [failedCount, setFailedCount] = useState('0');
  const [hardwareClass, setHardwareClass] = useState('nv-win');
  const [reportNotes, setReportNotes] = useState('');
  const [aiUsed, setAiUsed] = useState('none');
  const [copiedId, setCopiedId] = useState<'report' | 'check' | null>(null);

  // Auto-select first PR in "Needs 1 Report" when digest loads on desktop
  useEffect(() => {
    if (!digest) return;
    if (selectedPrNumber === null && !selectedGroup && digest.oneMoreReportPrs.length > 0) {
      setSelectedPrNumber(digest.oneMoreReportPrs[0]);
    }
  }, [digest, selectedPrNumber, selectedGroup]);

  // Fetch PR metadata + head SHA whenever selectedPrNumber changes
  useEffect(() => {
    if (selectedPrNumber === null) {
      setSelectedPrData(null);
      return;
    }

    const fromKnown = knownPulls.find((p) => p.number === selectedPrNumber);
    if (fromKnown && fromKnown.head_sha) {
      setSelectedPrData(fromKnown);
      setPrError(null);
      return;
    }

    let cancelled = false;
    setLoadingPr(true);
    setPrError(null);
    fetchPullByNumber('xcl-dev', 'xcl', selectedPrNumber)
      .then((item) => {
        if (!cancelled) setSelectedPrData(item);
      })
      .catch((err) => {
        if (!cancelled) {
          setSelectedPrData(fromKnown ?? null);
          setPrError(
            err instanceof Error
              ? err.message
              : `Could not fetch PR #${selectedPrNumber} head SHA from GitHub.`
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingPr(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedPrNumber, knownPulls]);

  const handleSelectPr = (prNum: number) => {
    setSelectedGroup(null);
    setSelectedPrNumber(prNum);
  };

  const handleSelectGroup = (group: XclGroupUnit) => {
    setSelectedPrNumber(null);
    setSelectedGroup(group);
    setGroupVerdict(
      group.kind === 'batch'
        ? 'pass'
        : group.prs.length >= 2
          ? `suggest ${group.prs[0]}; same change as ${group.prs.slice(1).join(' ')}`
          : ''
    );
    setGroupHow(
      group.kind === 'batch'
        ? 'merged together, cargo test --workspace, smoke base/batch'
        : 'read all diffs'
    );
    const initialHeads: Record<string, string> = {};
    for (const pr of group.prs) {
      const known = knownPulls.find((p) => p.number === pr);
      initialHeads[String(pr)] = known?.head_sha?.slice(0, 7) ?? '';
    }
    setGroupHeads(initialHeads);
  };

  const handleFetchGroupHeads = async () => {
    if (!selectedGroup) return;
    setLoadingGroupHeads(true);
    const nextHeads = { ...groupHeads };
    await Promise.all(
      selectedGroup.prs.map(async (pr) => {
        if (nextHeads[String(pr)]) return;
        try {
          const item = await fetchPullByNumber('xcl-dev', 'xcl', pr);
          if (item.head_sha) {
            nextHeads[String(pr)] = item.head_sha.slice(0, 7);
          }
        } catch {
          // Ignore individual failures
        }
      })
    );
    setGroupHeads(nextHeads);
    setLoadingGroupHeads(false);
  };

  const cleanQuery = search.trim().toLowerCase().replace(/^#/, '');

  const filteredOneMore = useMemo(() => {
    if (!digest) return [];
    if (!cleanQuery) return digest.oneMoreReportPrs;
    return digest.oneMoreReportPrs.filter((n) => String(n).includes(cleanQuery));
  }, [digest, cleanQuery]);

  const filteredZeroReport = useMemo(() => {
    if (!digest) return [];
    if (!cleanQuery) return digest.zeroReportPrs;
    return digest.zeroReportPrs.filter((n) => String(n).includes(cleanQuery));
  }, [digest, cleanQuery]);

  const filteredReadyRows = useMemo(() => {
    if (!digest) return [];
    if (!cleanQuery) return digest.readyToMerge;
    return digest.readyToMerge.filter(
      (r) =>
        String(r.number).includes(cleanQuery) ||
        r.title.toLowerCase().includes(cleanQuery) ||
        r.reporters.toLowerCase().includes(cleanQuery)
    );
  }, [digest, cleanQuery]);

  const activeGroupsCount = useMemo(() => {
    if (!digest) return 0;
    return digest.groups.filter((g) => g.status !== 'done').length;
  }, [digest]);

  const filteredGroups = useMemo(() => {
    if (!digest) return [];
    return digest.groups.filter((g) => {
      if (groupStatusFilter === 'all') {
        if (g.status === 'done') return false;
      } else if (g.status !== groupStatusFilter) {
        return false;
      }
      if (!cleanQuery) return true;
      return (
        g.unit.toLowerCase().includes(cleanQuery) ||
        g.summary.toLowerCase().includes(cleanQuery) ||
        g.checkedBy.toLowerCase().includes(cleanQuery) ||
        g.prs.some((p) => String(p).includes(cleanQuery)) ||
        (g.issueNumber !== null && String(g.issueNumber).includes(cleanQuery))
      );
    });
  }, [digest, groupStatusFilter, cleanQuery]);

  const generatedReport = useMemo(() => {
    return buildXclTestReportComment({
      headSha: selectedPrData?.head_sha ?? '',
      passedCount: Number(passedCount) || 0,
      failedCount: Number(failedCount) || 0,
      hardwareClass,
      notes: reportNotes,
      aiUsed,
    });
  }, [selectedPrData, passedCount, failedCount, hardwareClass, reportNotes, aiUsed]);

  const generatedCheckBlock = useMemo(() => {
    if (!selectedGroup) return '';
    return buildXclCheckBlock({
      unit: selectedGroup.unit,
      verdict: groupVerdict,
      heads: groupHeads,
      how: groupHow,
      hardwareClass,
      aiUsed,
    });
  }, [selectedGroup, groupVerdict, groupHeads, groupHow, hardwareClass, aiUsed]);

  const copyText = (text: string, which: 'report' | 'check') => {
    navigator.clipboard?.writeText(text).catch(() => {});
    setCopiedId(which);
    window.setTimeout(() => setCopiedId(null), 1800);
  };

  if (isLoading && !digest) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-8 text-center dark:border-slate-800 dark:bg-slate-950">
        <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
          Loading xcl-dev/xcl#1607 PR triage board...
        </p>
      </div>
    );
  }

  if (!digest) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-8 text-center dark:border-slate-800 dark:bg-slate-950">
        <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Could not load #1607 triage digest
        </p>
        <button
          type="button"
          onClick={onRefresh}
          className="mt-3 rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
        >
          Retry
        </button>
      </div>
    );
  }

  const selectedPrGroups =
    selectedPrNumber !== null ? digest.prToGroups[selectedPrNumber] ?? [] : [];

  return (
    <div className="space-y-5">
      {/* Header banner for #1607 board */}
      <section
        aria-label="XCL 1607 triage header"
        className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between dark:border-slate-800 dark:bg-slate-950"
      >
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-base font-bold text-slate-900 dark:text-slate-50">
              xcl-dev/xcl PR Triage Board (#1607)
            </h1>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              {digest.updatedLabel} · {source === 'live' ? 'Live' : 'Cached'}
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
            Tracking {digest.totalOpenPrs.toLocaleString('en-US')} open PRs ({digest.readyPrs} ready for test verification). Every report must name the PR&apos;s current 7-character head commit SHA.
          </p>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            onClick={onRefresh}
            disabled={isLoading}
            className="inline-flex h-9 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg border border-slate-300 bg-white px-3 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} aria-hidden="true" />
            <span>Refresh #1607</span>
          </button>

          <a
            href="https://github.com/xcl-dev/xcl/issues/1607"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex h-9 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg bg-slate-900 px-3.5 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-blue-600 dark:hover:bg-blue-500"
          >
            <span>Open #1607</span>
            <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
          </a>
        </div>
      </section>

      {/* 4-card actionable #1607 bucket strip */}
      <section
        aria-label="PR triage bucket counts"
        className="grid grid-cols-2 divide-x divide-y divide-slate-200 overflow-hidden rounded-xl border border-slate-200 bg-white sm:grid-cols-4 sm:divide-y-0 dark:divide-slate-800 dark:border-slate-800 dark:bg-slate-950"
      >
        <button
          type="button"
          onClick={() => setSubTab('needs_one')}
          className={`flex flex-col gap-0.5 px-4 py-3 text-left transition-colors ${
            subTab === 'needs_one'
              ? 'bg-blue-50/70 dark:bg-blue-950/30'
              : 'hover:bg-slate-50 dark:hover:bg-slate-900/50'
          }`}
        >
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Needs 1 More Report
          </span>
          <span className="font-mono text-lg font-bold text-blue-600 tabular-nums dark:text-blue-400">
            {digest.oneReport}
          </span>
        </button>

        <button
          type="button"
          onClick={() => setSubTab('needs_first')}
          className={`flex flex-col gap-0.5 px-4 py-3 text-left transition-colors ${
            subTab === 'needs_first'
              ? 'bg-blue-50/70 dark:bg-blue-950/30'
              : 'hover:bg-slate-50 dark:hover:bg-slate-900/50'
          }`}
        >
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
            No Report on Current Head
          </span>
          <span className="font-mono text-lg font-bold text-amber-700 tabular-nums dark:text-amber-400">
            {digest.zeroReports}
          </span>
        </button>

        <button
          type="button"
          onClick={() => setSubTab('ready_merge')}
          className={`flex flex-col gap-0.5 px-4 py-3 text-left transition-colors ${
            subTab === 'ready_merge'
              ? 'bg-blue-50/70 dark:bg-blue-950/30'
              : 'hover:bg-slate-50 dark:hover:bg-slate-900/50'
          }`}
        >
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
            2+ Reports (Ready)
          </span>
          <span className="font-mono text-lg font-bold text-emerald-700 tabular-nums dark:text-emerald-400">
            {digest.twoPlusReports}
          </span>
        </button>

        <button
          type="button"
          onClick={() => setSubTab('groups')}
          className={`flex flex-col gap-0.5 px-4 py-3 text-left transition-colors ${
            subTab === 'groups'
              ? 'bg-blue-50/70 dark:bg-blue-950/30'
              : 'hover:bg-slate-50 dark:hover:bg-slate-900/50'
          }`}
        >
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Active Dup &amp; Batch Groups
          </span>
          <span className="font-mono text-lg font-bold text-slate-900 tabular-nums dark:text-slate-50">
            {activeGroupsCount}
          </span>
        </button>
      </section>

      {/* Master-Detail Split */}
      <div className="grid grid-cols-1 items-start gap-5 lg:grid-cols-12">
        {/* Left List */}
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-white lg:col-span-7 dark:border-slate-800 dark:bg-slate-950">
          <div className="flex flex-col gap-3 border-b border-slate-200 p-4 dark:border-slate-800">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div
                role="group"
                aria-label="XCL 1607 queues"
                className="inline-flex flex-wrap items-center gap-1 rounded-lg bg-slate-100 p-1 dark:bg-slate-900"
              >
                {(
                  [
                    { id: 'needs_one', label: 'Needs 1 Report' },
                    { id: 'needs_first', label: 'No Report Yet' },
                    { id: 'ready_merge', label: 'Ready (2+)' },
                    { id: 'groups', label: 'Dup / Batch Groups' },
                  ] as const
                ).map((t) => {
                  const active = subTab === t.id;
                  return (
                    <button
                      key={t.id}
                      type="button"
                      onClick={() => setSubTab(t.id)}
                      aria-pressed={active}
                      className={`shrink-0 whitespace-nowrap rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                        active
                          ? 'bg-white text-slate-900 shadow-xs dark:bg-slate-800 dark:text-slate-50'
                          : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100'
                      }`}
                    >
                      {t.label}
                    </button>
                  );
                })}
              </div>

              {subTab === 'groups' && (
                <select
                  aria-label="Filter groups by check status"
                  value={groupStatusFilter}
                  onChange={(e) =>
                    setGroupStatusFilter(
                      e.target.value as 'all' | 'waiting_second' | 'disagree' | 'checked' | 'done'
                    )
                  }
                  className="h-8 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200"
                >
                  <option value="waiting_second">Waiting for 2nd Check</option>
                  <option value="disagree">Checks Disagree</option>
                  <option value="checked">Checked by Two</option>
                  <option value="all">All Active Groups ({activeGroupsCount})</option>
                  <option value="done">Done / Retired</option>
                </select>
              )}
            </div>

            <div className="relative">
              <Search
                className="pointer-events-none absolute top-1/2 left-3 h-3.5 w-3.5 -translate-y-1/2 text-slate-400"
                aria-hidden="true"
              />
              <label htmlFor="xcl1607-search" className="sr-only">
                Filter #1607 items by PR number, group ID, or keyword
              </label>
              <input
                id="xcl1607-search"
                type="search"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Filter by PR #number, D/B unit ID, title, or contributor..."
                className="h-9 w-full rounded-lg border border-slate-200 bg-slate-50 pr-3 pl-9 text-xs text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:bg-white focus:outline-none dark:border-slate-800 dark:bg-slate-900 dark:text-slate-100"
              />
            </div>
          </div>

          {subTab === 'needs_one' && (
            <div className="p-4">
              <p className="mb-3 text-xs text-slate-600 dark:text-slate-400">
                Highest-leverage queue: each PR below already has 1 independent test report on its current head. Click any PR number to load its live head commit SHA and draft the 2nd report.
              </p>
              {filteredOneMore.length === 0 ? (
                <p className="py-6 text-center text-xs text-slate-500">
                  No PR numbers match &ldquo;{search}&rdquo;.
                </p>
              ) : (
                <div className="grid grid-cols-3 gap-2 sm:grid-cols-5">
                  {filteredOneMore.map((prNum) => {
                    const active = selectedPrNumber === prNum;
                    const inDup = (digest.prToGroups[prNum] ?? []).length > 0;
                    return (
                      <button
                        key={prNum}
                        type="button"
                        onClick={() => handleSelectPr(prNum)}
                        className={`flex items-center justify-between rounded-lg border px-3 py-2 font-mono text-xs font-semibold tabular-nums transition-colors ${
                          active
                            ? 'border-blue-600 bg-blue-50 text-blue-800 dark:border-blue-500 dark:bg-blue-950/50 dark:text-blue-200'
                            : 'border-slate-200 bg-white text-slate-800 hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800'
                        }`}
                      >
                        <span>#{prNum}</span>
                        {inDup && (
                          <span
                            title={`In group ${digest.prToGroups[prNum].join(', ')}`}
                            className="text-[10px] font-medium text-amber-600 dark:text-amber-400"
                          >
                            {digest.prToGroups[prNum][0]}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {subTab === 'needs_first' && (
            <div className="p-4">
              <p className="mb-3 text-xs text-slate-600 dark:text-slate-400">
                Oldest ready PRs with 0 test reports on their current head ({digest.earlierHeadReports} of {digest.zeroReports} total have reports on an earlier head).
              </p>
              {filteredZeroReport.length === 0 ? (
                <p className="py-6 text-center text-xs text-slate-500">
                  No PR numbers match &ldquo;{search}&rdquo;.
                </p>
              ) : (
                <div className="grid grid-cols-3 gap-2 sm:grid-cols-5">
                  {filteredZeroReport.map((prNum) => {
                    const active = selectedPrNumber === prNum;
                    return (
                      <button
                        key={prNum}
                        type="button"
                        onClick={() => handleSelectPr(prNum)}
                        className={`rounded-lg border px-3 py-2 text-left font-mono text-xs font-semibold tabular-nums transition-colors ${
                          active
                            ? 'border-blue-600 bg-blue-50 text-blue-800 dark:border-blue-500 dark:bg-blue-950/50 dark:text-blue-200'
                            : 'border-slate-200 bg-white text-slate-800 hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800'
                        }`}
                      >
                        #{prNum}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {subTab === 'ready_merge' && (
            <div className="divide-y divide-slate-200 dark:divide-slate-800">
              {filteredReadyRows.length === 0 ? (
                <p className="px-4 py-8 text-center text-xs text-slate-500">
                  No ready PRs match &ldquo;{search}&rdquo;.
                </p>
              ) : (
                filteredReadyRows.map((row) => {
                  const active = selectedPrNumber === row.number;
                  return (
                    <button
                      key={row.number}
                      type="button"
                      onClick={() => handleSelectPr(row.number)}
                      className={`flex w-full flex-col gap-1 px-4 py-3 text-left transition-colors ${
                        active
                          ? 'bg-blue-50/70 dark:bg-blue-950/30'
                          : 'hover:bg-slate-50 dark:hover:bg-slate-900/60'
                      }`}
                    >
                      <div className="flex items-baseline justify-between gap-2">
                        <div className="flex items-baseline gap-2">
                          <span className="font-mono text-xs font-semibold text-slate-500 tabular-nums dark:text-slate-400">
                            #{row.number}
                          </span>
                          <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                            {row.title}
                          </span>
                        </div>
                        <span className="shrink-0 font-mono text-xs font-semibold text-emerald-700 tabular-nums dark:text-emerald-400">
                          {row.reportsCount} reports
                        </span>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                        <span>{row.linkedIssue ? 'Linked issue' : 'No linked issue'}</span>
                        <span aria-hidden="true">·</span>
                        <span>Tested by: {row.reporters}</span>
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          )}

          {subTab === 'groups' && (
            <div className="divide-y divide-slate-200 dark:divide-slate-800">
              {filteredGroups.length === 0 ? (
                <p className="px-4 py-8 text-center text-xs text-slate-500">
                  No duplicate or batch groups match the current filter.
                </p>
              ) : (
                filteredGroups.map((group) => {
                  const active = selectedGroup?.unit === group.unit;
                  const statusLabel =
                    group.status === 'waiting_second'
                      ? 'Needs 2nd check'
                      : group.status === 'disagree'
                        ? 'Checks disagree'
                        : group.status === 'done'
                          ? 'Done / retired'
                          : 'Checked by 2';
                  return (
                    <div
                      key={group.unit}
                      className={`flex flex-col gap-2 px-4 py-3 ${
                        active ? 'bg-blue-50/70 dark:bg-blue-950/30' : ''
                      }`}
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex flex-wrap items-center gap-2 text-xs">
                          <span className="font-mono font-bold text-slate-900 dark:text-slate-100">
                            {group.unit}
                          </span>
                          <span aria-hidden="true">·</span>
                          <span className="text-slate-600 dark:text-slate-300">
                            {group.kind === 'batch' ? 'Batch' : 'Duplicate group'}
                          </span>
                          {group.baseSha && (
                            <>
                              <span aria-hidden="true">·</span>
                              <span className="font-mono text-slate-500 dark:text-slate-400">
                                base {group.baseSha}
                              </span>
                            </>
                          )}
                          {group.issueNumber && (
                            <>
                              <span aria-hidden="true">·</span>
                              <span className="font-mono text-slate-600 dark:text-slate-400">
                                Issue #{group.issueNumber}
                              </span>
                            </>
                          )}
                          <span aria-hidden="true">·</span>
                          <span
                            className={`font-medium ${
                              group.status === 'waiting_second'
                                ? 'text-amber-700 dark:text-amber-400'
                                : group.status === 'disagree'
                                  ? 'text-rose-700 dark:text-rose-400'
                                  : group.status === 'done'
                                    ? 'text-slate-500 dark:text-slate-400'
                                    : 'text-emerald-700 dark:text-emerald-400'
                            }`}
                          >
                            {statusLabel}
                          </span>
                        </div>

                        <button
                          type="button"
                          onClick={() => handleSelectGroup(group)}
                          className="shrink-0 whitespace-nowrap rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-semibold text-slate-800 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
                        >
                          Draft xcl-check
                        </button>
                      </div>

                      <div className="flex flex-wrap items-center gap-1.5">
                        {group.prs.map((pr) => (
                          <button
                            key={pr}
                            type="button"
                            onClick={() => handleSelectPr(pr)}
                            className="rounded border border-slate-200 bg-slate-50 px-2 py-0.5 font-mono text-xs font-medium text-blue-700 hover:bg-blue-50 dark:border-slate-800 dark:bg-slate-900 dark:text-blue-400 dark:hover:bg-slate-800"
                          >
                            #{pr}
                          </button>
                        ))}
                      </div>

                      {group.summary && (
                        <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
                          {group.summary}
                        </p>
                      )}

                      {group.checkedBy && (
                        <p className="text-[11px] text-slate-500 dark:text-slate-400">
                          {group.status === 'waiting_second' ? '1st check: ' : 'Checked by: '}
                          {group.checkedBy}
                        </p>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>

        {/* Right Inspector & Report / xcl-check Builder */}
        <aside
          aria-label="XCL PR inspection and report generator"
          className="rounded-xl border border-slate-200 bg-white p-4 sm:p-5 lg:sticky lg:top-4 lg:col-span-5 dark:border-slate-800 dark:bg-slate-950"
        >
          {selectedGroup ? (
            <div className="space-y-4">
              <div className="flex items-start justify-between gap-3 border-b border-slate-200 pb-3 dark:border-slate-800">
                <div>
                  <span className="font-mono text-xs font-bold text-amber-700 dark:text-amber-400">
                    Unit {selectedGroup.unit} · {selectedGroup.kind === 'batch' ? 'Batch' : 'Duplicate'}
                  </span>
                  <h2 className="mt-1 text-base font-bold text-slate-900 dark:text-slate-50">
                    Draft <code className="font-mono">xcl-check</code> Verification
                  </h2>
                </div>
                <button
                  type="button"
                  onClick={handleFetchGroupHeads}
                  disabled={loadingGroupHeads}
                  className="shrink-0 whitespace-nowrap rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                >
                  {loadingGroupHeads ? 'Fetching SHAs...' : 'Fetch Head SHAs'}
                </button>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <label
                    htmlFor="xcl-check-verdict"
                    className="mb-1 block font-semibold text-slate-700 dark:text-slate-300"
                  >
                    Verdict
                  </label>
                  <input
                    id="xcl-check-verdict"
                    type="text"
                    value={groupVerdict}
                    onChange={(e) => setGroupVerdict(e.target.value)}
                    placeholder="suggest 1306; same change as 1309"
                    className="h-9 w-full rounded-lg border border-slate-300 bg-white px-3 text-xs text-slate-900 focus:border-blue-600 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                  />
                </div>

                <div>
                  <label
                    htmlFor="xcl-check-how"
                    className="mb-1 block font-semibold text-slate-700 dark:text-slate-300"
                  >
                    How Checked
                  </label>
                  <input
                    id="xcl-check-how"
                    type="text"
                    value={groupHow}
                    onChange={(e) => setGroupHow(e.target.value)}
                    className="h-9 w-full rounded-lg border border-slate-300 bg-white px-3 text-xs text-slate-900 focus:border-blue-600 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                  />
                </div>

                <div>
                  <span className="mb-1 block font-semibold text-slate-700 dark:text-slate-300">
                    Head SHAs (7 chars per PR)
                  </span>
                  <div className="grid grid-cols-2 gap-2">
                    {selectedGroup.prs.map((pr) => (
                      <div key={pr} className="flex items-center gap-2">
                        <span className="w-14 font-mono font-semibold text-slate-600 dark:text-slate-400">
                          #{pr}
                        </span>
                        <input
                          type="text"
                          aria-label={`Head SHA for PR #${pr}`}
                          value={groupHeads[String(pr)] ?? ''}
                          onChange={(e) =>
                            setGroupHeads((prev) => ({ ...prev, [String(pr)]: e.target.value }))
                          }
                          placeholder="7-char SHA"
                          className="h-8 w-full rounded-md border border-slate-300 bg-white px-2 font-mono text-xs text-slate-900 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                        />
                      </div>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2.5">
                  <div>
                    <label
                      htmlFor="xcl-check-hw"
                      className="mb-1 block font-semibold text-slate-700 dark:text-slate-300"
                    >
                      Hardware Class
                    </label>
                    <select
                      id="xcl-check-hw"
                      value={hardwareClass}
                      onChange={(e) => setHardwareClass(e.target.value)}
                      className="h-9 w-full rounded-lg border border-slate-300 bg-white px-2.5 text-xs text-slate-900 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                    >
                      {HARDWARE_CLASSES.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label
                      htmlFor="xcl-check-ai"
                      className="mb-1 block font-semibold text-slate-700 dark:text-slate-300"
                    >
                      AI Disclosure
                    </label>
                    <input
                      id="xcl-check-ai"
                      type="text"
                      value={aiUsed}
                      onChange={(e) => setAiUsed(e.target.value)}
                      placeholder="none"
                      className="h-9 w-full rounded-lg border border-slate-300 bg-white px-3 text-xs text-slate-900 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                    />
                  </div>
                </div>

                <pre className="whitespace-pre-wrap break-all rounded-lg border border-slate-200 bg-slate-50 p-3 font-mono text-[11px] leading-relaxed text-slate-800 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200">
                  <code>{generatedCheckBlock}</code>
                </pre>

                <div className="flex flex-wrap items-center justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => copyText(generatedCheckBlock, 'check')}
                    className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 text-xs font-semibold text-slate-800 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                  >
                    {copiedId === 'check' ? (
                      <Check className="h-3.5 w-3.5 text-emerald-600" aria-hidden="true" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" aria-hidden="true" />
                    )}
                    <span>{copiedId === 'check' ? 'Copied' : 'Copy xcl-check'}</span>
                  </button>
                  <a
                    href="https://github.com/xcl-dev/xcl/issues/1607#issuecomment-new"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-slate-900 px-3.5 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-blue-600 dark:hover:bg-blue-500"
                  >
                    <span>Post on #1607</span>
                    <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                  </a>
                </div>
              </div>
            </div>
          ) : selectedPrNumber !== null ? (
            <div className="space-y-4">
              <div className="border-b border-slate-200 pb-3 dark:border-slate-800">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-xs font-bold text-slate-900 tabular-nums dark:text-slate-100">
                    PR #{selectedPrNumber}
                  </span>
                  <a
                    href={`https://github.com/xcl-dev/xcl/pull/${selectedPrNumber}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-xs font-semibold text-blue-600 hover:underline dark:text-blue-400"
                  >
                    <span>View PR on GitHub</span>
                    <ExternalLink className="h-3 w-3" aria-hidden="true" />
                  </a>
                </div>

                {loadingPr ? (
                  <p className="mt-2 text-xs text-slate-500">Fetching live PR head commit...</p>
                ) : selectedPrData ? (
                  <div className="mt-2 space-y-1.5">
                    <h2 className="text-sm font-bold leading-snug text-slate-900 dark:text-slate-50">
                      {selectedPrData.title}
                    </h2>
                    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                      <span className="inline-flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300">
                        <Avatar
                          src={selectedPrData.user.avatar_url}
                          name={selectedPrData.user.login}
                          size="sm"
                        />
                        <span>{selectedPrData.user.login}</span>
                      </span>
                      {selectedPrData.head_ref && (
                        <>
                          <span aria-hidden="true">·</span>
                          <span className="inline-flex items-center gap-1 font-mono text-[11px]">
                            <GitBranch className="h-3 w-3" aria-hidden="true" />
                            {selectedPrData.head_ref}
                          </span>
                        </>
                      )}
                      {selectedPrData.head_sha && (
                        <>
                          <span aria-hidden="true">·</span>
                          <span className="font-mono text-[11px] font-semibold text-slate-800 dark:text-slate-200">
                            head {selectedPrData.head_sha.slice(0, 7)}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                ) : (
                  <p className="mt-2 text-xs text-slate-500">
                    {prError ?? 'Click View PR on GitHub to inspect files and head SHA.'}
                  </p>
                )}

                {selectedPrGroups.length > 0 && (
                  <p className="mt-2 text-xs font-medium text-amber-700 dark:text-amber-400">
                    Heads up: #{selectedPrNumber} appears in #1607 group{' '}
                    <span className="font-mono">{selectedPrGroups.join(', ')}</span> — check for duplicates before testing.
                  </p>
                )}
              </div>

              {/* Test Report Builder */}
              <div className="space-y-3 text-xs">
                <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                  Draft Counted Test Report (#1607 Format)
                </h3>

                <div className="grid grid-cols-2 gap-2.5">
                  <div>
                    <label
                      htmlFor="xcl-passed-count"
                      className="mb-1 block font-medium text-slate-700 dark:text-slate-300"
                    >
                      Tests Passed
                    </label>
                    <input
                      id="xcl-passed-count"
                      type="number"
                      min="0"
                      value={passedCount}
                      onChange={(e) => setPassedCount(e.target.value)}
                      className="h-9 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono text-xs text-slate-900 tabular-nums dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                    />
                  </div>
                  <div>
                    <label
                      htmlFor="xcl-failed-count"
                      className="mb-1 block font-medium text-slate-700 dark:text-slate-300"
                    >
                      Tests Failed
                    </label>
                    <input
                      id="xcl-failed-count"
                      type="number"
                      min="0"
                      value={failedCount}
                      onChange={(e) => setFailedCount(e.target.value)}
                      className="h-9 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono text-xs text-slate-900 tabular-nums dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2.5">
                  <div>
                    <label
                      htmlFor="xcl-hw-class"
                      className="mb-1 block font-medium text-slate-700 dark:text-slate-300"
                    >
                      Hardware Class
                    </label>
                    <select
                      id="xcl-hw-class"
                      value={hardwareClass}
                      onChange={(e) => setHardwareClass(e.target.value)}
                      className="h-9 w-full rounded-lg border border-slate-300 bg-white px-2.5 text-xs text-slate-900 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                    >
                      {HARDWARE_CLASSES.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label
                      htmlFor="xcl-ai-used"
                      className="mb-1 block font-medium text-slate-700 dark:text-slate-300"
                    >
                      AI Disclosure
                    </label>
                    <input
                      id="xcl-ai-used"
                      type="text"
                      value={aiUsed}
                      onChange={(e) => setAiUsed(e.target.value)}
                      placeholder="none"
                      className="h-9 w-full rounded-lg border border-slate-300 bg-white px-3 text-xs text-slate-900 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                    />
                  </div>
                </div>

                <div>
                  <label
                    htmlFor="xcl-report-notes"
                    className="mb-1 block font-medium text-slate-700 dark:text-slate-300"
                  >
                    Optional Smoke / Notes
                  </label>
                  <input
                    id="xcl-report-notes"
                    type="text"
                    value={reportNotes}
                    onChange={(e) => setReportNotes(e.target.value)}
                    placeholder="e.g. smoke pair 1226/1240 frames, no regressions"
                    className="h-9 w-full rounded-lg border border-slate-300 bg-white px-3 text-xs text-slate-900 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                  />
                </div>

                <pre className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-50 p-3 font-mono text-[11px] leading-relaxed text-slate-800 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200">
                  <code>{generatedReport}</code>
                </pre>

                <div className="flex flex-wrap items-center justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => copyText(generatedReport, 'report')}
                    className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 text-xs font-semibold text-slate-800 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                  >
                    {copiedId === 'report' ? (
                      <Check className="h-3.5 w-3.5 text-emerald-600" aria-hidden="true" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" aria-hidden="true" />
                    )}
                    <span>{copiedId === 'report' ? 'Copied' : 'Copy Report'}</span>
                  </button>

                  <a
                    href={`https://github.com/xcl-dev/xcl/pull/${selectedPrNumber}#issuecomment-new`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-slate-900 px-3.5 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-blue-600 dark:hover:bg-blue-500"
                  >
                    <span>Post on PR #{selectedPrNumber}</span>
                    <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                  </a>
                </div>
              </div>
            </div>
          ) : (
            <div className="py-12 text-center">
              <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Select a PR or Group
              </p>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                Click any PR number or click Draft xcl-check on a duplicate/batch group to generate a #1607-compliant verification comment.
              </p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
