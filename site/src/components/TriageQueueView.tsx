import { useMemo, useState } from 'react';
import { Check, Copy, ExternalLink, Pin, Trash2 } from 'lucide-react';
import type { TriageRecord, TriageStatus } from '../types/github';
import { formatRelativeTime } from '../utils/githubParse';
import { TRIAGE_STATUS_META } from '../utils/triageStore';

interface TriageQueueViewProps {
  repoSlug: string;
  records: TriageRecord[];
  onChangeStatus: (record: TriageRecord, status: TriageStatus | null) => void;
  onTogglePin: (record: TriageRecord) => void;
  onRemoveRecord: (record: TriageRecord) => void;
  onOpenItem: (record: TriageRecord) => void;
}

export function TriageQueueView({
  repoSlug,
  records,
  onChangeStatus,
  onTogglePin,
  onRemoveRecord,
  onOpenItem,
}: TriageQueueViewProps) {
  const [statusFilter, setStatusFilter] = useState<'all' | 'pinned' | TriageStatus>('all');
  const [onlyCurrentRepo, setOnlyCurrentRepo] = useState(true);
  const [copiedSummary, setCopiedSummary] = useState(false);

  const visibleRecords = useMemo(() => {
    return records
      .filter((r) => {
        if (onlyCurrentRepo && r.repoSlug.toLowerCase() !== repoSlug.toLowerCase()) {
          return false;
        }
        if (statusFilter === 'pinned') return r.pinned;
        if (statusFilter !== 'all') return r.status === statusFilter;
        return true;
      })
      .sort((a, b) => {
        if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
        return Date.parse(b.updatedAt) - Date.parse(a.updatedAt);
      });
  }, [records, onlyCurrentRepo, repoSlug, statusFilter]);

  const handleCopyMarkdown = () => {
    const lines = [
      `### Triage Summary (${onlyCurrentRepo ? repoSlug : 'All Repositories'})`,
      ...visibleRecords.map((r) => {
        const statusLabel = r.status ? TRIAGE_STATUS_META[r.status].label : 'Pinned';
        const notePart = r.note.trim() ? ` — ${r.note.trim()}` : '';
        return `- [${statusLabel}] ${r.repoSlug}#${r.number}: ${r.title}${notePart} (${r.html_url})`;
      }),
    ];
    navigator.clipboard?.writeText(lines.join('\n')).catch(() => {});
    setCopiedSummary(true);
    window.setTimeout(() => setCopiedSummary(false), 1800);
  };

  return (
    <section
      aria-label="Personal triage queue"
      className="overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950"
    >
      <div className="flex flex-col gap-3 border-b border-slate-200 p-4 sm:flex-row sm:items-center sm:justify-between dark:border-slate-800">
        <div>
          <h1 className="text-base font-bold text-slate-900 dark:text-slate-50">
            Personal Triage Queue
          </h1>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
            Saved locally in your browser so fast-moving issues and PRs never get lost.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select
            aria-label="Filter triage queue by status"
            value={statusFilter}
            onChange={(e) =>
              setStatusFilter(e.target.value as 'all' | 'pinned' | TriageStatus)
            }
            className="h-9 rounded-lg border border-slate-200 bg-white px-3 text-xs font-medium text-slate-700 focus:border-blue-600 focus:outline-none dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200"
          >
            <option value="all">All Saved ({records.length})</option>
            <option value="pinned">Pinned Only</option>
            <option value="needs_review">Needs Review</option>
            <option value="in_progress">In Progress</option>
            <option value="ready_to_merge">Ready to Merge</option>
            <option value="blocked">Blocked</option>
            <option value="snoozed">Snoozed</option>
          </select>

          <button
            type="button"
            onClick={() => setOnlyCurrentRepo((prev) => !prev)}
            aria-pressed={onlyCurrentRepo}
            className={`h-9 shrink-0 whitespace-nowrap rounded-lg border px-3 text-xs font-medium ${
              onlyCurrentRepo
                ? 'border-blue-600 bg-blue-50 text-blue-800 dark:border-blue-500 dark:bg-blue-950/50 dark:text-blue-200'
                : 'border-slate-200 bg-white text-slate-700 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300'
            }`}
          >
            {onlyCurrentRepo ? repoSlug : 'All Repos'}
          </button>

          <button
            type="button"
            disabled={visibleRecords.length === 0}
            onClick={handleCopyMarkdown}
            className="inline-flex h-9 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg border border-slate-300 bg-white px-3 text-xs font-semibold text-slate-800 hover:bg-slate-50 disabled:opacity-40 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
          >
            {copiedSummary ? (
              <Check className="h-3.5 w-3.5 text-emerald-600" aria-hidden="true" />
            ) : (
              <Copy className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            <span>{copiedSummary ? 'Copied Markdown' : 'Export Markdown'}</span>
          </button>
        </div>
      </div>

      {visibleRecords.length === 0 ? (
        <div className="px-6 py-14 text-center">
          <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Your triage queue is empty
          </p>
          <p className="mx-auto mt-1 max-w-md text-xs text-slate-500 dark:text-slate-400">
            Open any issue or pull request and assign a triage status, pin it, or add a note to track it here.
          </p>
        </div>
      ) : (
        <div className="divide-y divide-slate-200 dark:divide-slate-800">
          {visibleRecords.map((rec) => {
            const meta = rec.status ? TRIAGE_STATUS_META[rec.status] : null;
            return (
              <div
                key={rec.key}
                className="flex flex-col gap-3 px-4 py-3.5 sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline gap-2">
                    <span className="shrink-0 font-mono text-xs font-semibold text-slate-500 tabular-nums dark:text-slate-400">
                      {rec.repoSlug}#{rec.number}
                    </span>
                    <button
                      type="button"
                      onClick={() => onOpenItem(rec)}
                      className="text-left text-sm font-semibold text-slate-900 hover:text-blue-600 dark:text-slate-100 dark:hover:text-blue-400"
                    >
                      {rec.title}
                    </button>
                  </div>

                  <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                    <span>{rec.kind === 'pr' ? 'Pull Request' : 'Issue'}</span>
                    {meta && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className={`font-semibold ${meta.textClass}`}>{meta.label}</span>
                      </>
                    )}
                    <span aria-hidden="true">·</span>
                    <span>updated {formatRelativeTime(rec.updatedAt)}</span>
                  </div>

                  {rec.note.trim() && (
                    <p className="mt-1.5 text-xs text-slate-700 dark:text-slate-300">
                      Note: {rec.note}
                    </p>
                  )}
                </div>

                <div className="flex shrink-0 items-center gap-2">
                  <select
                    aria-label={`Triage status for #${rec.number}`}
                    value={rec.status ?? ''}
                    onChange={(e) =>
                      onChangeStatus(rec, (e.target.value || null) as TriageStatus | null)
                    }
                    className="h-8 rounded-md border border-slate-200 bg-white px-2 text-xs font-medium text-slate-700 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200"
                  >
                    <option value="">No status</option>
                    <option value="needs_review">Needs Review</option>
                    <option value="in_progress">In Progress</option>
                    <option value="ready_to_merge">Ready to Merge</option>
                    <option value="blocked">Blocked</option>
                    <option value="snoozed">Snoozed</option>
                  </select>

                  <button
                    type="button"
                    onClick={() => onTogglePin(rec)}
                    aria-label={rec.pinned ? `Unpin #${rec.number}` : `Pin #${rec.number}`}
                    aria-pressed={rec.pinned}
                    className={`inline-flex h-8 w-8 items-center justify-center rounded-md border ${
                      rec.pinned
                        ? 'border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-700 dark:bg-amber-950/50 dark:text-amber-300'
                        : 'border-slate-200 bg-white text-slate-500 hover:text-slate-900 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-400'
                    }`}
                  >
                    <Pin className="h-3.5 w-3.5" aria-hidden="true" />
                  </button>

                  <a
                    href={rec.html_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={`Open #${rec.number} on GitHub`}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-600 hover:text-slate-900 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300"
                  >
                    <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                  </a>

                  <button
                    type="button"
                    onClick={() => onRemoveRecord(rec)}
                    aria-label={`Remove #${rec.number} from triage queue`}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-500 hover:border-rose-300 hover:text-rose-600 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-400 dark:hover:text-rose-400"
                  >
                    <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
