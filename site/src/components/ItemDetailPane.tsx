import { useEffect, useState } from 'react';
import {
  ArrowLeft,
  Check,
  Copy,
  ExternalLink,
  GitBranch,
  MessageSquare,
  Pin,
} from 'lucide-react';
import { QUICK_REPLY_SNIPPETS } from '../data/seedData';
import type {
  GitHubComment,
  GitHubItem,
  TriageRecord,
  TriageStatus,
} from '../types/github';
import { fetchItemComments, fetchPullDetail } from '../utils/githubApi';
import { formatRelativeTime } from '../utils/githubParse';
import { Avatar } from './Avatar';
import { MarkdownLite } from './MarkdownLite';

interface ItemDetailPaneProps {
  owner: string;
  repo: string;
  item: GitHubItem | null;
  triageRecord?: TriageRecord;
  xclGroupIds?: string[];
  onUpdateTriage: (
    item: GitHubItem,
    patch: Partial<Pick<TriageRecord, 'status' | 'pinned' | 'note'>>
  ) => void;
  onCloseMobile: () => void;
}

export function ItemDetailPane({
  owner,
  repo,
  item,
  triageRecord,
  xclGroupIds,
  onUpdateTriage,
  onCloseMobile,
}: ItemDetailPaneProps) {
  const [comments, setComments] = useState<GitHubComment[]>([]);
  const [loadingComments, setLoadingComments] = useState(false);
  const [prStats, setPrStats] = useState<{
    additions?: number;
    deletions?: number;
    changed_files?: number;
    head_sha?: string;
  }>({});
  const [replyDraft, setReplyDraft] = useState('');
  const [copiedKey, setCopiedKey] = useState<'link' | 'reply' | null>(null);

  useEffect(() => {
    if (!item) return;
    let cancelled = false;
    setComments([]);
    setPrStats({
      additions: item.additions,
      deletions: item.deletions,
      changed_files: item.changed_files,
      head_sha: item.head_sha,
    });
    setReplyDraft('');

    if (item.comments > 0) {
      setLoadingComments(true);
      fetchItemComments(owner, repo, item.number)
        .then((res) => {
          if (!cancelled) setComments(res);
        })
        .catch(() => {
          // Graceful no-op when offline or rate-limited
        })
        .finally(() => {
          if (!cancelled) setLoadingComments(false);
        });
    }

    if (item.kind === 'pr' && item.additions === undefined) {
      fetchPullDetail(owner, repo, item.number)
        .then((detail) => {
          if (!cancelled) {
            setPrStats({
              additions: detail.additions,
              deletions: detail.deletions,
              changed_files: detail.changed_files,
              head_sha: detail.head_sha ?? item.head_sha,
            });
          }
        })
        .catch(() => {
          // Graceful no-op
        });
    }

    return () => {
      cancelled = true;
    };
  }, [owner, repo, item]);

  const copyText = (text: string, which: 'link' | 'reply') => {
    navigator.clipboard?.writeText(text).catch(() => {});
    setCopiedKey(which);
    window.setTimeout(() => setCopiedKey(null), 1800);
  };

  if (!item) {
    return (
      <aside
        aria-label="Issue and pull request details"
        className="hidden h-full min-h-[420px] flex-col items-center justify-center rounded-xl border border-slate-200 bg-white p-8 text-center lg:flex dark:border-slate-800 dark:bg-slate-950"
      >
        <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          Select an issue or pull request
        </p>
        <p className="mt-1 max-w-xs text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          Inspect full context, branch targets, comments, and save local triage statuses and notes.
        </p>
      </aside>
    );
  }

  const stateText =
    item.kind === 'pr'
      ? item.merged_at
        ? 'Merged'
        : item.state === 'closed'
          ? 'Closed'
          : item.draft
            ? 'Draft'
            : 'Open'
      : item.state === 'closed'
        ? 'Closed'
        : 'Open';

  return (
    <aside
      aria-label={`Details for #${item.number}`}
      className="flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950"
    >
      {/* Top detail header */}
      <div className="border-b border-slate-200 p-4 sm:p-5 dark:border-slate-800">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2.5">
            <button
              type="button"
              onClick={onCloseMobile}
              className="inline-flex shrink-0 items-center gap-1 whitespace-nowrap text-xs font-semibold text-slate-600 hover:text-slate-900 lg:hidden dark:text-slate-300 dark:hover:text-white"
            >
              <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
              <span>Back to list</span>
            </button>

            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
              <span className="font-mono font-semibold text-slate-900 tabular-nums dark:text-slate-100">
                #{item.number}
              </span>
              <span aria-hidden="true">·</span>
              <span className="font-medium text-slate-700 dark:text-slate-300">{stateText}</span>
              <span aria-hidden="true">·</span>
              <span className="whitespace-nowrap">{item.kind === 'pr' ? 'Pull Request' : 'Issue'}</span>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={() => copyText(item.html_url, 'link')}
              className="inline-flex h-8 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              {copiedKey === 'link' ? (
                <Check className="h-3.5 w-3.5 text-emerald-600" aria-hidden="true" />
              ) : (
                <Copy className="h-3.5 w-3.5" aria-hidden="true" />
              )}
              <span>{copiedKey === 'link' ? 'Copied' : 'Copy URL'}</span>
            </button>

            <a
              href={item.html_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-8 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-md bg-slate-900 px-3 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-blue-600 dark:hover:bg-blue-500"
            >
              <span>Open on GitHub</span>
              <ExternalLink className="h-3 w-3" aria-hidden="true" />
            </a>
          </div>
        </div>

        <h2 className="text-base font-bold leading-snug text-slate-900 sm:text-lg dark:text-slate-50">
          {item.title}
        </h2>

        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
          <span className="inline-flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300">
            <Avatar src={item.user.avatar_url} name={item.user.login} size="sm" />
            <span>{item.user.login}</span>
          </span>
          <span aria-hidden="true">·</span>
          <span>opened {formatRelativeTime(item.created_at)}</span>
          <span aria-hidden="true">·</span>
          <span>updated {formatRelativeTime(item.updated_at)}</span>
        </div>

        {item.kind === 'pr' && item.head_ref && (
          <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-600 dark:text-slate-300">
            <span className="inline-flex items-center gap-1.5 font-mono">
              <GitBranch className="h-3.5 w-3.5 text-slate-400" aria-hidden="true" />
              <span>{item.head_ref}</span>
              <span aria-hidden="true">→</span>
              <span>{item.base_ref ?? 'main'}</span>
            </span>
            {prStats.head_sha && (
              <span className="font-mono text-[11px] text-slate-500 tabular-nums dark:text-slate-400">
                head {prStats.head_sha.slice(0, 7)}
              </span>
            )}
            {typeof prStats.additions === 'number' && typeof prStats.deletions === 'number' && (
              <span className="font-mono tabular-nums">
                <span className="text-emerald-700 dark:text-emerald-400">
                  +{prStats.additions}
                </span>{' '}
                /{' '}
                <span className="text-rose-700 dark:text-rose-400">-{prStats.deletions}</span>
                {typeof prStats.changed_files === 'number' && (
                  <span className="text-slate-500 dark:text-slate-400">
                    {' '}
                    · {prStats.changed_files} files
                  </span>
                )}
              </span>
            )}
          </div>
        )}

        {xclGroupIds && xclGroupIds.length > 0 && (
          <p className="mt-2.5 text-xs font-medium text-amber-700 dark:text-amber-400">
            Tracked in #1607 triage groups: <span className="font-mono">{xclGroupIds.join(', ')}</span>
          </p>
        )}

        {item.labels.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
            {item.labels.map((lbl) => (
              <span
                key={lbl.id || lbl.name}
                className="inline-flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300"
              >
                <span
                  aria-hidden="true"
                  className="inline-block h-2 w-2 rounded-[2px]"
                  style={{ backgroundColor: `#${lbl.color}` }}
                />
                <span>{lbl.name}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Local Triage Controls */}
      <div className="border-b border-slate-200 bg-slate-50/70 p-4 sm:px-5 dark:border-slate-800 dark:bg-slate-900/40">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <label
              htmlFor="detail-triage-status"
              className="text-xs font-semibold text-slate-700 dark:text-slate-300"
            >
              Item Triage Status
            </label>
            <select
              id="detail-triage-status"
              value={triageRecord?.status ?? ''}
              onChange={(e) =>
                onUpdateTriage(item, {
                  status: (e.target.value || null) as TriageStatus | null,
                })
              }
              className="h-8 rounded-md border border-slate-300 bg-white px-2.5 text-xs font-medium text-slate-800 focus:border-blue-600 focus:outline-none dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
            >
              <option value="">Untriaged</option>
              <option value="needs_review">Needs Review</option>
              <option value="in_progress">In Progress</option>
              <option value="ready_to_merge">Ready to Merge</option>
              <option value="blocked">Blocked</option>
              <option value="snoozed">Snoozed</option>
            </select>
          </div>

          <button
            type="button"
            onClick={() => onUpdateTriage(item, { pinned: !triageRecord?.pinned })}
            aria-pressed={Boolean(triageRecord?.pinned)}
            className={`inline-flex h-8 items-center gap-1.5 rounded-md border px-3 text-xs font-medium transition-colors ${
              triageRecord?.pinned
                ? 'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-700 dark:bg-amber-950/50 dark:text-amber-200'
                : 'border-slate-300 bg-white text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:bg-slate-800'
            }`}
          >
            <Pin className="h-3.5 w-3.5" aria-hidden="true" />
            <span>{triageRecord?.pinned ? 'Pinned' : 'Pin to Queue'}</span>
          </button>
        </div>

        <div className="mt-3">
          <label htmlFor="triage-note-input" className="sr-only">
            Personal triage note
          </label>
          <textarea
            id="triage-note-input"
            rows={2}
            value={triageRecord?.note ?? ''}
            onChange={(e) => onUpdateTriage(item, { note: e.target.value })}
            placeholder="Add local triage note (saved in browser)..."
            className="w-full resize-none rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-800 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none dark:border-slate-800 dark:bg-slate-950 dark:text-slate-100"
          />
        </div>
      </div>

      {/* Body & Discussion */}
      <div className="flex-1 space-y-6 overflow-y-auto p-4 sm:p-5">
        <section aria-label="Description">
          <MarkdownLite content={item.body} />
        </section>

        {/* Comments thread */}
        <section aria-label="Comments" className="border-t border-slate-200 pt-5 dark:border-slate-800">
          <div className="mb-3 flex items-center gap-2 text-xs font-semibold text-slate-900 dark:text-slate-100">
            <MessageSquare className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" />
            <span className="tabular-nums">Comments ({item.comments})</span>
          </div>

          {loadingComments ? (
            <p className="text-xs text-slate-500 dark:text-slate-400">Loading discussion...</p>
          ) : comments.length > 0 ? (
            <div className="space-y-4">
              {comments.map((c) => (
                <div
                  key={c.id}
                  className="rounded-lg border border-slate-200 bg-slate-50/50 p-3.5 dark:border-slate-800 dark:bg-slate-900/40"
                >
                  <div className="mb-2 flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                    <Avatar src={c.user.avatar_url} name={c.user.login} size="sm" />
                    <span className="font-semibold text-slate-800 dark:text-slate-200">
                      {c.user.login}
                    </span>
                    <span aria-hidden="true">·</span>
                    <span>{formatRelativeTime(c.created_at)}</span>
                  </div>
                  <MarkdownLite content={c.body} />
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {item.comments === 0
                ? 'No comments yet on this thread.'
                : 'Comments could not be loaded right now — open on GitHub to view full thread.'}
            </p>
          )}
        </section>

        {/* Quick Reply Composer */}
        <section
          aria-label="Quick response composer"
          className="border-t border-slate-200 pt-5 dark:border-slate-800"
        >
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
              Draft Quick Reply
            </h3>
            <div className="flex flex-wrap items-center gap-1.5">
              {QUICK_REPLY_SNIPPETS.map((snippet) => (
                <button
                  key={snippet.label}
                  type="button"
                  onClick={() => setReplyDraft(snippet.text)}
                  className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-100 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800"
                >
                  {snippet.label}
                </button>
              ))}
            </div>
          </div>

          <label htmlFor="quick-reply-textarea" className="sr-only">
            Comment draft
          </label>
          <textarea
            id="quick-reply-textarea"
            rows={3}
            value={replyDraft}
            onChange={(e) => setReplyDraft(e.target.value)}
            placeholder="Draft a response or select a snippet above..."
            className="w-full rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-800 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none dark:border-slate-800 dark:bg-slate-900 dark:text-slate-100"
          />

          <div className="mt-2.5 flex flex-wrap items-center justify-end gap-2">
            <button
              type="button"
              disabled={!replyDraft.trim()}
              onClick={() => copyText(replyDraft, 'reply')}
              className="inline-flex h-8 items-center gap-1.5 rounded-md border border-slate-300 bg-white px-3 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
            >
              {copiedKey === 'reply' ? (
                <Check className="h-3.5 w-3.5 text-emerald-600" aria-hidden="true" />
              ) : (
                <Copy className="h-3.5 w-3.5" aria-hidden="true" />
              )}
              <span>{copiedKey === 'reply' ? 'Copied' : 'Copy Comment'}</span>
            </button>

            <a
              href={`${item.html_url}#issuecomment-new`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-8 items-center gap-1.5 rounded-md bg-slate-900 px-3 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-blue-600 dark:hover:bg-blue-500"
            >
              <span>Comment on GitHub</span>
              <ExternalLink className="h-3 w-3" aria-hidden="true" />
            </a>
          </div>
        </section>
      </div>
    </aside>
  );
}
