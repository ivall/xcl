import React, { useEffect, useState } from 'react';
import { Check, Copy, ExternalLink, X } from 'lucide-react';
import { ISSUE_TEMPLATES } from '../data/seedData';
import { buildNewIssueUrl } from '../utils/githubParse';

interface NewIssueModalProps {
  isOpen: boolean;
  owner: string;
  repo: string;
  onClose: () => void;
}

export function NewIssueModal({ isOpen, owner, repo, onClose }: NewIssueModalProps) {
  const [templateId, setTemplateId] = useState(ISSUE_TEMPLATES[0].id);
  const [title, setTitle] = useState(ISSUE_TEMPLATES[0].titlePrefix);
  const [labelsInput, setLabelsInput] = useState(ISSUE_TEMPLATES[0].defaultLabels.join(', '));
  const [body, setBody] = useState(ISSUE_TEMPLATES[0].body);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const applyTemplate = (id: string) => {
    const tpl = ISSUE_TEMPLATES.find((t) => t.id === id);
    if (!tpl) return;
    setTemplateId(tpl.id);
    setTitle(tpl.titlePrefix);
    setLabelsInput(tpl.defaultLabels.join(', '));
    setBody(tpl.body);
    setError(null);
  };

  const parsedLabels = labelsInput
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);

  const validate = (): boolean => {
    const clean = title.trim();
    if (!clean || clean.endsWith(':')) {
      setError('Enter a descriptive issue title before opening or copying.');
      return false;
    }
    setError(null);
    return true;
  };

  const handleCopy = () => {
    if (!validate()) return;
    const markdown = `# ${title.trim()}\n\n${body.trim()}`;
    navigator.clipboard?.writeText(markdown).catch(() => {});
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  };

  const handleOpenGitHub = (e: React.MouseEvent<HTMLAnchorElement>) => {
    if (!validate()) {
      e.preventDefault();
    }
  };

  const githubUrl = buildNewIssueUrl(owner, repo, title, body, parsedLabels);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="new-issue-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4"
    >
      <div className="w-full max-w-2xl overflow-hidden rounded-xl border border-slate-200 bg-white shadow-lg dark:border-slate-800 dark:bg-slate-950">
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <div>
            <h2
              id="new-issue-modal-title"
              className="text-base font-bold text-slate-900 dark:text-slate-50"
            >
              Draft New Issue for <span className="font-mono">{owner}/{repo}</span>
            </h2>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              Prepare a structured report and open it pre-filled on GitHub.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close new issue modal"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-900 dark:hover:text-white"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>

        <div className="space-y-4 p-5">
          <div>
            <span className="mb-1.5 block text-xs font-semibold text-slate-700 dark:text-slate-300">
              Template
            </span>
            <div className="flex flex-wrap gap-1.5">
              {ISSUE_TEMPLATES.map((tpl) => {
                const active = templateId === tpl.id;
                return (
                  <button
                    key={tpl.id}
                    type="button"
                    onClick={() => applyTemplate(tpl.id)}
                    aria-pressed={active}
                    className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                      active
                        ? 'border-blue-600 bg-blue-50 text-blue-800 dark:border-blue-500 dark:bg-blue-950/50 dark:text-blue-200'
                        : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300'
                    }`}
                  >
                    {tpl.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <label
              htmlFor="new-issue-title"
              className="mb-1 block text-xs font-semibold text-slate-700 dark:text-slate-300"
            >
              Title
            </label>
            <input
              id="new-issue-title"
              type="text"
              value={title}
              onChange={(e) => {
                setTitle(e.target.value);
                if (error) setError(null);
              }}
              placeholder="e.g. Recompiler: wrong mask on wave ballot"
              className="h-9 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-900 focus:border-blue-600 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
            />
          </div>

          <div>
            <label
              htmlFor="new-issue-labels"
              className="mb-1 block text-xs font-semibold text-slate-700 dark:text-slate-300"
            >
              Suggested Labels (comma-separated)
            </label>
            <input
              id="new-issue-labels"
              type="text"
              value={labelsInput}
              onChange={(e) => setLabelsInput(e.target.value)}
              placeholder="bug, recompiler"
              className="h-9 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono text-xs text-slate-900 focus:border-blue-600 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
            />
          </div>

          <div>
            <label
              htmlFor="new-issue-body"
              className="mb-1 block text-xs font-semibold text-slate-700 dark:text-slate-300"
            >
              Description (Markdown)
            </label>
            <textarea
              id="new-issue-body"
              rows={7}
              value={body}
              onChange={(e) => setBody(e.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white p-3 font-mono text-xs leading-relaxed text-slate-900 focus:border-blue-600 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
            />
          </div>

          {error && (
            <p role="alert" className="text-xs font-medium text-rose-600 dark:text-rose-400">
              {error}
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2 border-t border-slate-200 bg-slate-50 px-5 py-3.5 dark:border-slate-800 dark:bg-slate-900/50">
          <button
            type="button"
            onClick={onClose}
            className="h-9 rounded-lg border border-slate-300 bg-white px-4 text-xs font-medium text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
          >
            Cancel
          </button>

          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-4 text-xs font-semibold text-slate-800 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
          >
            {copied ? (
              <Check className="h-3.5 w-3.5 text-emerald-600" aria-hidden="true" />
            ) : (
              <Copy className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            <span>{copied ? 'Copied Markdown' : 'Copy Markdown'}</span>
          </button>

          <a
            href={githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            onClick={handleOpenGitHub}
            className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-slate-900 px-4 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-blue-600 dark:hover:bg-blue-500"
          >
            <span>Open Pre-filled on GitHub</span>
            <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
          </a>
        </div>
      </div>
    </div>
  );
}
