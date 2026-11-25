import { useEffect } from 'react';
import { X } from 'lucide-react';

interface DeployGuideModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function DeployGuideModal({ isOpen, onClose }: DeployGuideModalProps) {
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="deploy-guide-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4"
    >
      <div className="w-full max-w-lg overflow-hidden rounded-xl border border-slate-200 bg-white shadow-lg dark:border-slate-800 dark:bg-slate-950">
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <h2
            id="deploy-guide-title"
            className="text-base font-bold text-slate-900 dark:text-slate-50"
          >
            Host on GitHub Pages
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close hosting guide"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-900 dark:hover:text-white"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>

        <div className="space-y-4 p-5 text-sm leading-relaxed text-slate-700 dark:text-slate-300">
          <p>
            RepoPulse runs completely in the browser with no backend server required, so you can host it directly on your GitHub Pages website.
          </p>

          <ol className="list-decimal space-y-2.5 pl-5 text-xs sm:text-sm">
            <li>
              Push this project to a GitHub repository (for example{' '}
              <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs dark:bg-slate-900">
                your-name/repopulse
              </code>{' '}
              or{' '}
              <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs dark:bg-slate-900">
                your-name.github.io
              </code>
              ).
            </li>
            <li>
              In your repository on GitHub, open <strong>Settings → Pages</strong> and set{' '}
              <strong>Source</strong> to <strong>GitHub Actions</strong>.
            </li>
            <li>
              The included{' '}
              <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs dark:bg-slate-900">
                .github/workflows/deploy-pages.yml
              </code>{' '}
              workflow builds{' '}
              <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs dark:bg-slate-900">
                dist/
              </code>{' '}
              with relative paths and publishes automatically on every push to{' '}
              <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs dark:bg-slate-900">
                main
              </code>
              .
            </li>
          </ol>

          <p className="text-xs text-slate-500 dark:text-slate-400">
            All triage statuses, pins, and notes are stored in your browser&apos;s local storage, and live repository updates sync directly from GitHub&apos;s public REST API.
          </p>
        </div>

        <div className="flex justify-end border-t border-slate-200 bg-slate-50 px-5 py-3 dark:border-slate-800 dark:bg-slate-900/50">
          <button
            type="button"
            onClick={onClose}
            className="h-9 rounded-lg bg-slate-900 px-4 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-blue-600 dark:hover:bg-blue-500"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
