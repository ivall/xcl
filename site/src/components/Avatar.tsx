import { useState } from 'react';

interface AvatarProps {
  src?: string | null;
  name: string;
  size?: 'sm' | 'md';
}

export function Avatar({ src, name, size = 'sm' }: AvatarProps) {
  const [failed, setFailed] = useState(false);
  const initials = name
    .replace(/[^a-zA-Z0-9]/g, '')
    .slice(0, 2)
    .toUpperCase() || '?';

  const dimClass = size === 'md' ? 'w-7 h-7 text-xs' : 'w-5 h-5 text-[10px]';

  if (!src || failed) {
    return (
      <span
        aria-hidden="true"
        className={`${dimClass} inline-flex shrink-0 items-center justify-center rounded-md border border-slate-200 bg-slate-100 font-mono font-semibold text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300`}
      >
        {initials}
      </span>
    );
  }

  return (
    <img
      src={src}
      alt={name}
      referrerPolicy="no-referrer"
      onError={() => setFailed(true)}
      className={`${dimClass} shrink-0 rounded-md border border-slate-200 object-cover dark:border-slate-700`}
    />
  );
}
