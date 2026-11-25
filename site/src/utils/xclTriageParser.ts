export interface XclReadyPrRow {
  number: number;
  linkedIssue: boolean;
  reportsCount: number;
  reporters: string;
  title: string;
}

export interface XclGroupUnit {
  unit: string;
  kind: 'dup' | 'batch';
  prs: number[];
  baseSha: string | null;
  issueNumber: number | null;
  status: 'waiting_second' | 'disagree' | 'checked' | 'done';
  summary: string;
  checkedBy: string;
}

export interface Xcl1607Digest {
  updatedLabel: string;
  totalOpenPrs: number;
  readyPrs: number;
  twoPlusReports: number;
  oneReport: number;
  zeroReports: number;
  earlierHeadReports: number;
  readyToMerge: XclReadyPrRow[];
  oneMoreReportPrs: number[];
  zeroReportPrs: number[];
  groups: XclGroupUnit[];
  prToGroups: Record<number, string[]>;
}

function cleanMarkdownCell(raw: string): string {
  const cleaned = raw
    .replace(/\u200b/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .trim();
  return cleaned === '-' || cleaned === '–' ? '' : cleaned;
}

function extractPrNumbers(text: string): number[] {
  const matches = text.match(/#\u200b?(\d+)/g) ?? [];
  const nums: number[] = [];
  for (const m of matches) {
    const n = Number(m.replace(/[^0-9]/g, ''));
    if (Number.isFinite(n) && n > 0 && !nums.includes(n)) {
      nums.push(n);
    }
  }
  return nums;
}

function parseMarkdownTableRows(sectionText: string): string[][] {
  const lines = sectionText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.startsWith('|'));

  const rows: string[][] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const isDivider = /^\|[\s:-]+\|/.test(line) && !/[A-Za-z0-9#]/.test(line);
    if (isDivider) {
      continue;
    }
    // Skip header row if the very next line is a |---| divider
    const nextLine = lines[i + 1] ?? '';
    const nextIsDivider = /^\|[\s:-]+\|/.test(nextLine) && !/[A-Za-z0-9#]/.test(nextLine);
    if (nextIsDivider) {
      continue;
    }

    const cells = line
      .replace(/^\|/, '')
      .replace(/\|$/, '')
      .split('|')
      .map((c) => c.trim());
    rows.push(cells);
  }
  return rows;
}

function sliceBetween(body: string, startMarker: string, endMarkers: string[]): string {
  const startIdx = body.indexOf(startMarker);
  if (startIdx < 0) return '';
  const afterStart = startIdx + startMarker.length;
  let endIdx = body.length;
  for (const marker of endMarkers) {
    const idx = body.indexOf(marker, afterStart);
    if (idx >= 0 && idx < endIdx) {
      endIdx = idx;
    }
  }
  return body.slice(afterStart, endIdx);
}

export function parseXcl1607Digest(body: string): Xcl1607Digest {
  const cleanBody = body.replace(/\u200b/g, '');

  const reportsSection = sliceBetween(cleanBody, '<!-- xcl-reports:begin -->', [
    '<!-- xcl-reports:end -->',
  ]);
  const digestSection = sliceBetween(cleanBody, '<!-- xcl-digest:begin -->', [
    '<!-- xcl-digest:end -->',
  ]);

  const updatedMatch = reportsSection.match(/Counted\s+([0-9-]+\s+[0-9:]+\s+UTC)/i);
  const totalOpenMatch = reportsSection.match(/over all\s+(\d+)\s+open PRs/i);
  const readyMatch = reportsSection.match(/Ready PRs[^:]*:\s*\*\*(\d+)\*\*/i);
  const twoPlusMatch = reportsSection.match(/2\+ independent reports[^:]*:\s*\*\*(\d+)\*\*/i);
  const oneMatch = reportsSection.match(/with 1:\s*\*\*(\d+)\*\*/i);
  const zeroMatch = reportsSection.match(/with none on the current head:\s*\*\*(\d+)\*\*/i);
  const earlierMatch = reportsSection.match(/\((\d+)\s+of them have reports on an earlier head\)/i);

  // Ready to merge table
  const readyTableSlice = sliceBetween(reportsSection, '### Most likely ready to merge', [
    '### One more test report would make these ready',
  ]);
  const readyToMerge: XclReadyPrRow[] = parseMarkdownTableRows(readyTableSlice)
    .map((cells) => {
      const prNum = Number((cells[0] ?? '').replace(/[^0-9]/g, ''));
      if (!prNum) return null;
      return {
        number: prNum,
        linkedIssue: (cells[1] ?? '').toLowerCase().includes('yes'),
        reportsCount: Number((cells[2] ?? '0').replace(/[^0-9]/g, '')) || 0,
        reporters: cleanMarkdownCell(cells[3] ?? ''),
        title: cleanMarkdownCell(cells[4] ?? ''),
      };
    })
    .filter((row): row is XclReadyPrRow => row !== null);

  // One more test report list
  const oneMoreSlice = sliceBetween(
    reportsSection,
    '### One more test report would make these ready',
    ['<details>', '<!-- xcl-reports:end -->']
  );
  // Strip descriptive sentence before the #PR list
  const oneMoreFirstHash = oneMoreSlice.indexOf('#');
  const oneMoreReportPrs =
    oneMoreFirstHash >= 0 ? extractPrNumbers(oneMoreSlice.slice(oneMoreFirstHash)) : [];

  // Zero report list inside <details>
  const detailsSlice = sliceBetween(reportsSection, '<details>', ['</details>']);
  const summaryEnd = detailsSlice.indexOf('</summary>');
  const zeroReportPrs =
    summaryEnd >= 0 ? extractPrNumbers(detailsSlice.slice(summaryEnd + 10)) : [];

  // Parse <!-- xcl-ids:{...} -->
  const idsMatch = cleanBody.match(/<!--\s*xcl-ids:(\{[\s\S]*?\})\s*-->/);
  const groupMap = new Map<string, XclGroupUnit>();
  const prToGroups: Record<number, string[]> = {};

  if (idsMatch) {
    try {
      const rawMap = JSON.parse(idsMatch[1]) as Record<string, string>;
      for (const [spec, unitId] of Object.entries(rawMap)) {
        const [kindRaw, prsRaw, baseRaw] = spec.split('|');
        const prs = (prsRaw ?? '')
          .split(',')
          .map((s) => Number(s.trim()))
          .filter((n) => Number.isFinite(n) && n > 0);
        const kind: 'dup' | 'batch' = kindRaw === 'batch' ? 'batch' : 'dup';
        groupMap.set(unitId, {
          unit: unitId,
          kind,
          prs,
          baseSha: baseRaw?.trim() || null,
          issueNumber: null,
          status: 'checked',
          summary: '',
          checkedBy: '',
        });
        for (const pr of prs) {
          if (!prToGroups[pr]) prToGroups[pr] = [];
          if (!prToGroups[pr].includes(unitId)) {
            prToGroups[pr].push(unitId);
          }
        }
      }
    } catch {
      // Ignore malformed xcl-ids JSON
    }
  }

  // Enrich from digest tables
  const checkedSlice = sliceBetween(digestSection, '### Checked by two contributors', [
    '### Checks disagree',
    '### Waiting for a second check',
    '### Label checks',
  ]);
  for (const cells of parseMarkdownTableRows(checkedSlice)) {
    const unit = (cells[0] ?? '').trim();
    if (!unit) continue;
    const issueNum = Number((cells[1] ?? '').replace(/[^0-9]/g, '')) || null;
    const prs = extractPrNumbers(cells[2] ?? '');
    const existing = groupMap.get(unit);
    groupMap.set(unit, {
      unit,
      kind: unit.startsWith('B') ? 'batch' : 'dup',
      prs: prs.length > 0 ? prs : existing?.prs ?? [],
      baseSha: existing?.baseSha ?? null,
      issueNumber: issueNum,
      status: 'checked',
      summary: cleanMarkdownCell(cells[3] ?? ''),
      checkedBy: cleanMarkdownCell(cells[5] ?? ''),
    });
  }

  const disagreeSlice = sliceBetween(digestSection, '### Checks disagree', [
    '### Waiting for a second check',
    '### Label checks',
  ]);
  for (const cells of parseMarkdownTableRows(disagreeSlice)) {
    const unit = (cells[0] ?? '').trim();
    if (!unit) continue;
    const issueNum = Number((cells[1] ?? '').replace(/[^0-9]/g, '')) || null;
    const prs = extractPrNumbers(cells[2] ?? '');
    const existing = groupMap.get(unit);
    groupMap.set(unit, {
      unit,
      kind: unit.startsWith('B') ? 'batch' : 'dup',
      prs: prs.length > 0 ? prs : existing?.prs ?? [],
      baseSha: existing?.baseSha ?? null,
      issueNumber: issueNum,
      status: 'disagree',
      summary: cleanMarkdownCell(cells[3] ?? ''),
      checkedBy: cleanMarkdownCell(cells[4] ?? ''),
    });
  }

  const waitingSlice = sliceBetween(digestSection, '### Waiting for a second check', [
    '### Label checks',
    '<!-- xcl-ids:',
  ]);
  for (const cells of parseMarkdownTableRows(waitingSlice)) {
    const unit = (cells[0] ?? '').trim();
    if (!unit) continue;
    const issueNum = Number((cells[1] ?? '').replace(/[^0-9]/g, '')) || null;
    const prs = extractPrNumbers(cells[3] ?? '');
    const existing = groupMap.get(unit);
    groupMap.set(unit, {
      unit,
      kind: unit.startsWith('B') ? 'batch' : 'dup',
      prs: prs.length > 0 ? prs : existing?.prs ?? [],
      baseSha: existing?.baseSha ?? null,
      issueNumber: issueNum,
      status: 'waiting_second',
      summary: cleanMarkdownCell(cells[4] ?? ''),
      checkedBy: cleanMarkdownCell(cells[5] ?? ''),
    });
  }

  // Mark groups inside <details>Done (...)</details> or <details>Taken out...</details> as done
  const retiredRegex = /^-\s+([A-Z]\d+):/gm;
  const takenOutSlice = sliceBetween(digestSection, 'Taken out of the table', ['</details>']);
  const doneSlice = sliceBetween(digestSection, '<summary>Done', ['</details>']);
  for (const retiredText of [takenOutSlice, doneSlice]) {
    let m: RegExpExecArray | null;
    retiredRegex.lastIndex = 0;
    while ((m = retiredRegex.exec(retiredText)) !== null) {
      const unitId = m[1];
      const existing = groupMap.get(unitId);
      if (existing) {
        existing.status = 'done';
      }
    }
  }

  // Keep prToGroups focused on active groups (exclude already-done groups)
  for (const [prStr, unitIds] of Object.entries(prToGroups)) {
    const pr = Number(prStr);
    const activeIds = unitIds.filter((id) => groupMap.get(id)?.status !== 'done');
    if (activeIds.length > 0) {
      prToGroups[pr] = activeIds;
    } else {
      delete prToGroups[pr];
    }
  }

  const groups = Array.from(groupMap.values()).sort((a, b) => {
    const order = { waiting_second: 0, disagree: 1, checked: 2, done: 3 };
    if (order[a.status] !== order[b.status]) return order[a.status] - order[b.status];
    return a.unit.localeCompare(b.unit, undefined, { numeric: true });
  });

  return {
    updatedLabel: updatedMatch?.[1] ?? 'Snapshot',
    totalOpenPrs: Number(totalOpenMatch?.[1] ?? 0),
    readyPrs: Number(readyMatch?.[1] ?? 0),
    twoPlusReports: Number(twoPlusMatch?.[1] ?? readyToMerge.length),
    oneReport: Number(oneMatch?.[1] ?? oneMoreReportPrs.length),
    zeroReports: Number(zeroMatch?.[1] ?? zeroReportPrs.length),
    earlierHeadReports: Number(earlierMatch?.[1] ?? 0),
    readyToMerge,
    oneMoreReportPrs,
    zeroReportPrs,
    groups,
    prToGroups,
  };
}

export interface BuildTestReportOptions {
  headSha: string;
  passedCount: number;
  failedCount: number;
  hardwareClass: string;
  notes: string;
  aiUsed: string;
}

export function buildXclTestReportComment(opts: BuildTestReportOptions): string {
  const shortSha = opts.headSha.trim().slice(0, 7) || '<head-sha>';
  const lines = [
    `Tested commit \`${shortSha}\` (${opts.hardwareClass}):`,
    `- \`cargo test --workspace\`: ${opts.passedCount} passed, ${opts.failedCount} failed`,
  ];
  if (opts.notes.trim()) {
    lines.push(`- Notes: ${opts.notes.trim()}`);
  }
  lines.push(`- AI use: ${opts.aiUsed.trim() || 'none'}`);
  return lines.join('\n');
}

export interface BuildXclCheckOptions {
  unit: string;
  verdict: string;
  heads: Record<string, string>;
  how: string;
  hardwareClass: string;
  aiUsed: string;
}

export function buildXclCheckBlock(opts: BuildXclCheckOptions): string {
  const payload = {
    unit: opts.unit.trim(),
    verdict: opts.verdict.trim(),
    heads: opts.heads,
    how: opts.how.trim(),
    class: opts.hardwareClass,
    ai: opts.aiUsed.trim() || 'none',
  };
  return '```xcl-check\n' + JSON.stringify(payload) + '\n```';
}
