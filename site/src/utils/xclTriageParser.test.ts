import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildXclCheckBlock,
  buildXclTestReportComment,
  parseXcl1607Digest,
} from './xclTriageParser';

const SAMPLE_1607_BODY = `
**The project's PR triage.**
<!-- xcl-digest:begin -->
### Checked by two contributors (same finding)
| unit | issue | PRs | finding | agreeing checks | checked by |
|---|---|---|---|---|---|
| D20 | #105 | #297 #313 #1371 | different approaches: packed members | 2 | [alice](https://github.com/alice) |

### Checks disagree (a third look would help)
| unit | issue | PRs | finding | checks |
|---|---|---|---|---|

### Waiting for a second check
| unit | issue | kind | PRs | note | checks on current heads | outdated checks |
|---|---|---|---|---|---|---|
| D51 | #175 | dup | #867 #1395 | vkd3d-proton's vtable assertions on adopted objects under Wine | [bob](https://github.com/bob) | 2 |
| B9 | – | batch on 4178e88 | #632 #647 | – | [carol](https://github.com/carol) (nv-linux) | 1 |

<details><summary>Done (1)</summary>
- D10: #543 #1531 (closed)
</details>

<!-- xcl-ids:{"dup|297,313,1371|":"D20","dup|867,1395|":"D51","dup|543,1531|":"D10","batch|448,1273|4178e88":"B10","batch|632,647|4178e88":"B9"} -->
<!-- xcl-digest:end -->
<!-- xcl-reports:begin -->
### Independent test reports per PR
_Counted 2026-11-25 01:24 UTC by a script over all 1112 open PRs._
Ready PRs (not draft, \`PR description\` check green): **538**. With 2+ independent reports on their current head: **155**; with 1: **196**; with none on the current head: **187** (82 of them have reports on an earlier head).

### Most likely ready to merge
| PR | linked issue | reports on current head | by | title |
|---|---|---|---|---|
| #587 | yes | 3 | CryingLaughing, MesshM | Recompiler: v_frexp_mant_f32 keep sign (#\u200b580) |
| #1341 | – | 6 | Aniseto, Creaa | Recompiler: reuse lane predicate |

### One more test report would make these ready
_Same rules, with one independent report on the current head so far._
#543 #1050 #1691

<details><summary>Ready PRs with no report on their current head (oldest 40)</summary>#264 #271 #272</details>
<!-- xcl-reports:end -->
`;

test('parseXcl1607Digest extracts counts, tables, report queues, and xcl-ids map', () => {
  const digest = parseXcl1607Digest(SAMPLE_1607_BODY);

  assert.equal(digest.updatedLabel, '2026-11-25 01:24 UTC');
  assert.equal(digest.totalOpenPrs, 1112);
  assert.equal(digest.readyPrs, 538);
  assert.equal(digest.twoPlusReports, 155);
  assert.equal(digest.oneReport, 196);
  assert.equal(digest.zeroReports, 187);
  assert.equal(digest.earlierHeadReports, 82);

  assert.equal(digest.readyToMerge.length, 2);
  assert.deepEqual(digest.readyToMerge[0], {
    number: 587,
    linkedIssue: true,
    reportsCount: 3,
    reporters: 'CryingLaughing, MesshM',
    title: 'Recompiler: v_frexp_mant_f32 keep sign (#580)',
  });

  assert.deepEqual(digest.oneMoreReportPrs, [543, 1050, 1691]);
  assert.deepEqual(digest.zeroReportPrs, [264, 271, 272]);

  assert.equal(digest.groups.length, 5);
  const b9 = digest.groups.find((g) => g.unit === 'B9');
  assert.equal(b9?.summary, '');
  assert.equal(b9?.checkedBy, 'carol (nv-linux)');
  const d10 = digest.groups.find((g) => g.unit === 'D10');
  assert.equal(d10?.status, 'done');
  assert.equal(digest.prToGroups[543], undefined);
  assert.deepEqual(digest.prToGroups[867], ['D51']);
  assert.deepEqual(digest.prToGroups[448], ['B10']);
});

test('buildXclTestReportComment includes 7-char head SHA and N passed, N failed format', () => {
  const comment = buildXclTestReportComment({
    headSha: 'a1b2c3d4e5f6',
    passedCount: 812,
    failedCount: 0,
    hardwareClass: 'nv-win',
    notes: 'Smoke check clean',
    aiUsed: 'none',
  });
  assert.match(comment, /`a1b2c3d`/);
  assert.match(comment, /812 passed, 0 failed/);
  assert.match(comment, /nv-win/);
});

test('buildXclCheckBlock formats valid xcl-check fenced JSON block', () => {
  const block = buildXclCheckBlock({
    unit: 'D51',
    verdict: 'suggest 867; cleaner fix',
    heads: { '867': '1122334', '1395': '5566778' },
    how: 'read both diffs',
    hardwareClass: 'nv-win',
    aiUsed: 'none',
  });
  assert.ok(block.startsWith('```xcl-check\n'));
  const jsonText = block.replace(/^```xcl-check\n/, '').replace(/\n```$/, '');
  const parsed = JSON.parse(jsonText) as { unit: string; verdict: string };
  assert.equal(parsed.unit, 'D51');
  assert.equal(parsed.verdict, 'suggest 867; cleaner fix');
});
