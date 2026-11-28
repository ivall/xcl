import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchAllPages } from './githubApi';

test('fetchAllPages requests every page and preserves API order', async () => {
  const originalFetch = globalThis.fetch;
  const requested: string[] = [];

  globalThis.fetch = async (input) => {
    const url = String(input);
    requested.push(url);
    const page = Number(new URL(url).searchParams.get('page'));
    const body = page === 1 ? Array.from({ length: 100 }, (_, id) => id) : [100, 101];
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };

  try {
    const items = await fetchAllPages<number>('https://api.github.test/items?state=all');
    assert.equal(items.length, 102);
    assert.deepEqual(items.slice(-3), [99, 100, 101]);
    assert.deepEqual(requested, [
      'https://api.github.test/items?state=all&per_page=100&page=1',
      'https://api.github.test/items?state=all&per_page=100&page=2',
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('fetchAllPages stops after an empty first page', async () => {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return new Response('[]', { status: 200 });
  };

  try {
    assert.deepEqual(await fetchAllPages('https://api.github.test/items'), []);
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
