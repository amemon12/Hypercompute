import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GIB, initialRoom, machineCapacity, weightBytes, uniquePools, copyLiveRoom, sumKnown, retainedReserveBytes } from './calculations.ts';
test('1T ideal 4-bit is500decimalGB; effective4.85bpw costs606.25GB', () => { assert.equal(weightBytes(1000, 4), 500e9); assert.equal(weightBytes(1000, 4.85), 606.25e9); });
test('unknown specs remain unknown', () => { assert.equal(machineCapacity(initialRoom()[0]).bytes, null); assert.equal(sumKnown([100, null]), null); });
test('unified memory counted once with reserve removed', () => { assert.equal(machineCapacity({ ...initialRoom()[0], memoryGiB: '128', reserveGiB: '16' }).bytes, 112 * GIB); });
test('fresh free memory caps hypothetical usable memory', () => { assert.equal(machineCapacity({ ...initialRoom()[0], memoryGiB: '128', freeGiB: '60', reserveGiB: '16' }).bytes, 44 * GIB); });
test('invalid free memory cannot generate a false fit', () => { assert.equal(machineCapacity({ ...initialRoom()[0], memoryGiB: '16', freeGiB: '20', reserveGiB: '4' }).bytes, null); });
test('Windows host RAM is added only to the host, VRAM counted separately', () => { const m = { ...initialRoom()[3], memoryGiB: '32', reserveGiB: '8', vramGiB: '12' }; assert.equal(machineCapacity(m).bytes, 35.5 * GIB); assert.equal(machineCapacity({ ...m, isHost: false }).bytes, 11.5 * GIB); });
test('pool ids are namespaced by node; duplicate reports do not double count', () => { const pool = { id: 'uma', kind: 'unified', total_bytes: 16 * GIB, free_bytes: 10 * GIB }; const a = { id: 'a', inventory: { memory_pools: [pool, pool] } }; const b = { id: 'b', inventory: { memory_pools: [pool] } }; assert.equal(uniquePools([a, b]).length, 2); });
test('copying live data creates independent scenario rows', () => { const node = { id: 'a', display_name: 'Mac', is_local: true, inventory: { memory_pools: [{ id: 'uma', kind: 'unified', total_bytes: 16 * GIB, free_bytes: 10 * GIB }] }, contribution: { reserve_bytes: 4 * GIB } }; const result = copyLiveRoom([node]); result[0].name = 'Edited'; assert.equal(node.display_name, 'Mac'); assert.equal(machineCapacity(result[0]).bytes, 6 * GIB); });

test('invalid nonempty free values never assume full installed memory', () => { for (const freeGiB of ['-1', 'Infinity', 'NaN']) assert.equal(machineCapacity({ ...initialRoom()[0], memoryGiB: '16', reserveGiB: '4', freeGiB }).bytes, null); });
test('overflow values remain unknown', () => { assert.equal(weightBytes(1e308, 4), null); assert.equal(machineCapacity({ ...initialRoom()[0], memoryGiB: '1e308', reserveGiB: '4' }).bytes, null); assert.equal(sumKnown([1e308, 1e308]), null); });
test('retained segment cannot exceed currently free memory', () => { assert.equal(retainedReserveBytes({ ...initialRoom()[0], memoryGiB: '16', reserveGiB: '8', freeGiB: '6' }), 6 * GIB); });

test('sharing cap limits unified available memory after reserves', () => {
  const machine = { ...initialRoom()[0], memoryGiB: '32', freeGiB: '24', reserveGiB: '4', contributionLimitGiB: '12' };
  assert.equal(machineCapacity(machine).bytes, 12 * GIB);
  assert.equal(retainedReserveBytes(machine), 12 * GIB);
});
test('sharing cap applies once to combined host RAM and VRAM', () => {
  const machine = { ...initialRoom()[3], memoryGiB: '32', reserveGiB: '8', vramGiB: '12', contributionLimitGiB: '20' };
  assert.equal(machineCapacity(machine).bytes, 20 * GIB);
  assert.equal(retainedReserveBytes(machine), 24 * GIB);
});
test('blank or larger sharing limit keeps available memory and zero explicitly shares none', () => {
  const machine = { ...initialRoom()[0], memoryGiB: '16', reserveGiB: '4' };
  assert.equal(machineCapacity({ ...machine, contributionLimitGiB: '' }).bytes, 12 * GIB);
  assert.equal(machineCapacity({ ...machine, contributionLimitGiB: '30' }).bytes, 12 * GIB);
  assert.equal(machineCapacity({ ...machine, contributionLimitGiB: '0' }).bytes, 0);
});
test('invalid sharing limits remain unknown', () => {
  const machine = { ...initialRoom()[0], memoryGiB: '16', reserveGiB: '4' };
  for (const contributionLimitGiB of ['-1', 'NaN', 'Infinity', 'not a number', '1e308']) {
    assert.equal(machineCapacity({ ...machine, contributionLimitGiB }).bytes, null);
    assert.equal(retainedReserveBytes({ ...machine, contributionLimitGiB }), null);
  }
});
test('live copy preserves the exact sharing cap without editing live inventory', () => {
  const maxBytes = 3 * GIB + 12345;
  const node = { id: 'limited', display_name: 'Mac', is_local: true, inventory: { memory_pools: [{ id: 'uma', kind: 'unified', total_bytes: 16 * GIB, free_bytes: 10 * GIB }] }, contribution: { reserve_bytes: 4 * GIB, max_bytes: maxBytes } };
  const [copy] = copyLiveRoom([node]);
  assert.equal(Number(copy.contributionLimitGiB) * GIB, maxBytes);
  assert.equal(machineCapacity(copy).bytes, maxBytes);
  copy.contributionLimitGiB = '1';
  assert.equal(node.contribution.max_bytes, maxBytes);
});
