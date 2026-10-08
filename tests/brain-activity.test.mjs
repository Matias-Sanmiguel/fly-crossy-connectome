import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { BRAIN_ACTIVITY_CONTRAST, createBrainActivityBuffer, displayedActivityStrength } from '../src/lib/brainActivity.ts';

test('brain activity reuses its buffer and indexed IDs, clears absent values and never invents positions', () => {
  const buffer = createBrainActivityBuffer([30, 10, 20]);
  const original = buffer.values;
  buffer.write([[10, 0.5], [20, 1], [99, 1]]);
  assert.deepEqual(Array.from(buffer.values), [0, 0.5, 1]);
  buffer.write([[30, 0.25], [10, 0]]);
  assert.deepEqual(Array.from(buffer.values), [0.25, 0, 0]);
  buffer.write([]);
  assert.deepEqual(Array.from(buffer.values), [0, 0, 0]);
  assert.equal(buffer.values, original);
});

test('expo makes genuine weak signals perceptible while keeping zero dark and preserving ordering', () => {
  assert.equal(displayedActivityStrength(0, 'expo'), 0);
  assert.ok(displayedActivityStrength(1 / 255, 'expo') > 0.6);
  let previous = 0;
  for (let i = 0; i <= 255; i++) {
    const strength = displayedActivityStrength(i / 255, 'expo');
    assert.ok(strength >= previous && strength <= 1);
    previous = strength;
  }
  assert.equal(displayedActivityStrength(0.5, 'standard'), Math.pow(0.5, 0.4));
});

test('expo enlarges only active markers; anatomy and the main profile remain unchanged', async () => {
  const brain = await readFile(new URL('../src/components/BrainScene.tsx', import.meta.url), 'utf8');
  assert.match(brain, /activityContrast = 'standard'/);
  assert.match(brain, /activity > 0\. \? activityFloor/);
  assert.match(brain, /activity > 0\. && activePointSize > 0\. \? activePointSize : 0\.9 \+ clamp\(activity,0\.,1\.\) \* 2\.0/);
  assert.equal(BRAIN_ACTIVITY_CONTRAST.standard.activePointSize, 0);
  assert.equal(BRAIN_ACTIVITY_CONTRAST.expo.activePointSize, 2.6);
  assert.doesNotMatch(brain, /gl_PointSize[^;]*(?:activityGain|activityFloor)/);
  assert.doesNotMatch(brain, /new Map\(signal\.current/);
});
