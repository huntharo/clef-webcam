import assert from 'node:assert/strict';
import test from 'node:test';
import { HistoryBuffer, motionPixels, validCrop, cropPoint, cropRectangle, historyRequest, historyQuestions, SPEAKING_MOTION } from '../camera.mjs';

function frame(time, level = 0, at = time, width = 2, height = 2) {
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let p = 0; p < pixels.length; p += 4) { pixels.fill(level, p, p + 3); pixels[p + 3] = 255; }
  return { at, mediaTime: time, width, height, pixels };
}
const fill = (buffer, end, step = 1000 / 30) => { for (let t = 0; t <= end + 0.001; t += step) buffer.push(frame(t)); };

test('selects three distinct camera samples near 100ms offsets from the newest frame', () => {
  const h = new HistoryBuffer(); fill(h, 600);
  const s = h.snapshot(600);
  assert.deepEqual(s.offsets, [-300, -200, -100]);
  assert.equal(s.samples.length, 10);
  assert.equal(s.current.mediaTime, 600);
  assert.deepEqual(s.history.map(f => Math.round(f.mediaTime)), [300, 400, 500]);
});

test('source timestamps determine real offsets despite uneven presentation times', () => {
  const h = new HistoryBuffer();
  [0, 110, 210, 290, 410].forEach((t, i) => h.push(frame(t, 0, 100 + i * 110)));
  assert.deepEqual(h.snapshot(540).offsets, [-300, -200, -120]);
  assert.equal(h.snapshot(691), null);
});

test('duplicates, gaps, reversed clocks, and source resolution changes cannot produce invented history', () => {
  const h = new HistoryBuffer(); fill(h, 400);
  assert.equal(h.push(frame(h.frames.at(-1).mediaTime)), false);
  assert.equal(h.snapshot(600), null);
  h.push(frame(800)); assert.equal(h.snapshot(800), null);
  h.push(frame(0, 0, 900)); assert.equal(h.frames.length, 1);
  h.push(frame(100, 0, 850)); assert.equal(h.frames.length, 1);
  h.push(frame(200, 0, 950, 1, 1)); assert.equal(h.frames.length, 1);
  h.reset(); assert.equal(h.snapshot(1000), null);
});

test('rejects invalid source data and requires a genuine history window', () => {
  const h = new HistoryBuffer();
  assert.equal(h.push(frame(NaN)), false);
  assert.equal(h.push({ ...frame(0), pixels: new Uint8ClampedArray(3) }), false);
  assert.equal(h.push(frame(0, 0, 0, 641, 1)), false);
  fill(h, 200); assert.equal(h.snapshot(200), null);
  assert.equal(h.snapshot(-1), null);
});

test('sampling remains bounded across long inference and high source frame rates', () => {
  const h = new HistoryBuffer(); fill(h, 10_000, 1);
  assert.ok(h.frames.length <= 64);
  assert.ok(h.frames.at(-1).mediaTime - h.frames[0].mediaTime <= 650);
  // At extreme frame rates the cap deliberately refuses a shorter fake window.
  assert.equal(h.snapshot(10_000), null);
  const normal = new HistoryBuffer(); fill(normal, 10_000);
  assert.deepEqual(normal.snapshot(normal.frames.at(-1).at).offsets, [-300, -200, -100]);
});

test('motion map includes movement and return between selected still panels', () => {
  const samples = [frame(0), frame(33, 100), frame(66), frame(100)];
  const map = motionPixels(samples, undefined, 2, 2);
  assert.equal(map[0], 255); assert.equal(map[3], 255);
  const still = motionPixels([frame(0), frame(100)], undefined, 2, 2);
  assert.equal(still[0], 0);
  const noise = motionPixels([frame(0, 10), frame(100, 15)], undefined, 2, 2);
  assert.equal(noise[0], 0);
});

test('crop excludes movement outside its region and keeps a quiet interior black', () => {
  const a = frame(0), b = frame(100); b.pixels.fill(255, 0, 4);
  const crop = { x: 0.5, y: 0, w: 0.5, h: 1 };
  assert.equal(motionPixels([a, b], crop, 1, 2)[0], 0);
  assert.equal(motionPixels([a, b], { x: 0, y: 0, w: 0.5, h: 1 }, 1, 2)[0], 255);
});

test('motion map letterboxes wide crops instead of stretching anatomy', () => {
  const out = motionPixels([frame(0, 0, 0, 4, 2), frame(100, 100, 100, 4, 2)], undefined, 4, 4);
  assert.equal(out[0], 0); assert.equal(out[4 * 4], 255); assert.equal(out[3 * 4 * 4], 0);
});

test('preview crop mapping accounts for CSS cover and mirrored input coordinates', () => {
  assert.deepEqual(cropPoint(0, 0, 400, 400, 1280, 720), { x: 0.21875, y: 0 });
  assert.deepEqual(cropPoint(400, 400, 400, 400, 1280, 720), { x: 0.78125, y: 1 });
  assert.deepEqual(cropRectangle({x: 0.8, y: 0.8}, {x: 0.2, y: 0.2}), {x: 0.2, y: 0.2, w: 0.6000000000000001, h: 0.6000000000000001});
  assert.equal(cropRectangle({x: 0.2, y: 0.2}, {x: 0.21, y: 0.21}), null);
  for (const c of [null, {}, {x: -1,y: 0,w: 1,h: 1}, {x: 0,y: 0,w: Infinity,h: 1}]) assert.equal(validCrop(c), false);
});

test('request uses two images with exact layout/window metadata and preserves saved questions', () => {
  const h = new HistoryBuffer(); fill(h, 400);
  const questions = { gesture: {type: 'noul', instructions: 'Custom question'} };
  const body = historyRequest(h.snapshot(400), questions, 'current', 'sheet', true);
  assert.deepEqual(body.images, ['current', 'sheet']); assert.equal(body.videos, undefined);
  assert.match(body.state, /-300, -200, -100 ms/); assert.match(body.state, /10 source frames/);
  assert.match(body.state, /not.*video frame/); assert.match(body.state, /does not encode direction/);
  assert.deepEqual(body.questions.speaking_motion, SPEAKING_MOTION);
  assert.equal(questions.speaking_motion, undefined); assert.equal(body.questions.gesture, questions.gesture);
  const own = {speaking_motion: {type: 'noul', instructions: 'Keep my schema'}};
  assert.equal(historyRequest(h.snapshot(400), own, 'a', 'b', false).questions, own);
});

 test('adding the motion question never exceeds the 64-question HTTP limit', () => {
   const full = Object.fromEntries(Array.from({length:64}, (_, i) => [`q${i}`, {type:'noul'}]));
   assert.equal(historyQuestions(full), full);
   const oneFree = {...full}; delete oneFree.q0;
   assert.equal(Object.keys(historyQuestions(oneFree)).length,64);
 });
