import assert from "node:assert/strict";
import test from "node:test";
import { CaptureState, SYSTEMONE_URL, SINGLE_STATE, MOTION_STATE, inferenceTiming } from "../camera.mjs";

const questions = { talking: { type: "noul", instructions: "Talking?" } };
const frame = (data, at) => ({ data, at });

test("single-frame mode uses the System One image request and preserves edited questions", () => {
  const capture = new CaptureState();
  assert.equal(SYSTEMONE_URL, "/v1/systemone");
  assert.deepEqual(capture.prepare(frame("current", 100), questions), {
    model: "clef-flash", state: SINGLE_STATE, questions, images: ["current"]
  });
  assert.equal(capture.waitMs(101), 0);
});

test("motion mode sends one two-frame video in chronological order with the measured frame rate", () => {
  const capture = new CaptureState();
  capture.setMotion(true);
  assert.equal(capture.prepare(frame("earlier", 1000), questions), null);
  assert.equal(capture.waitMs(1100), 150);
  assert.equal(capture.prepare(frame("too-close", 1100), questions), null);
  assert.deepEqual(capture.prepare(frame("later", 1250), questions), {
    model: "clef-flash", state: MOTION_STATE, questions,
    videos: [{ frames: ["earlier", "later"], fps: 4 }]
  });
  assert.deepEqual(capture.prepare(frame("next", 1750), questions).videos,
    [{ frames: ["later", "next"], fps: 2 }]);
  assert.equal(questions.talking.instructions, "Talking?");
});

test("pause, schema edits, and mode changes reset history and invalidate an old response", () => {
  const capture = new CaptureState();
  capture.setMotion(true);
  capture.prepare(frame("old", 1000), questions);
  const inFlightGeneration = capture.generation;
  capture.reset();
  assert.notEqual(capture.generation, inFlightGeneration);
  assert.equal(capture.prepare(frame("fresh", 3000), questions), null);
  capture.setMotion(false);
  assert.deepEqual(capture.prepare(frame("single", 3250), questions).images, ["single"]);
  capture.setMotion(true);
  assert.equal(capture.prepare(frame("new-start", 3500), questions), null);
});

test("stale or reversed frame times cannot be sent as a motion pair", () => {
  const capture = new CaptureState();
  capture.setMotion(true);
  capture.prepare(frame("old", 1000), questions);
  assert.equal(capture.prepare(frame("stale", 4000), questions), null);
  assert.equal(capture.prepare(frame("reversed", 3900), questions), null);
  assert.deepEqual(capture.prepare(frame("fresh", 4150), questions).videos,
    [{ frames: ["reversed", "fresh"], fps: 4 }]);
});

test("model timing remains separate from missing or malformed request timing", () => {
  assert.equal(inferenceTiming("inference;dur=251.2"), 251.2);
  assert.equal(inferenceTiming("queue;dur=10, inference;dur=120"), 120);
  for (const header of [null, "", "queue;dur=4", "inference;dur=-1", "inference;dur=NaN", "inference;dur=1.2.3"]) {
    assert.equal(inferenceTiming(header), null);
  }
});
