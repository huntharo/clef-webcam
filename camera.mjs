// Capture helpers shared by the browser loop and its tests.
export const SYSTEMONE_URL = "/v1/systemone";
export const SINGLE_STATE = "A live webcam frame from a laptop.";
export const MOTION_STATE = "Two chronological webcam frames, earlier then later. Look for motion, especially mouth/lip changes consistent with talking.";

// An unset preference uses two frames; preserve the stored single-frame opt-out.
export const motionPreference = stored => stored !== "false";

export function captureFrame(video, draw, timeoutMs = 1000) {
  // Sample a newly presented source frame, not the interpolated playback clock.
  return new Promise((resolve, reject) => {
    const callback = video.requestVideoFrameCallback((at, metadata) => {
      clearTimeout(timeout);
      try {
        resolve({ data: draw(), at, mediaTime: metadata.mediaTime * 1000 });
      } catch (error) { reject(error); }
    });
    // A stopped source must not leave the loop stuck across pause/mode changes.
    const timeout = setTimeout(() => {
      video.cancelVideoFrameCallback(callback);
      resolve(null);
    }, timeoutMs);
  });
}

export class CaptureState {
  constructor(motion = true) {
    this.motion = !!motion;
    this.generation = 0;
    this.previous = null;
  }

  reset() {
    this.previous = null;
    this.generation++;
  }

  setMotion(enabled) {
    this.motion = !!enabled;
    this.reset();
  }

  waitMs(now) {
    return this.motion && this.previous ? Math.max(0, 250 - (now - this.previous.at)) : 0;
  }

  prepare(frame, questions) {
    const earlier = this.previous;
    // mediaTime identifies the source frame, independently of when we copy it.
    // A stalled camera must not become two frames by copying it again later.
    if (earlier && frame.mediaTime === earlier.mediaTime) return null;
    if (!this.motion) {
      this.previous = frame;
      return { model: "clef-flash", state: SINGLE_STATE, questions, images: [frame.data] };
    }
    if (!earlier || frame.at <= earlier.at || frame.at - earlier.at > 2000 ||
        frame.mediaTime <= earlier.mediaTime || frame.mediaTime - earlier.mediaTime > 2000) {
      this.previous = frame;
      return null;
    }
    const gap = frame.mediaTime - earlier.mediaTime;
    if (gap < 250 || frame.at - earlier.at < 250) return null;
    this.previous = frame;
    return { model: "clef-flash", state: MOTION_STATE, questions,
             videos: [{ frames: [earlier.data, frame.data], fps: 1000 / gap }] };
  }
}

export function inferenceTiming(header) {
  const match = (header || "").match(/(?:^|,)\s*inference\s*;\s*dur=([\d.]+)/);
  if (!match) return null;
  const value = Number(match[1]);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

export const SPEAKING_MOTION = {
  type: "choice",
  instructions: "Is there visible speech-like lip or jaw articulation across the timestamped history panels and current photo? Judge mouth changes, not earbuds, hand gestures, head movement or an open mouth alone. Use uncertain when the mouth is too small, obscured, or the evidence is ambiguous.",
  criteria: {
    speaking_motion: "Visible lip or jaw articulation changes consistent with speaking.",
    no_speaking_motion: "Mouth visible without speech-like articulation changes.",
    uncertain: "Mouth not sufficiently visible or motion cannot be identified."
  }
};

// Source frames continue arriving while the single inference request runs.
// Keep memory bounded even for a high-fps camera or a slow model response.
export class HistoryBuffer {
  constructor() { this.reset(); }
  reset() { this.frames = []; }
  push(frame) {
    if (![frame.at, frame.mediaTime, frame.width, frame.height].every(Number.isFinite) ||
        !Number.isInteger(frame.width) || !Number.isInteger(frame.height) ||
        frame.width < 1 || frame.height < 1 || frame.width > 640 || frame.height > 640 ||
        frame.pixels.length !== frame.width * frame.height * 4) return false;
    const last = this.frames.at(-1);
    if (last && frame.mediaTime === last.mediaTime) return false;
    if (last && (frame.mediaTime < last.mediaTime || frame.at <= last.at ||
        frame.mediaTime - last.mediaTime > 150 || frame.at - last.at > 150 ||
        frame.width !== last.width || frame.height !== last.height)) this.reset();
    this.frames.push(frame);
    this.frames = this.frames.filter(f => frame.mediaTime - f.mediaTime <= 650).slice(-64);
    return true;
  }
  snapshot(now) {
    const current = this.frames.at(-1);
    if (!current || now - current.at > 150 || now < current.at) return null;
    const history = [300, 200, 100].map(offset => {
      const target = current.mediaTime - offset;
      return this.frames.reduce((best, f) => Math.abs(f.mediaTime - target) < Math.abs(best.mediaTime - target) ? f : best);
    });
    if (new Set(history).size !== 3 || history.some((f, i) =>
        Math.abs(current.mediaTime - f.mediaTime - [300, 200, 100][i]) > 45)) return null;
    const samples = this.frames.filter(f => f.mediaTime >= history[0].mediaTime);
    return { current, history, samples,
      offsets: history.map(f => Math.round(f.mediaTime - current.mediaTime)) };
  }
}

export function validCrop(value) {
  return !!value && ["x", "y", "w", "h"].every(k => Number.isFinite(value[k])) &&
    value.x >= 0 && value.y >= 0 && value.w >= 0.05 && value.h >= 0.05 &&
    value.x + value.w <= 1.000001 && value.y + value.h <= 1.000001;
}

export function cropPoint(x, y, boxWidth, boxHeight, videoWidth, videoHeight) {
  const scale = Math.max(boxWidth / videoWidth, boxHeight / videoHeight);
  const w = videoWidth * scale, h = videoHeight * scale;
  return { x: Math.max(0, Math.min(1, (x - (boxWidth - w) / 2) / w)),
           y: Math.max(0, Math.min(1, (y - (boxHeight - h) / 2) / h)) };
}

export function cropRectangle(a, b) {
  const crop = { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y),
                 w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) };
  return validCrop(crop) ? crop : null;
}

const FULL_FRAME = { x: 0, y: 0, w: 1, h: 1 };

function fitTile(source, crop, width, height) {
  const scale = Math.min(width / (source.width * crop.w), height / (source.height * crop.h));
  const w = source.width * crop.w * scale, h = source.height * crop.h * scale;
  return { x: (width - w) / 2, y: (height - h) / 2, w, h };
}

// Maximum adjacent-frame RGB change, not just an endpoint subtraction. A mouth
// that moves and returns between selected history panels can leave evidence.
// Fixed noise floor/gain avoids turning every nearly static frame into white.
export function motionPixels(samples, crop = FULL_FRAME, width = 168, height = 148) {
  if (!validCrop(crop) || !samples.length) throw new Error("invalid motion window");
  const output = new Uint8ClampedArray(width * height * 4);
  const source = samples[0];
  const fit = fitTile(source, crop, width, height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const o = (y * width + x) * 4;
    output[o + 3] = 255;
    if (x + 0.5 < fit.x || x + 0.5 >= fit.x + fit.w || y + 0.5 < fit.y || y + 0.5 >= fit.y + fit.h) continue;
    const sx = Math.min(source.width - 1, Math.floor((crop.x + (x + 0.5 - fit.x) / fit.w * crop.w) * source.width));
    const sy = Math.min(source.height - 1, Math.floor((crop.y + (y + 0.5 - fit.y) / fit.h * crop.h) * source.height));
    const p = (sy * source.width + sx) * 4;
    let change = 0;
    for (let i = 1; i < samples.length; i++) {
      const before = samples[i - 1].pixels, after = samples[i].pixels;
      const delta = (Math.abs(after[p] - before[p]) + Math.abs(after[p + 1] - before[p + 1]) +
                     Math.abs(after[p + 2] - before[p + 2])) / 3;
      change = Math.max(change, Math.max(0, delta - 6) * 3);
    }
    output[o] = output[o + 1] = output[o + 2] = Math.min(255, change);
  }
  return output;
}

export function historyQuestions(questions) {
  return Object.hasOwn(questions, "speaking_motion") || Object.keys(questions).length >= 64
    ? questions : { ...questions, speaking_motion: SPEAKING_MOTION };
}

export function historyRequest(snapshot, questions, current, sheet, cropped) {
  return {
    model: "clef-flash", images: [current, sheet],
    state: `Image 1 is the current full webcam photo (0 ms). Image 2 is a 2x2 diagnostic contact sheet, not a second scene or a video frame. Read top-left, top-right, bottom-left as chronological ${cropped ? "cropped" : "full-scene"} camera history at ${snapshot.offsets.join(", ")} ms relative to image 1. Bottom-right is a motion map over ${snapshot.samples.length} source frames from ${snapshot.offsets[0]} to 0 ms: black means no detected change, brighter means more adjacent-frame RGB change. This map is not a person or a natural photograph and does not encode direction or prove speech. For talking, require visible lip/jaw articulation across the natural panels; head motion, lighting, gestures and earbuds are not evidence of speech. Use uncertain when the mouth is too small or unclear.`,
    questions: historyQuestions(questions)
  };
}

export function contactSheet(snapshot, crop = FULL_FRAME, createCanvas = () => document.createElement("canvas")) {
  const source = createCanvas(), sheet = createCanvas(), current = createCanvas();
  source.width = snapshot.current.width; source.height = snapshot.current.height;
  const sourceContext = source.getContext("2d");
  const put = frame => {
    const image = sourceContext.createImageData(frame.width, frame.height);
    image.data.set(frame.pixels); sourceContext.putImageData(image, 0, 0);
  };
  sheet.width = sheet.height = 336;
  const ctx = sheet.getContext("2d");
  ctx.fillStyle = "#111"; ctx.fillRect(0, 0, 336, 336);
  snapshot.history.forEach((frame, i) => {
    put(frame);
    const fit = fitTile(frame, crop, 168, 148);
    ctx.drawImage(source, crop.x * frame.width, crop.y * frame.height,
      crop.w * frame.width, crop.h * frame.height,
      (i % 2) * 168 + fit.x, Math.floor(i / 2) * 168 + 20 + fit.y, fit.w, fit.h);
  });
  const map = ctx.createImageData(168, 148);
  map.data.set(motionPixels(snapshot.samples, crop));
  ctx.putImageData(map, 168, 188);
  ctx.fillStyle = "#fff"; ctx.font = "bold 14px sans-serif";
  [...snapshot.offsets.map(ms => `${ms} ms`), "motion map"].forEach((label, i) =>
    ctx.fillText(label, (i % 2) * 168 + 6, Math.floor(i / 2) * 168 + 15));
  put(snapshot.current);
  current.width = 448; current.height = Math.round(448 * source.height / source.width);
  current.getContext("2d").drawImage(source, 0, 0, current.width, current.height);
  return { current: current.toDataURL("image/jpeg", 0.85),
           sheet: sheet.toDataURL("image/jpeg", 0.9), canvas: sheet };
}
