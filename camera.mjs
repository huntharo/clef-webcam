// Pure capture state shared by the browser loop and its tests.
export const SYSTEMONE_URL = "/v1/systemone";
export const SINGLE_STATE = "A live webcam frame from a laptop.";
export const MOTION_STATE = "Two chronological webcam frames, earlier then later. Look for motion, especially mouth/lip changes consistent with talking.";

export class CaptureState {
  constructor() {
    this.motion = false;
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
    if (!this.motion) {
      return { model: "clef-flash", state: SINGLE_STATE, questions, images: [frame.data] };
    }
    const earlier = this.previous;
    if (!earlier || frame.at <= earlier.at || frame.at - earlier.at > 2000) {
      this.previous = frame;
      return null;
    }
    const gap = frame.at - earlier.at;
    if (gap < 250) return null;
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
