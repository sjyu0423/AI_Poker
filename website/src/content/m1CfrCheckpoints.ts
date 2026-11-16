/**
 * Frozen subset of the M1 Kuhn vanilla-CFR run.
 *
 * Source: local reproducible artifact `runs/m1-cfr/metrics.json`
 * (command: `pokermind train-cfr --output-dir runs/m1-cfr --seed 0`).
 * Documented summary: docs/concepts/cfr.md, docs/overview.md
 * (50,000 iterations → exploitability ≈ 0.0010 chips/hand).
 *
 * Strategy probabilities below are the *average* strategies recorded at
 * each checkpoint—not invented illustration data.
 */

export interface ActionProbabilities {
  readonly [action: string]: number;
}

export interface M1Checkpoint {
  readonly iteration: number;
  /** Exact exploitability = NashConv / 2, chips per hand. */
  readonly exploitability: number;
  readonly nashConv: number;
  /** Exact self-play EV for seat 0 under the average strategy. */
  readonly selfPlayValueSeat0: number;
  /** self_play_value_seat0 − (−1/18). */
  readonly gameValueError: number;
  readonly strategyDigestPrefix: string;
  /**
   * Selected information-set probabilities from the average strategy.
   * Action labels: x=check, b=bet, f=fold, c=call.
   */
  readonly selectedAverageStrategy: Readonly<Record<string, ActionProbabilities>>;
}

export const M1_SOURCE = {
  experiment: "kuhn-vanilla-cfr",
  artifact: "runs/m1-cfr/metrics.json",
  command: "pokermind train-cfr --output-dir runs/m1-cfr --seed 0",
  docs: "docs/concepts/cfr.md",
  algorithm: "vanilla CFR, simultaneous updates, full traversal",
  exploitabilityConvention: "NashConv / 2, chips per hand",
  knownGameValue: -1 / 18,
} as const;

/** Info sets highlighted in the training timeline UI. */
export const M1_FOCUS_INFOSETS = [
  {
    key: "J|",
    label: "P0 opens with J",
    actions: [
      { code: "x", name: "Check" },
      { code: "b", name: "Bet" },
    ],
  },
  {
    key: "J|x",
    label: "P1 after check, holds J (bluff spot)",
    actions: [
      { code: "x", name: "Check" },
      { code: "b", name: "Bet" },
    ],
  },
  {
    key: "K|b",
    label: "P1 faces bet with K",
    actions: [
      { code: "f", name: "Fold" },
      { code: "c", name: "Call" },
    ],
  },
  {
    key: "Q|b",
    label: "P1 faces bet with Q",
    actions: [
      { code: "f", name: "Fold" },
      { code: "c", name: "Call" },
    ],
  },
] as const;

/**
 * All recorded M1 checkpoints. The interactive timeline surfaces the
 * visitor-facing subset in `M1_TIMELINE_ITERATIONS`.
 */
export const M1_CHECKPOINTS: readonly M1Checkpoint[] = [
  {
    iteration: 1,
    exploitability: 0.45833333333333326,
    nashConv: 0.9166666666666665,
    selfPlayValueSeat0: 0.125,
    gameValueError: 0.18055555555555555,
    strategyDigestPrefix: "f3c147c465420dc4",
    selectedAverageStrategy: {
      "J|": { x: 0.5, b: 0.5 },
      "J|x": { x: 0.5, b: 0.5 },
      "K|b": { f: 0.5, c: 0.5 },
      "Q|b": { f: 0.5, c: 0.5 },
    },
  },
  {
    iteration: 10,
    exploitability: 0.09620850020140503,
    nashConv: 0.19241700040281007,
    selfPlayValueSeat0: -0.03519276118701568,
    gameValueError: 0.020362794368539876,
    strategyDigestPrefix: "a37b0f93768886cf",
    selectedAverageStrategy: {
      "J|": { x: 0.8125, b: 0.1875 },
      "J|x": { x: 0.7, b: 0.3 },
      "K|b": { f: 0.05, c: 0.95 },
      "Q|b": { f: 0.10799799758313955, c: 0.8920020024168605 },
    },
  },
  {
    iteration: 100,
    exploitability: 0.02567473584694787,
    nashConv: 0.05134947169389574,
    selfPlayValueSeat0: -0.05598721160991619,
    gameValueError: -0.00043165605436063653,
    strategyDigestPrefix: "312a75d531d40747",
    selectedAverageStrategy: {
      "J|": { x: 0.8057773405186036, b: 0.19422265948139633 },
      "J|x": { x: 0.693502261182714, b: 0.30649773881728604 },
      "K|b": { f: 0.005, c: 0.995 },
      "Q|b": { f: 0.5637263198127731, c: 0.436273680187227 },
    },
  },
  {
    iteration: 500,
    exploitability: 0.010681129492920681,
    nashConv: 0.021362258985841362,
    selfPlayValueSeat0: -0.055260296874070525,
    gameValueError: 0.00029525868148502776,
    strategyDigestPrefix: "670ebb4a8ad55aa2",
    selectedAverageStrategy: {
      "J|": { x: 0.8293028225700023, b: 0.1706971774299977 },
      "J|x": { x: 0.6548730641696836, b: 0.3451269358303165 },
      "K|b": { f: 0.001, c: 0.999 },
      "Q|b": { f: 0.6541035443988696, c: 0.34589645560113036 },
    },
  },
  {
    iteration: 1000,
    exploitability: 0.007269106408563809,
    nashConv: 0.014538212817127617,
    selfPlayValueSeat0: -0.05555721950490657,
    gameValueError: -0.0000016639493510203773,
    strategyDigestPrefix: "5bb370d76d1fb0af",
    selectedAverageStrategy: {
      "J|": { x: 0.7989913289431153, b: 0.2010086710568847 },
      "J|x": { x: 0.6723111661398348, b: 0.3276888338601652 },
      "K|b": { f: 0.0005, c: 0.9995 },
      "Q|b": { f: 0.6312096928634128, c: 0.36879030713658717 },
    },
  },
  {
    iteration: 5000,
    exploitability: 0.003302448056612603,
    nashConv: 0.006604896113225206,
    selfPlayValueSeat0: -0.055506400503874585,
    gameValueError: 0.00004915505168096718,
    strategyDigestPrefix: "ef0a181b981a8536",
    selectedAverageStrategy: {
      "J|": { x: 0.7906220745016843, b: 0.20937792549831571 },
      "J|x": { x: 0.6620431330749141, b: 0.33795686692508586 },
      "K|b": { f: 0.0001, c: 0.9999 },
      "Q|b": { f: 0.6638180717341825, c: 0.3361819282658175 },
    },
  },
  {
    iteration: 10000,
    exploitability: 0.002317786314974535,
    nashConv: 0.00463557262994907,
    selfPlayValueSeat0: -0.05554639582927171,
    gameValueError: 0.000009159726283840264,
    strategyDigestPrefix: "fb8dccc01a470db4",
    selectedAverageStrategy: {
      "J|": { x: 0.7922453071465074, b: 0.20775469285349268 },
      "J|x": { x: 0.6654552724682011, b: 0.33454472753179887 },
      "K|b": { f: 0.00005, c: 0.99995 },
      "Q|b": { f: 0.6607153696327401, c: 0.3392846303672598 },
    },
  },
  {
    iteration: 50000,
    exploitability: 0.0009988288043084083,
    nashConv: 0.0019976576086168166,
    selfPlayValueSeat0: -0.055555956950980444,
    gameValueError: -4.013954248913154e-7,
    strategyDigestPrefix: "b8217a0c74657df7",
    selectedAverageStrategy: {
      "J|": { x: 0.7927040190707378, b: 0.2072959809292622 },
      "J|x": { x: 0.6673260936337511, b: 0.33267390636624894 },
      "K|b": { f: 0.00001, c: 0.99999 },
      "Q|b": { f: 0.6621144409776915, c: 0.33788555902230855 },
    },
  },
] as const;

/** Checkpoints highlighted in the interactive timeline UI. */
export const M1_TIMELINE_ITERATIONS = [1, 10, 100, 1000, 10000, 50000] as const;

export function getM1Checkpoint(iteration: number): M1Checkpoint {
  const found = M1_CHECKPOINTS.find((row) => row.iteration === iteration);
  if (!found) {
    throw new Error(`No M1 checkpoint for iteration ${iteration}`);
  }
  return found;
}

export function formatExploitability(value: number): string {
  if (value >= 0.1) {
    return value.toFixed(4);
  }
  if (value >= 0.01) {
    return value.toFixed(4);
  }
  return value.toFixed(4);
}
