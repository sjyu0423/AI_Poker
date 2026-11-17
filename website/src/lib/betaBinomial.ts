/**
 * Educational Beta–Binomial learner for a single binary tendency.
 *
 * PokerMind's production model is hierarchical Dirichlet-categorical over
 * many public contexts (`docs/design/opponent_modeling.md`). This module is
 * the conjugate special case for one binary action, used only in the
 * interactive exhibit.
 */

export interface BetaParams {
  alpha: number;
  beta: number;
}

export interface CredibleInterval {
  lower: number;
  upper: number;
  level: number;
}

export interface BetaBinomialState {
  /** Prior mean in (0, 1). */
  priorMean: number;
  /** Prior strength s > 0: α₀ = s·μ, β₀ = s·(1 − μ). */
  priorStrength: number;
  successes: number;
  trials: number;
}

export interface BetaBinomialPosterior extends BetaParams {
  mean: number;
  /** Observed frequency, or null when there are no trials. */
  observedFrequency: number | null;
  variance: number;
  stdDev: number;
  interval90: CredibleInterval;
  /** Effective prior pseudo-counts. */
  prior: BetaParams;
}

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.984369654078561e-6,
  1.5056327351493116e-7,
] as const;

/** Natural log of Γ(z) for z > 0 (Lanczos approximation). */
export function logGamma(z: number): number {
  if (!(z > 0) || !Number.isFinite(z)) {
    throw new Error(`logGamma requires a positive finite argument, got ${z}`);
  }
  if (z < 0.5) {
    // Reflection: Γ(z)Γ(1−z) = π / sin(πz)
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  const x = z - 1;
  let a = LANCZOS[0];
  for (let i = 1; i < LANCZOS.length; i += 1) {
    a += LANCZOS[i] / (x + i);
  }
  const t = x + 7.5;
  return Math.log(Math.sqrt(2 * Math.PI)) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function betacf(a: number, b: number, x: number): number {
  const maxIter = 200;
  const eps = 3e-12;
  let am = 1;
  let bm = 1;
  let az = 1;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let bz = 1 - (qab * x) / qap;
  for (let m = 1; m <= maxIter; m += 1) {
    const em = m;
    const tem = em + em;
    let d = (em * (b - em) * x) / ((qam + tem) * (a + tem));
    const ap = az + d * am;
    const bp = bz + d * bm;
    d = (-(a + em) * (qab + em) * x) / ((a + tem) * (qap + tem));
    const app = ap + d * az;
    const bpp = bp + d * bz;
    const aold = az;
    am = ap / bpp;
    bm = bp / bpp;
    az = app / bpp;
    bz = 1;
    if (Math.abs(az - aold) < eps * Math.abs(az)) {
      return az;
    }
  }
  return az;
}

/** Regularized incomplete beta I_x(a, b) for a,b > 0 and x in [0, 1]. */
export function regularizedIncompleteBeta(x: number, a: number, b: number): number {
  if (!(a > 0 && b > 0)) {
    throw new Error(`regularizedIncompleteBeta requires positive a,b; got ${a}, ${b}`);
  }
  if (x <= 0) {
    return 0;
  }
  if (x >= 1) {
    return 1;
  }
  const bt =
    Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betacf(a, b, x)) / a;
  }
  return 1 - (bt * betacf(b, a, 1 - x)) / b;
}

/** Beta CDF at x. */
export function betaCdf(x: number, alpha: number, beta: number): number {
  return regularizedIncompleteBeta(x, alpha, beta);
}

/** Inverse Beta CDF (quantile) via bisection. */
export function betaQuantile(p: number, alpha: number, beta: number): number {
  if (!(p >= 0 && p <= 1)) {
    throw new Error(`betaQuantile probability must be in [0,1], got ${p}`);
  }
  if (p === 0) {
    return 0;
  }
  if (p === 1) {
    return 1;
  }
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 80; i += 1) {
    const mid = 0.5 * (lo + hi);
    if (betaCdf(mid, alpha, beta) < p) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

export function priorParams(priorMean: number, priorStrength: number): BetaParams {
  if (!(priorMean > 0 && priorMean < 1)) {
    throw new Error(`priorMean must be in (0,1), got ${priorMean}`);
  }
  if (!(priorStrength > 0)) {
    throw new Error(`priorStrength must be positive, got ${priorStrength}`);
  }
  return {
    alpha: priorStrength * priorMean,
    beta: priorStrength * (1 - priorMean),
  };
}

export function posteriorParams(state: BetaBinomialState): BetaParams {
  if (state.successes < 0 || state.trials < 0 || state.successes > state.trials) {
    throw new Error(
      `invalid counts successes=${state.successes} trials=${state.trials}`,
    );
  }
  if (!Number.isInteger(state.successes) || !Number.isInteger(state.trials)) {
    throw new Error("successes and trials must be integers");
  }
  const prior = priorParams(state.priorMean, state.priorStrength);
  return {
    alpha: prior.alpha + state.successes,
    beta: prior.beta + (state.trials - state.successes),
  };
}

export function centralCredibleInterval(
  alpha: number,
  beta: number,
  level = 0.9,
): CredibleInterval {
  if (!(level > 0 && level < 1)) {
    throw new Error(`credible level must be in (0,1), got ${level}`);
  }
  const tail = (1 - level) / 2;
  return {
    lower: betaQuantile(tail, alpha, beta),
    upper: betaQuantile(1 - tail, alpha, beta),
    level,
  };
}

export function betaBinomialPosterior(state: BetaBinomialState): BetaBinomialPosterior {
  const prior = priorParams(state.priorMean, state.priorStrength);
  const { alpha, beta } = posteriorParams(state);
  const mean = alpha / (alpha + beta);
  const variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
  return {
    alpha,
    beta,
    prior,
    mean,
    observedFrequency: state.trials === 0 ? null : state.successes / state.trials,
    variance,
    stdDev: Math.sqrt(variance),
    interval90: centralCredibleInterval(alpha, beta, 0.9),
  };
}
