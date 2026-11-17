import { links } from "@/content/site";
import { M1_SOURCE } from "@/content/m1CfrCheckpoints";

export const cfrPageMeta = {
  title: "Teaching an AI to Regret",
  description:
    "An interactive exhibit of Counterfactual Regret Minimization—the game-theoretic foundation behind PokerMind’s near-equilibrium strategies.",
  path: "/research/cfr/",
} as const;

export const cfrNav = [
  { label: "Idea", href: "#intro" },
  { label: "Regret matching", href: "#regret-matching" },
  { label: "Kuhn Poker", href: "#kuhn" },
  { label: "Training", href: "#training" },
  { label: "Average strategy", href: "#average" },
  { label: "Verification", href: "#verification" },
] as const;

export const cfrIntro = {
  kicker: "Research exhibit · M1",
  headline: "Teaching an AI to Regret.",
  lede:
    "Counterfactual Regret Minimization (CFR) is how PokerMind learns a strategy that is hard to exploit—before it ever tries to model a particular opponent.",
  centralQuestion: "What if another decision would have produced a better expected outcome?",
  plain:
    "At every decision point, CFR asks how much better each alternative action would have been in expectation, given what the rest of the game tree is doing. It accumulates those “counterfactual regrets,” then shifts probability toward actions it regrets not taking. Over many iterations, the average of those strategies approaches a Nash equilibrium.",
  clarify:
    "CFR evaluates counterfactual expected utility across game-tree situations. It is not merely regretting the chips won or lost on one sampled hand.",
} as const;

export const regretMatchingCopy = {
  title: "Regret matching, live",
  summary:
    "This two-action toy shows the exact formula PokerMind uses at each information set. Drag the cumulative regrets and watch the next strategy appear. It is an isolated mathematical example—not a full CFR poker solver.",
  formula:
    "R⁺(a) = max(R(a), 0);  σ(a) = R⁺(a) / Σ R⁺  if the sum is positive, else uniform.",
  labels: {
    actionA: "Check",
    actionB: "Bet",
  },
  defaults: {
    regretA: 2.5,
    regretB: 1.0,
  },
  notes: [
    "Positive regret → that action gets probability proportional to how positive it is.",
    "Negative regret is clipped to zero: it never receives probability from matching.",
    "If every regret is ≤ 0, the strategy falls back to uniform (here 50 / 50).",
    "Equal positive regrets → equal probabilities.",
  ],
} as const;

export const kuhnCopy = {
  title: "Kuhn Poker: imperfect information in miniature",
  summary:
    "Before Leduc, PokerMind’s M1 baseline trained on Kuhn Poker—a three-card game small enough that every strategy can be evaluated exactly.",
  cards: [
    { rank: "J", name: "Jack", note: "Weakest" },
    { rank: "Q", name: "Queen", note: "Middle" },
    { rank: "K", name: "King", note: "Strongest" },
  ],
  points: [
    {
      title: "Private cards",
      detail:
        "Each player is dealt one private card. You never see the opponent’s card unless the hand reaches showdown.",
    },
    {
      title: "Betting",
      detail:
        "After the deal, players may check or bet; facing a bet they may fold or call. Antes and bets are one chip each.",
    },
    {
      title: "Information sets",
      detail:
        "A decision depends only on your card and the public betting line—for example Q|xb: “I hold Queen; betting so far is check, then bet.” The opponent’s card is not part of the key.",
    },
    {
      title: "Why mixing matters",
      detail:
        "Pure strategies are readable. In equilibrium, player 1 bluffs with Jack after a check exactly 1/3 of the time so a Queen is indifferent between calling and folding.",
    },
  ],
  treeCaption:
    "Simplified view of one information set: the acting player cannot tell which of the two hidden opponent cards is present.",
} as const;

export const trainingCopy = {
  title: "CFR training progression",
  summary:
    "These checkpoints come from the real M1 Kuhn run (seed 0). Select an iteration to see the exact exploitability of the average strategy and how a few information-set probabilities evolved.",
  metric:
    "Exploitability = NashConv / 2: how many chips per hand a perfect counter-strategy can win against this profile, on average.",
  sourceNote: `Source: ${M1_SOURCE.artifact} via \`${M1_SOURCE.command}\`.`,
} as const;

export const averageCopy = {
  title: "Why the average strategy wins",
  summary:
    "The strategy that regret matching outputs at a single iteration often keeps oscillating. The strategy PokerMind exports, certifies, and plays is the reach-weighted average of those iterates.",
  distinctions: [
    {
      title: "Current strategy",
      detail:
        "σᵗ — what regret matching recommends right now from cumulative regrets. It need not converge on its own.",
    },
    {
      title: "Cumulative regret",
      detail:
        "Rᵀ(I, a) — running total of counterfactual regrets for not playing action a at information set I.",
    },
    {
      title: "Average strategy",
      detail:
        "σ̄ᵀ — normalized sum of current strategies, weighted by how often the player reached each information set. In two-player zero-sum games this average approaches Nash equilibrium.",
    },
  ],
  mathTitle: "Optional mathematics",
  math: [
    "Instantaneous counterfactual regret: rᵗ(I, a) = vᵢ(σᵗ, I, a) − vᵢ(σᵗ, I).",
    "Cumulative regret: Rᵀ(I, a) = Σ_{t=1..T} rᵗ(I, a).",
    "Average strategy (reach-weighted): σ̄ᵀ(I, a) ∝ Σ_t πᵢ^{σᵗ}(I) · σᵗ(I, a).",
    "Worst-case average regret shrinks like O(1/√T); Kuhn’s measured curve is better in practice.",
  ],
} as const;

export const verificationCopy = {
  title: "How PokerMind verified CFR",
  summary:
    "Training and evaluation stay independent. The exact evaluator never imports the learning code, so it is a separate check on CFR—not a circular report.",
  items: [
    {
      title: "Exact best response",
      detail:
        "For each information set, the evaluator picks the action that maximizes expected utility against the fixed opponent strategy, respecting imperfect information.",
    },
    {
      title: "NashConv and exploitability",
      detail:
        "NashConv sums each player’s gain from best-responding. Exploitability is NashConv / 2, matching OpenSpiel’s convention, in chips per hand.",
    },
    {
      title: "Known game value",
      detail:
        "Kuhn’s value is −1/18 for the first player. After 50,000 iterations the M1 average strategy matched that value to about 4×10⁻⁷.",
    },
    {
      title: "OpenSpiel cross-checks",
      detail:
        "Optional tests compare PokerMind’s simultaneous-update CFR against OpenSpiel’s base solver configured the same way, agreeing to rounding error.",
    },
    {
      title: "Deterministic digests",
      detail:
        "Every exported strategy file carries a SHA-256 digest. Checkpoints in the M1 metrics record those digests so runs can be compared bit for bit.",
    },
  ],
  links: [
    { label: "CFR concept note", href: links.cfr },
    { label: "Learning code (cfr.py)", href: `${links.github}/blob/main/src/pokermind/learning/cfr.py` },
    { label: "Exact evaluator", href: `${links.github}/blob/main/src/pokermind/evaluation/exact.py` },
    { label: "OpenSpiel cross-check tests", href: `${links.github}/blob/main/tests/test_openspiel_crosscheck.py` },
  ],
} as const;

export const continueCopy = {
  headline: "CFR taught PokerMind how to play robustly. But could it learn how a particular opponent plays?",
  body:
    "The next exhibit shows how PokerMind updates Bayesian beliefs from public betting actions—and why hidden cards still bound what can be learned.",
  primary: { label: "Opponent modeling exhibit", href: links.opponentModelingPage },
  secondary: { label: "Back to homepage", href: "/" },
  tertiary: { label: "Read the CFR note", href: links.cfr },
} as const;
