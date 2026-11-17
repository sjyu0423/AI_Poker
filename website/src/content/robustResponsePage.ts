import { links } from "@/content/site";
import { M7_SOURCE, RNR_P_CONCEPT } from "@/content/m7RobustResponse";

export const robustPageMeta = {
  title: "Learning to Exploit. Knowing When to Stop.",
  description:
    "An interactive exhibit of Restricted Nash Response—how PokerMind turns opponent predictions into exploitability-constrained adaptive strategies.",
  path: "/research/robust-response/",
} as const;

export const robustNav = [
  { label: "Idea", href: "#intro" },
  { label: "Danger", href: "#danger" },
  { label: "How RNR works", href: "#rnr" },
  { label: "Frontier", href: "#frontier" },
  { label: "Why RNR", href: "#why" },
  { label: "Personalization", href: "#personal" },
  { label: "Reliability", href: "#reliability" },
  { label: "Deep dive", href: "#deep" },
] as const;

export const robustIntro = {
  kicker: "Research exhibit · M7A / M7B",
  headline: "Learning to Exploit. Knowing When to Stop.",
  lede:
    "A strategy that exploits an opponent’s mistakes may itself become vulnerable. PokerMind must balance winning more against an estimated opponent with protecting itself against an optimal counter-strategy.",
  plain:
    "Exploitability is how many chips per hand a perfect adversary could win against your strategy. PokerMind starts from a near-unexploitable CFR baseline, then searches for adaptive replies that stay inside a certified exploitability budget.",
} as const;

export const dangerCopy = {
  title: "Why naive exploitation is dangerous",
  summary:
    "Chasing every guessed weakness can leave you open to anyone who plays differently from your model. Mixing a little best-response into CFR helps only a little; Restricted Nash Response rewrites play where exploitation is cheap in exploitability.",
  cfrNote: "Exact: CFR teacher exploitability 0.0030 chips/hand.",
  brNote:
    "Measured (M6): unrestricted best responses to estimated opponents sat around 1–4 chips/hand of exploitability—outside any safety budget PokerMind ships.",
  mixNote:
    "Measured (M7A, held-out, 100 hands, δ = 0.10): M6 mixing gained only +0.016 chips/hand while staying ≤ 0.1030 exploitability.",
  rnrNote:
    "Measured (same setting): RNR gained +0.138 chips/hand at the same exact budget.",
} as const;

export const rnrCopy = {
  title: "How Restricted Nash Response works",
  summary:
    "Before each hand, a hidden coin decides the opponent’s mode. PokerMind never sees the coin—it must prepare for both realities at once.",
  pDefinition: RNR_P_CONCEPT.definition,
  pNotBudget: RNR_P_CONCEPT.notBudget,
  extremes: RNR_P_CONCEPT.extremes,
  dialDisclaimer:
    "This dial teaches the mathematical construction only. It does not invent payoff or exploitability numbers for arbitrary p. Recorded results appear in the frontier section with their documented budgets.",
} as const;

export const frontierCopy = {
  title: "Interactive exploitability frontier",
  summary:
    "Select a documented exploitability budget to display the recorded M7A gains over fixed CFR after 100 hands. Strategies were compared at the same exact budgets—this control only switches among measured rows.",
  unit: "Gain units: chips per hand vs the true opponent (seats alternate).",
  sourceNote: `Source: ${M7_SOURCE.m7aDocs} (${M7_SOURCE.m7aArtifact}).`,
} as const;

export const whyCopy = {
  title: "Why PokerMind chose RNR",
  bullets: [
    {
      title: "A better frontier than mixing",
      body: "Mixing CFR with a best response traces a chord under the (exploitability, payoff) frontier. RNR changes play information-set by information-set, so small budgets buy far more gain (+0.138 vs +0.016 at δ = 0.10, held-out, 100 hands).",
    },
    {
      title: "DBR did not win here",
      body: "Data-Biased Response beat mixing but lost to RNR. In Leduc most card-conditional counts are inferred and sparse; DBR’s per-set freedom discards the population prior that RNR uses.",
    },
    {
      title: "Exact certification, not a p-promise",
      body: "Neither p nor P_max guarantees a budget a priori. PokerMind measures every candidate’s exploitability exactly and keeps only those ≤ expl(CFR) + δ, then re-certifies the pick.",
    },
    {
      title: "Model uncertainty still matters",
      body: "Exact exploitability bounds the worst-case loss against any opponent. Statistical uncertainty about opponent behavior is separate—and is what M7B’s calibrated selection rule addresses.",
    },
  ],
} as const;

export const personalCopy = {
  title: "Does personalization help?",
  summary:
    "Against familiar held-out styles, most of RNR’s gain is already available from the population prior at zero hands. Personal observations still add a measurable increment with enough data—and matter far more against out-of-population opponents.",
  caveat:
    "All figures use simulated Leduc opponents. No real-human validation is claimed.",
} as const;

export const reliabilityCopy = {
  title: "Reliability improvements (M7B)",
  summary:
    "The frozen research configuration scores RNR candidates with a calibrated 95% lower confidence bound on posterior gain, then falls back to CFR if none look safely positive.",
  tradeoff:
    "PokerMind sacrificed a small amount of average gain to reduce the frequency of harmful adaptations. These are measured experimental outcomes, not universal guarantees.",
} as const;

export const deepDiveCopy = {
  title: "Technical deep dive",
  summary: "Optional mathematical and engineering detail for reviewers.",
  items: [
    {
      title: "Modified game",
      body: "A chance node (visible only to the opponent) selects restricted mode with probability p—playing σ̂—or free mode with probability 1 − p. Hero’s information sets omit that coin, so the equilibrium hedges both branches.",
    },
    {
      title: "Solving",
      body: "Batched CFR+ (500 iterations in the freeze) solves a grid of p values. Approximate equilibria: the RNR-game gap is ≤ ~0.0031 at 500 iterations; conclusions at δ ≥ 0.02 are unchanged at 1,000–2,000 iterations.",
    },
    {
      title: "Exact evaluation",
      body: "Exploitability = NashConv / 2 in the original Leduc game via the M3 evaluator. Modified-game values are never trusted for selection comparisons.",
    },
    {
      title: "Frozen selection rule",
      body: "Keep candidates with exact exploitability ≤ 0.0030 + δ (δ = 0.1 frozen). Score by posterior mean gain − 1.645 × 1.25 × sd with temperature τ = 0.5. Play the best positive score, else CFR; re-certify the pick.",
    },
  ],
  implLink: {
    label: "Implementation · src/pokermind/robust/",
    href: "https://github.com/sjyu0423/AI_Poker/tree/main/src/pokermind/robust",
  },
  designLink: { label: "Design note", href: links.m7a },
  freezeLink: { label: "Research freeze", href: links.researchFreeze },
} as const;

export const closingCopy = {
  headline: "The full PokerMind stack",
  body: "The frozen system combines a CFR baseline, a Bayesian opponent model, hidden-card-aware strategy reconstruction, Restricted Nash Response, and exact exploitability certification.",
  steps: [
    "CFR baseline",
    "Bayesian opponent model",
    "Hidden-card-aware reconstruction",
    "Restricted Nash Response",
    "Exact exploitability certification",
  ],
  primary: { label: "Back to homepage", href: "/" },
  secondary: { label: "Opponent modeling", href: links.opponentModelingPage },
  tertiary: { label: "CFR explainer", href: links.cfrPage },
  docs: { label: "Research freeze", href: links.researchFreeze },
  app: { label: "Local app quick start", href: links.quickStart },
} as const;
