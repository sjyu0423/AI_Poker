import { links } from "@/content/site";
import { M5_SOURCE } from "@/content/m5OpponentModeling";

export const opponentPageMeta = {
  title: "Learning the Player Behind the Cards",
  description:
    "An interactive exhibit of Bayesian opponent modeling—how PokerMind learns betting tendencies from limited public observations under uncertainty.",
  path: "/research/opponent-modeling/",
} as const;

export const opponentNav = [
  { label: "Idea", href: "#intro" },
  { label: "Bayesian learner", href: "#bayes" },
  { label: "Uncertainty", href: "#uncertainty" },
  { label: "Hidden cards", href: "#hidden" },
  { label: "M5 results", href: "#results" },
  { label: "Change", href: "#change" },
] as const;

export const opponentIntro = {
  kicker: "Research exhibit · M5 / M6",
  headline: "Learning the Player Behind the Cards.",
  lede:
    "PokerMind can observe bets, checks, calls and folds—but it usually cannot see the opponent’s private card. The AI must learn behavior without knowing everything that caused it.",
  plain:
    "From a thin stream of public actions, PokerMind updates a Bayesian model of how often an opponent folds, calls or raises in each situation. With little data the estimate stays near a population prior; with more data it moves, and its uncertainty shrinks—though never enough to erase hidden information.",
} as const;

export const bayesCopy = {
  title: "A Bayesian learner you can drive",
  summary:
    "This toy tracks one binary tendency—“bet when the pot is unopened”—with a Beta–Binomial update. Drag the prior, click outcomes, and watch the posterior leave the observed frequency behind when data are scarce.",
  disclaimer:
    "Illustrative simplification. PokerMind’s real model is hierarchical Dirichlet-categorical over many public contexts, with population back-off—not a single Beta coin flip.",
  tendencyLabel: "Bet when unchecked",
  successLabel: "Bet",
  failureLabel: "Check",
  formula:
    "Prior Beta(s·μ, s·(1−μ)); after k bets in n chances → Beta(s·μ+k, s·(1−μ)+n−k). Mean = α/(α+β).",
  presets: [
    {
      id: "prior",
      label: "Prior only",
      priorMean: 0.42,
      priorStrength: 8,
      successes: 0,
      trials: 0,
    },
    {
      id: "sparse",
      label: "3 aggressive bets",
      priorMean: 0.42,
      priorStrength: 8,
      successes: 3,
      trials: 3,
    },
    {
      id: "mixed",
      label: "2 bets in 10",
      priorMean: 0.42,
      priorStrength: 8,
      successes: 2,
      trials: 10,
    },
    {
      id: "plenty",
      label: "40 bets in 100",
      priorMean: 0.42,
      priorStrength: 8,
      successes: 40,
      trials: 100,
    },
  ],
} as const;

export const uncertaintyCopy = {
  title: "Why three observations are not a hundred",
  summary:
    "The same raw frequency can hide very different posteriors. With few trials the prior still owns the mean and the 90% credible interval stays wide; with many trials the interval tightens around the data.",
  intervalNote:
    "The interval below is the central 90% credible interval of the Beta posterior—computed from the quantile function, not drawn for decoration. In the real project, nominal 90% intervals covered truth only 74% of the time after 20 hands on simulated opponents, and they were never validated for humans.",
} as const;

export const hiddenCopy = {
  title: "When public frequencies collide",
  summary:
    "Two opponents can bet at the same public rate for opposite reasons. A reply that works against one can fail against the other—so M6 reconstructs card-conditional policies instead of copying the public marginal onto every card.",
} as const;

export const resultsCopy = {
  title: "What M5 measured on held-out opponents",
  summary:
    "Against simulated Leduc opponents, with PokerMind playing near equilibrium, excess log loss falls as hands accumulate—but a perfect public predictor still cannot match an oracle that sees private cards.",
  metric:
    "Excess log loss = expected log loss minus that of the best public-information predictor, in nats per opponent decision. Lower is better. It is not “accuracy.”",
  sourceNote: `Source: ${M5_SOURCE.docs} (${M5_SOURCE.artifact}).`,
} as const;

export const changeCopy = {
  title: "When opponents change style",
  summary:
    "Keeping every past hand forever is slow after a switch. Exponential forgetting adapts faster but wastes useful stationary data. A CUSUM detector that resets to recent hands recovers quickest after a change—and still false-alarms on some stationary sessions.",
  caveat:
    "The detector is useful, not infallible: 5 of 240 switches were missed within the evaluation window, and about 40% of 500-hand stationary runs had at least one false alarm.",
} as const;

export const continueCopy = {
  headline:
    "Knowing how someone plays is one thing. How do you exploit those tendencies without becoming vulnerable yourself?",
  body:
    "Restricted Nash Response and exact exploitability budgets are the next chapter. Until that interactive exhibit ships, return to the homepage for the measured RNR results—or revisit how CFR built the safe baseline.",
  primary: { label: "Back to homepage", href: "/" },
  secondary: { label: "CFR explainer", href: links.cfrPage },
  tertiary: { label: "M5 results note", href: links.m5 },
} as const;
