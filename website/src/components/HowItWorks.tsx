"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { pipelineSteps } from "@/content/site";

function subscribeReducedMotion(onChange: () => void) {
  const media = window.matchMedia("(prefers-reduced-motion: reduce)");
  media.addEventListener("change", onChange);
  return () => media.removeEventListener("change", onChange);
}

function getReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function usePrefersReducedMotion() {
  return useSyncExternalStore(subscribeReducedMotion, getReducedMotion, () => false);
}

export function HowItWorks() {
  const ref = useRef<HTMLOListElement>(null);
  const reduceMotion = usePrefersReducedMotion();
  const [scrollActive, setScrollActive] = useState(0);
  const active = reduceMotion ? pipelineSteps.length - 1 : scrollActive;

  useEffect(() => {
    const root = ref.current;
    if (!root || reduceMotion) {
      return;
    }

    const items = Array.from(root.querySelectorAll<HTMLElement>("[data-step]"));
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) {
          return;
        }
        const index = Number((visible.target as HTMLElement).dataset.step);
        if (!Number.isNaN(index)) {
          setScrollActive(index);
        }
      },
      { root: null, threshold: [0.35, 0.6], rootMargin: "-20% 0px -35% 0px" },
    );

    items.forEach((item) => observer.observe(item));
    return () => observer.disconnect();
  }, [reduceMotion]);

  return (
    <section className="section how" id="how-it-works" aria-labelledby="how-title">
      <div className="shell">
        <header className="section-header">
          <p className="section-kicker">Pipeline</p>
          <h2 id="how-title">How it works</h2>
          <p>
            A five-step loop from observed play to a certified strategy—built so
            adaptation never silently exceeds its exploitability budget.
          </p>
        </header>

        <div className="pipeline" aria-hidden="true">
          <div className="pipeline__track">
            <div
              className="pipeline__progress"
              style={{ width: `${((active + 1) / pipelineSteps.length) * 100}%` }}
            />
          </div>
        </div>

        <ol className="pipeline-steps" ref={ref}>
          {pipelineSteps.map((step, index) => (
            <li
              key={step.step}
              data-step={index}
              className={`pipeline-step ${index <= active ? "is-active" : ""}`}
            >
              <div className="pipeline-step__num mono">{step.step}</div>
              <div>
                <h3>{step.title}</h3>
                <p>{step.detail}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
