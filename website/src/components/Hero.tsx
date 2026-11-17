import { hero } from "@/content/site";
import { Button } from "@/components/Button";
import { HeroVisual } from "@/components/HeroVisual";
import { withBasePath } from "@/lib/paths";

export function Hero() {
  return (
    <section className="hero section" id="top" aria-labelledby="hero-brand">
      <div className="shell hero__grid">
        <div className="hero__copy">
          <p className="section-kicker">Adaptive poker AI · research portfolio</p>
          <h1 id="hero-brand">{hero.brand}</h1>
          <p className="hero__headline">{hero.headline}</p>
          <p className="hero__summary">{hero.summary}</p>
          <div className="hero__actions">
            <Button href={hero.primaryCta.href}>{hero.primaryCta.label}</Button>
            <Button href={withBasePath(hero.cfrCta.href)} variant="secondary">
              {hero.cfrCta.label}
            </Button>
            <Button href={withBasePath(hero.opponentCta.href)} variant="secondary">
              {hero.opponentCta.label}
            </Button>
            <Button href={hero.secondaryCta.href} variant="ghost" external>
              {hero.secondaryCta.label}
            </Button>
            <Button href={hero.releaseCta.href} variant="ghost" external>
              {hero.releaseCta.label}
            </Button>
          </div>
          <p className="hero__note muted">{hero.localNote}</p>
        </div>
        <HeroVisual />
      </div>
    </section>
  );
}
