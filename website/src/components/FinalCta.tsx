import { finalCta } from "@/content/site";
import { Button } from "@/components/Button";

export function FinalCta() {
  return (
    <section className="section final-cta" aria-labelledby="final-cta-title">
      <div className="shell final-cta__panel">
        <h2 id="final-cta-title">{finalCta.headline}</h2>
        <div className="hero__actions">
          {finalCta.links.map((item, index) => (
            <Button
              key={item.href}
              href={item.href}
              variant={index === 0 ? "primary" : index === 1 ? "secondary" : "ghost"}
              external
            >
              {item.label}
            </Button>
          ))}
        </div>
      </div>
    </section>
  );
}
