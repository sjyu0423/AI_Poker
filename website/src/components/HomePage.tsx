import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";
import { Hero } from "@/components/Hero";
import { Difference } from "@/components/Difference";
import { HowItWorks } from "@/components/HowItWorks";
import { ResearchHighlights } from "@/components/ResearchHighlights";
import { AppPreview } from "@/components/AppPreview";
import { Technology } from "@/components/Technology";
import { FinalCta } from "@/components/FinalCta";
import { withBasePath } from "@/lib/paths";

export function HomePage() {
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <SiteHeader brandHref={withBasePath("/")} />
      <main id="main">
        <Hero />
        <Difference />
        <HowItWorks />
        <ResearchHighlights />
        <AppPreview />
        <Technology />
        <FinalCta />
      </main>
      <SiteFooter />
    </>
  );
}
