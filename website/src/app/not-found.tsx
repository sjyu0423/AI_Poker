import { links, siteMeta } from "@/content/site";
import { Button } from "@/components/Button";
import { withBasePath } from "@/lib/paths";

export default function NotFound() {
  return (
    <main className="not-found">
      <div className="shell not-found__panel">
        <p className="section-kicker">404</p>
        <h1>Page not found</h1>
        <p>
          This static showcase only publishes the {siteMeta.name} homepage. The
          playable app and experiment tools live in the repository.
        </p>
        <div className="hero__actions">
          <Button href={withBasePath("/")}>Back to homepage</Button>
          <Button href={links.github} variant="secondary" external>
            View on GitHub
          </Button>
        </div>
      </div>
    </main>
  );
}
