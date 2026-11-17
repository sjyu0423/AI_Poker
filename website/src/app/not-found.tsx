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
          That URL is not part of the static {siteMeta.name} research showcase.
          Try the homepage or one of the interactive exhibits—or open the
          repository for the local playable app.
        </p>
        <div className="hero__actions">
          <Button href={withBasePath("/")}>Back to homepage</Button>
          <Button href={withBasePath(links.robustResponsePage)} variant="secondary">
            Research exhibits
          </Button>
          <Button href={links.github} variant="ghost" external>
            View on GitHub
          </Button>
        </div>
        <ul className="not-found__links">
          <li>
            <a href={withBasePath(links.cfrPage)}>CFR explainer</a>
          </li>
          <li>
            <a href={withBasePath(links.opponentModelingPage)}>Opponent modeling</a>
          </li>
          <li>
            <a href={withBasePath(links.robustResponsePage)}>Restricted Nash Response</a>
          </li>
        </ul>
      </div>
    </main>
  );
}
