"use client";

import { useEffect, useState } from "react";
import { navItems, links, siteMeta } from "@/content/site";

export function SiteHeader() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <header className="site-header">
      <div className="shell site-header__inner">
        <a className="site-header__brand" href="#top" aria-label={`${siteMeta.name} home`}>
          <span className="site-header__mark" aria-hidden="true">
            ♠
          </span>
          <span>{siteMeta.name}</span>
        </a>

        <button
          type="button"
          className="site-header__toggle"
          aria-expanded={open}
          aria-controls="site-nav"
          onClick={() => setOpen((value) => !value)}
        >
          <span className="sr-only">{open ? "Close menu" : "Open menu"}</span>
          <span aria-hidden="true">{open ? "✕" : "☰"}</span>
        </button>

        <nav id="site-nav" className={`site-header__nav ${open ? "is-open" : ""}`} aria-label="Primary">
          <ul>
            {navItems.map((item) => (
              <li key={item.href}>
                <a href={item.href} onClick={() => setOpen(false)}>
                  {item.label}
                </a>
              </li>
            ))}
          </ul>
          <a
            className="site-header__github"
            href={links.github}
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => setOpen(false)}
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
