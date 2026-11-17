"use client";

import { useEffect, useState } from "react";
import { navItems as homeNavItems, links, siteMeta } from "@/content/site";
import { withBasePath } from "@/lib/paths";

export interface NavItem {
  label: string;
  href: string;
}

interface SiteHeaderProps {
  brandHref?: string;
  navItems?: readonly NavItem[];
  trailingHref?: string;
  trailingLabel?: string;
}

export function SiteHeader({
  brandHref = "#top",
  navItems = homeNavItems,
  trailingHref = links.github,
  trailingLabel = "GitHub",
}: SiteHeaderProps) {
  const [open, setOpen] = useState(false);
  const trailingExternal = trailingHref.startsWith("http");

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
        <a className="site-header__brand" href={brandHref} aria-label={`${siteMeta.name} home`}>
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
            {navItems.map((item) => {
              const href =
                item.href.startsWith("/") && !item.href.startsWith("//")
                  ? withBasePath(item.href)
                  : item.href;
              return (
                <li key={item.href}>
                  <a href={href} onClick={() => setOpen(false)}>
                    {item.label}
                  </a>
                </li>
              );
            })}
          </ul>
          <a
            className="site-header__github"
            href={trailingHref}
            {...(trailingExternal
              ? { target: "_blank", rel: "noopener noreferrer" }
              : {})}
            onClick={() => setOpen(false)}
          >
            {trailingLabel}
          </a>
        </nav>
      </div>
    </header>
  );
}
