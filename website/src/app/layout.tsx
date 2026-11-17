import type { Metadata, Viewport } from "next";
import { Fraunces, Source_Sans_3, IBM_Plex_Mono } from "next/font/google";
import { siteMeta } from "@/content/site";
import "./globals.css";
import "./components.css";
import "./cfr.css";
import "./opponent.css";
import "./robust.css";

const display = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const body = Source_Sans_3({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL(siteMeta.url),
  title: {
    default: `${siteMeta.name} — Adaptive Poker AI Research`,
    template: `%s · ${siteMeta.name}`,
  },
  description: siteMeta.description,
  applicationName: siteMeta.name,
  authors: [{ name: "PokerMind" }],
  keywords: [
    "PokerMind",
    "Leduc Hold'em",
    "CFR",
    "Restricted Nash Response",
    "Bayesian opponent modeling",
    "game theory",
    "poker AI research",
  ],
  openGraph: {
    type: "website",
    url: siteMeta.url,
    title: `${siteMeta.name} — ${siteMeta.tagline}`,
    description: siteMeta.description,
    siteName: siteMeta.name,
    locale: siteMeta.locale,
    images: [
      {
        url: siteMeta.ogImage,
        width: 1280,
        height: 720,
        alt: siteMeta.ogImageAlt,
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: `${siteMeta.name} — ${siteMeta.tagline}`,
    description: siteMeta.description,
    images: [siteMeta.ogImage],
  },
  robots: {
    index: true,
    follow: true,
  },
  alternates: {
    canonical: "/",
  },
};

export const viewport: Viewport = {
  themeColor: "#12151a",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body
        style={
          {
            "--pm-font-display": "var(--font-display), Georgia, serif",
            "--pm-font-body": "var(--font-body), system-ui, sans-serif",
            "--pm-font-mono": "var(--font-mono), ui-monospace, monospace",
          } as React.CSSProperties
        }
      >
        {children}
      </body>
    </html>
  );
}
