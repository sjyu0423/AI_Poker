import type { Metadata } from "next";
import { OpponentModelingPage } from "@/components/opponent/OpponentModelingPage";
import { opponentPageMeta } from "@/content/opponentModelingPage";
import { siteMeta } from "@/content/site";

export const metadata: Metadata = {
  title: opponentPageMeta.title,
  description: opponentPageMeta.description,
  alternates: {
    canonical: opponentPageMeta.path,
  },
  openGraph: {
    title: `${opponentPageMeta.title} · ${siteMeta.name}`,
    description: opponentPageMeta.description,
    url: opponentPageMeta.path,
    type: "article",
  },
};

export default function Page() {
  return <OpponentModelingPage />;
}
