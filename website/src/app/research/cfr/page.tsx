import type { Metadata } from "next";
import { CfrPage } from "@/components/cfr/CfrPage";
import { cfrPageMeta } from "@/content/cfrPage";
import { siteMeta } from "@/content/site";

export const metadata: Metadata = {
  title: cfrPageMeta.title,
  description: cfrPageMeta.description,
  alternates: {
    canonical: cfrPageMeta.path,
  },
  openGraph: {
    title: `${cfrPageMeta.title} · ${siteMeta.name}`,
    description: cfrPageMeta.description,
    url: cfrPageMeta.path,
    type: "article",
  },
};

export default function Page() {
  return <CfrPage />;
}
