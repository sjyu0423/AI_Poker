import type { Metadata } from "next";
import { RobustResponsePage } from "@/components/robust/RobustResponsePage";
import { robustPageMeta } from "@/content/robustResponsePage";
import { siteMeta } from "@/content/site";

export const metadata: Metadata = {
  title: robustPageMeta.title,
  description: robustPageMeta.description,
  alternates: {
    canonical: robustPageMeta.path,
  },
  openGraph: {
    title: `${robustPageMeta.title} · ${siteMeta.name}`,
    description: robustPageMeta.description,
    url: robustPageMeta.path,
    type: "article",
  },
};

export default function Page() {
  return <RobustResponsePage />;
}
