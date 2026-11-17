import type { Metadata } from "next";
import { siteMeta } from "@/content/site";

interface ExhibitMetaInput {
  title: string;
  description: string;
  path: string;
}

/** Shared Open Graph / Twitter card fields for research exhibit pages. */
export function exhibitMetadata({
  title,
  description,
  path,
}: ExhibitMetaInput): Metadata {
  return {
    title,
    description,
    alternates: {
      canonical: path,
    },
    openGraph: {
      title: `${title} · ${siteMeta.name}`,
      description,
      url: path,
      type: "article",
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
      title: `${title} · ${siteMeta.name}`,
      description,
      images: [siteMeta.ogImage],
    },
  };
}
