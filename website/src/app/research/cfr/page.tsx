import { CfrPage } from "@/components/cfr/CfrPage";
import { cfrPageMeta } from "@/content/cfrPage";
import { exhibitMetadata } from "@/lib/pageMetadata";

export const metadata = exhibitMetadata(cfrPageMeta);

export default function Page() {
  return <CfrPage />;
}
