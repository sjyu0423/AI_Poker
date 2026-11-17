import { OpponentModelingPage } from "@/components/opponent/OpponentModelingPage";
import { opponentPageMeta } from "@/content/opponentModelingPage";
import { exhibitMetadata } from "@/lib/pageMetadata";

export const metadata = exhibitMetadata(opponentPageMeta);

export default function Page() {
  return <OpponentModelingPage />;
}
