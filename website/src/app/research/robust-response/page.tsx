import { RobustResponsePage } from "@/components/robust/RobustResponsePage";
import { robustPageMeta } from "@/content/robustResponsePage";
import { exhibitMetadata } from "@/lib/pageMetadata";

export const metadata = exhibitMetadata(robustPageMeta);

export default function Page() {
  return <RobustResponsePage />;
}
