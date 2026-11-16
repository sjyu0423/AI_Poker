import type { HandView } from "@/lib/types";

import { chips } from "./format";

/** The most recent completed hands, newest first (from `GET /history`). */
export function History({ hands, total }: { hands: HandView[]; total: number }) {
  return (
    <section className="panel" aria-labelledby="history-title">
      <h2 id="history-title" className="panel-title">
        Recent hands
      </h2>
      {hands.length === 0 ? (
        <p className="muted">Completed hands will appear here.</p>
      ) : (
        <table className="history">
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">You</th>
              <th scope="col">Board</th>
              <th scope="col">AI</th>
              <th scope="col">Result</th>
            </tr>
          </thead>
          <tbody>
            {hands.map((h) => {
              const net = h.result?.human_net ?? 0;
              return (
                <tr key={h.hand_number}>
                  <td>{h.hand_number}</td>
                  <td className="mono">{h.human_card}</td>
                  <td className="mono">{h.public_card ?? "–"}</td>
                  <td className="mono" title={h.result?.ai_card ? undefined : "not shown (fold)"}>
                    {h.result?.ai_card ?? "··"}
                  </td>
                  <td className={net > 0 ? "pos" : net < 0 ? "neg" : ""}>
                    {chips(net)}
                    <span className="muted"> {h.result?.ended_by === "fold" ? "fold" : "sd"}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {total > hands.length && <p className="muted small">Showing the last {hands.length} of {total}.</p>}
    </section>
  );
}
