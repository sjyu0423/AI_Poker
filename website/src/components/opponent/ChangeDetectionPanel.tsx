import { M5_DETECTOR, M5_SWITCH_ADAPTATION } from "@/content/m5OpponentModeling";
import { changeCopy } from "@/content/opponentModelingPage";

export function ChangeDetectionPanel() {
  return (
    <div className="demo-panel">
      <div className="demo-panel__toolbar">
        <p className="mono demo-panel__badge">Measured · switching opponents</p>
      </div>
      <p>{changeCopy.summary}</p>

      <div className="change-table-wrap">
        <table className="change-table">
          <caption className="sr-only">
            Excess log loss after a style switch for cumulative, forgetting, and CUSUM reset
          </caption>
          <thead>
            <tr>
              <th scope="col">Hands since switch</th>
              <th scope="col">Cumulative</th>
              <th scope="col">
                Forgetting (γ={M5_DETECTOR.forgettingGamma})
              </th>
              <th scope="col">CUSUM + reset</th>
            </tr>
          </thead>
          <tbody>
            {M5_SWITCH_ADAPTATION.map((row) => (
              <tr key={row.handsSinceSwitch}>
                <td className="mono">{row.handsSinceSwitch}</td>
                <td className="mono">{row.cumulative.toFixed(3)}</td>
                <td className="mono">{row.forgetting.toFixed(3)}</td>
                <td className="mono">{row.cusumReset.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted">Units: excess log loss, nats per opponent decision (lower is better).</p>

      <ul className="diff-grid">
        <li className="diff-card">
          <h3>Detection delay</h3>
          <p>
            Median {M5_DETECTOR.detectionDelayMedianHands} hands; mean{" "}
            {M5_DETECTOR.detectionDelayMeanHands} [
            {M5_DETECTOR.detectionDelayCi[0]}, {M5_DETECTOR.detectionDelayCi[1]}]. Switches found:{" "}
            {M5_DETECTOR.switchesDetected}.
          </p>
        </li>
        <li className="diff-card">
          <h3>False alarms</h3>
          <p>
            About {M5_DETECTOR.falseAlarmsBeforeSwitchPerThousandHands} per 1,000 hands before a
            real switch. On stationary opponents, ~
            {Math.round(M5_DETECTOR.stationaryFalseAlarmShareOfRuns * 100)}% of 500-hand runs had
            at least one reset.
          </p>
        </li>
        <li className="diff-card">
          <h3>Stationary cost at 500 hands</h3>
          <p>
            Excess log loss: cumulative {M5_DETECTOR.stationaryExcessAt500.cumulative.toFixed(3)};
            CUSUM {M5_DETECTOR.stationaryExcessAt500.cusum.toFixed(3)}; forgetting{" "}
            {M5_DETECTOR.stationaryExcessAt500.forgetting.toFixed(3)}.
          </p>
        </li>
      </ul>

      <p className="research-disclaimer" role="note">
        {changeCopy.caveat}
      </p>
    </div>
  );
}
