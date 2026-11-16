#!/usr/bin/env node
/**
 * Capture documentation screenshots from a REAL PokerMind session (M9).
 *
 * It drives the installed Google Chrome (headless) over the DevTools protocol, with real
 * mouse and keyboard input, against an already running app:
 *
 *   pokermind launch --demo            # terminal 1 (strategy updates every 10 hands)
 *   node web/scripts/capture-screenshots.mjs [--url http://localhost:3000] [--out docs/screenshots]
 *
 * Nothing is staged: deals come from the server, and the scripted player simply bets or
 * raises whenever that is legal (an aggressive style the dashboard can then describe).
 * Results, statistics and strategy versions are whatever the real session produced.
 * No dependencies; needs Node.js >= 22 (global WebSocket) and Google Chrome.
 */

import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const URL_ = opt("--url", "http://localhost:3000");
const OUT = resolve(opt("--out", "docs/screenshots"));
const CHROME =
  process.env.CHROME_PATH ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9333;
const HANDS = Number(opt("--hands", "24"));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitFor(fn, what, timeout = 60_000) {
  const end = Date.now() + timeout;
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await sleep(150);
  }
}

// -- Chrome + CDP -------------------------------------------------------------------------

const profile = mkdtempSync(join(tmpdir(), "pokermind-shots-"));
const chrome = spawn(
  CHROME,
  [
    "--headless=new",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${profile}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--hide-scrollbars",
    "--force-color-profile=srgb",
    "about:blank",
  ],
  { stdio: "ignore" },
);

let ws;
let nextId = 1;
const pending = new Map();

function send(method, params = {}) {
  const id = nextId++;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((res, rej) => pending.set(id, { res, rej }));
}

async function connect() {
  const target = await waitFor(async () => {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      return list.find((t) => t.type === "page");
    } catch {
      return null;
    }
  }, "Chrome DevTools");
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  ws.addEventListener("message", (e) => {
    const msg = JSON.parse(e.data);
    if (msg.id && pending.has(msg.id)) {
      const { res, rej } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) rej(new Error(msg.error.message));
      else res(msg.result);
    }
  });
  await send("Page.enable");
  await send("Runtime.enable");
}

async function evaluate(expression) {
  const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text);
  return r.result.value;
}

async function viewport(width, height, mobile = false) {
  await send("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: 2,
    mobile,
  });
}

/** Real mouse click on the first button whose text starts with `text`. */
async function click(text) {
  const box = await waitFor(
    () =>
      evaluate(`(() => {
        const b = [...document.querySelectorAll('button')].find(
          (x) => x.innerText.trim().startsWith(${JSON.stringify(text)}) && !x.disabled);
        if (!b) return null;
        b.scrollIntoView({ block: 'center' });
        const r = b.getBoundingClientRect();
        return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
      })()`),
    `button "${text}"`,
  );
  for (const type of ["mousePressed", "mouseReleased"]) {
    await send("Input.dispatchMouseEvent", { type, x: box.x, y: box.y, button: "left", clickCount: 1 });
  }
}

/** Real key press (the app's keyboard shortcuts). */
async function key(k) {
  await send("Input.dispatchKeyEvent", { type: "keyDown", key: k, text: k });
  await send("Input.dispatchKeyEvent", { type: "keyUp", key: k });
}

async function shot(name, { fullPage = false, scrollTo = null } = {}) {
  if (scrollTo) {
    await evaluate(`document.querySelector(${JSON.stringify(scrollTo)})?.scrollIntoView({block: 'start'})`);
  } else {
    await evaluate("window.scrollTo(0, 0)");
  }
  await sleep(400);
  const { data } = await send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: fullPage,
  });
  writeFileSync(join(OUT, name), Buffer.from(data, "base64"));
  console.log(`  saved ${name}`);
}

const idle = () =>
  waitFor(
    () =>
      evaluate(`(() => {
        const b = document.querySelector('.action-bar button');
        return !!b && !b.disabled;
      })()`),
    "an idle action bar",
  );

const labels = () =>
  evaluate(`[...document.querySelectorAll('.action-bar button')].map(b => b.innerText.split('\\n')[0])`);
const handsPlayed = () =>
  evaluate(`Number(document.querySelector('.session-bar .stat strong')?.textContent ?? 0)`);

/** One decision of the scripted (aggressive) player. Returns false once the hand is over. */
async function decide() {
  await idle();
  const l = await labels();
  if (l.includes("Next hand") || l.includes("Deal first hand")) return false;
  await key(l.includes("Bet") ? "b" : l.includes("Raise") ? "r" : "c");
  await sleep(120);
  return true;
}

// -- the session ----------------------------------------------------------------------------

async function main() {
  mkdirSync(OUT, { recursive: true });
  await connect();
  await viewport(1440, 1000);
  await send("Page.navigate", { url: URL_ });
  await waitFor(() => evaluate(`!!document.querySelector('h1')`), "the app");
  await evaluate(`localStorage.removeItem('pokermind.guide-dismissed'); sessionStorage.clear()`);
  await send("Page.reload");
  await sleep(800);

  console.log("capturing:");
  await click("Start a new session");
  await waitFor(() => evaluate(`!!document.querySelector('[role=tab]')`), "a session");
  await shot("01-new-session.png");

  await click("Got it"); // dismiss the first-time guide for the gameplay shots
  await click("Deal first hand");
  await waitFor(() => evaluate(`!!document.querySelector('[data-card]')`), "a dealt hand");
  await shot("02-first-hand.png");

  let showdownShot = false;
  let learningShot = false;
  while ((await handsPlayed()) < HANDS) {
    while (await decide());
    const result = await evaluate(`document.querySelector('[data-testid=result]')?.innerText ?? ''`);
    if (!showdownShot && /showdown/i.test(result)) {
      await shot("03-showdown-result.png");
      showdownShot = true;
    }
    if (await evaluate(`!!document.querySelector('.learning.on')`)) {
      if (!learningShot) {
        await shot("04-learning.png");
        learningShot = true;
      }
      // human-like pacing: let the background update finish (it continues during play,
      // but a fast script would otherwise outrun it and skip versions)
      await waitFor(
        async () => !(await evaluate(`!!document.querySelector('.learning.on')`)),
        "the background update",
        120_000,
      );
    }
    await key("n");
    await sleep(150);
  }
  while (await decide());

  // wait for the last scheduled update to finish so the history is complete
  await waitFor(
    async () => !(await evaluate(`!!document.querySelector('.learning.on')`)),
    "the background update",
    120_000,
  );
  await click("How PokerMind learns");
  await waitFor(() => evaluate(`!!document.querySelector('[data-testid=dash-version]')`), "the dashboard");
  await sleep(600);
  await shot("05-dashboard-observed.png");
  await shot("06-dashboard-tendencies.png", { scrollTo: ".tgrid" });
  await shot("07-dashboard-strategy.png", { scrollTo: "#evo-title" });

  await viewport(390, 844, true);
  await click("Play");
  await sleep(500);
  await shot("08-mobile-table.png");

  const summary = await evaluate(`({
    hands: Number(document.querySelector('.session-bar .stat strong')?.textContent),
  })`);
  console.log(`done: ${summary.hands} hands played in a real session; screenshots in ${OUT}`);
}

main()
  .catch((err) => {
    console.error(`capture failed: ${err.message}`);
    process.exitCode = 1;
  })
  .finally(() => {
    try {
      ws?.close();
    } catch {}
    chrome.kill("SIGTERM");
    setTimeout(() => rmSync(profile, { recursive: true, force: true }), 500);
  });
