import { describe, expect, it } from "vitest";

import { ApiError, ConnectionError, PokerMindApi } from "@/lib/api";

function api(reply: () => Response | Promise<Response>, seen: Request[] = []): PokerMindApi {
  return new PokerMindApi("/api", (async (input: RequestInfo | URL, init?: RequestInit) => {
    seen.push(new Request(new URL(String(input), "http://ui.test"), init));
    return reply();
  }) as typeof fetch);
}

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("PokerMindApi", () => {
  it("sends the action with the state_version the player saw", async () => {
    const seen: Request[] = [];
    await api(() => json(200, {}), seen).act("abc", "raise", 7);
    const req = seen[0]!;
    expect(req.method).toBe("POST");
    expect(new URL(req.url).pathname).toBe("/api/sessions/abc/actions");
    expect(await req.json()).toEqual({ action: "raise", state_version: 7 });
  });

  it("carries the HTTP status and backend detail", async () => {
    for (const status of [400, 404, 409]) {
      const err = await api(() => json(status, { detail: `d${status}` }))
        .getSession("x")
        .catch((e: unknown) => e);
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).status).toBe(status);
      expect((err as ApiError).detail).toBe(`d${status}`);
    }
  });

  it("summarises FastAPI validation errors (422)", async () => {
    const err = await api(() =>
      json(422, { detail: [{ msg: "Input should be 'fold', 'call' or 'raise'" }] }),
    )
      .act("x", "call", 1)
      .catch((e: unknown) => e);
    expect((err as ApiError).status).toBe(422);
    expect((err as ApiError).detail).toContain("Input should be");
  });

  it("reports an unreachable backend as a connection error", async () => {
    const network = await api(() => Promise.reject(new TypeError("fetch failed")))
      .health()
      .catch((e: unknown) => e);
    expect(network).toBeInstanceOf(ConnectionError);
    const gateway = await api(() => new Response("Bad Gateway", { status: 502 }))
      .health()
      .catch((e: unknown) => e);
    expect(gateway).toBeInstanceOf(ConnectionError);
  });

  it("handles 204 No Content", async () => {
    await expect(api(() => new Response(null, { status: 204 })).deleteSession("x")).resolves.toBe(
      undefined,
    );
  });
});
