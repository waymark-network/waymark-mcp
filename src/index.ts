/*
 * Copyright 2026 MC Software, LLC. Licensed under the Apache License, Version 2.0.
 * Waymark is a service of MC Software, LLC — est. 1974. https://waymark.network
 */
/**
 * Waymark MCP server — collective procedural-knowledge network for AI agents.
 *
 * Tools:
 *   waymark_query      — find verified routes (step sequences that worked) for a task
 *   waymark_contribute — submit a sanitized trace of what worked / failed (API key)
 *   waymark_attest     — report the outcome of following a route (trust consensus)
 *
 * v0.2: full activity log (every tool call recorded to KV, 30-day TTL) +
 * public observability endpoints: /stats, /routes, /activity, /dashboard.
 *
 * Alpha storage: Cloudflare KV (key = route id, plus a scan-friendly index).
 * Scale path: D1 for relational attestation history, Vectorize for semantic
 * route matching (replace keywordScore below with an embedding query).
 */

import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

interface Env {
  ROUTES: KVNamespace;
  MCP_OBJECT: DurableObjectNamespace;
  AI: Ai;
  VEC: VectorizeIndex;
  WRITE_KEY: string;
  SERVER_NAME: string;
  SERVER_VERSION: string;
  PUBLIC_URL: string;
}

interface Route {
  id: string;
  task: string;            // natural-language task this route accomplishes
  domain: string;          // e.g. "stripe.com", "github-api", "salesforce"
  steps: string[];         // ordered procedural steps / tool-call sequence
  gotchas: string[];       // known failure modes, rate limits, quirks
  contributor: string;     // contributor handle (never PII from traces)
  created: string;
  attestations: { success: number; failure: number; lastAt: string | null };
}

type EventType = "query" | "contribute" | "attest";
interface ActivityEvent {
  t: string;               // ISO timestamp
  type: EventType;
  detail: Record<string, unknown>;
}

const EVENT_TTL_SECONDS = 60 * 60 * 24 * 30; // 30 days

/** Abuse guard: max same-outcome attestations per route per hour (network total ≈23 attests today, so 10/hr/route/outcome only trips bulk spam). */
const ATTEST_OUTCOME_HOURLY_CAP = 10;

/** Activity log. Key sorts chronologically (ISO prefix). Returns the put promise
 * (never rejects) — callers MUST await it or hand it to ctx.waitUntil; an
 * un-awaited call gets cancelled when the response returns (the 2026-06-12
 * telemetry outage: zero evt: writes landed for ~15h). */
function logEvent(env: Env, type: EventType, detail: Record<string, unknown>): Promise<void> {
  const t = new Date().toISOString();
  const key = `evt:${t}:${crypto.randomUUID().slice(0, 8)}`;
  return env.ROUTES.put(key, JSON.stringify({ t, type, detail } satisfies ActivityEvent), {
    expirationTtl: EVENT_TTL_SECONDS,
  }).catch(() => {});
}

const tokenize = (s: string) =>
  s.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 2);

/* ---------------- v0.4 indexed retrieval ----------------
 * One KV value holds a compact index of every route (id, task, domain,
 * attestation counts). Queries read ONE key, score in memory (IDF +
 * bigram + domain boost + trust), then fetch only the top-k full routes.
 * Removes the N-reads-per-query cost and the 1000-route list cap on the
 * query path. Confidence threshold: a wrong route is worse than no route
 * (benchmark-proven), so low-confidence matches are refused.
 * Still interim: true semantic matching = Vectorize migration. */

type IdxEntry = Route;
const INDEX_KEY = "idx:routes";

async function listAllRouteKeys(env: Env): Promise<string[]> {
  const keys: string[] = [];
  let cursor: string | undefined;
  do {
    const page = await env.ROUTES.list({ prefix: "route:", limit: 1000, cursor });
    keys.push(...page.keys.map((k) => k.name));
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);
  return keys;
}

async function buildIndex(env: Env): Promise<IdxEntry[]> {
  const keys = await listAllRouteKeys(env);
  const entries: IdxEntry[] = [];
  for (let i = 0; i < keys.length; i += 100) {
    const chunk = await Promise.all(keys.slice(i, i + 100).map((k) => env.ROUTES.get(k)));
    for (const v of chunk) {
      if (!v) continue;
      const r: Route = JSON.parse(v);
      entries.push(r);
    }
  }
  await env.ROUTES.put(INDEX_KEY, JSON.stringify(entries));
  return entries;
}

async function getIndex(env: Env): Promise<IdxEntry[]> {
  const raw = await env.ROUTES.get(INDEX_KEY);
  if (raw) return JSON.parse(raw);
  return buildIndex(env);
}

/** Best-effort index mutation (alpha: last-writer-wins is acceptable). */
async function patchIndex(env: Env, fn: (idx: IdxEntry[]) => IdxEntry[]): Promise<void> {
  try {
    const idx = await getIndex(env);
    await env.ROUTES.put(INDEX_KEY, JSON.stringify(fn(idx)));
  } catch { /* rebuildable */ }
}

const bigrams = (tokens: string[]) => {
  const out = new Set<string>();
  for (let i = 0; i < tokens.length - 1; i++) out.add(tokens[i] + " " + tokens[i + 1]);
  return out;
};

interface Scored { e: IdxEntry; score: number; coverage: number }

/** IDF + bigram + domain-boost + trust scoring with a refusal threshold. */
function rankIndex(idx: IdxEntry[], query: string, domainHint: string | undefined, limit: number): Scored[] {
  const qTokens = tokenize(query + (domainHint ? " " + domainHint : ""));
  if (qTokens.length === 0) return [];
  const qSet = new Set(qTokens), qBi = bigrams(qTokens);
  const N = idx.length || 1;
  const df = new Map<string, number>();
  const entryTokens: string[][] = idx.map((e) => tokenize(e.task + " " + e.domain));
  for (const toks of entryTokens) for (const t of new Set(toks)) df.set(t, (df.get(t) ?? 0) + 1);
  const idf = (t: string) => Math.log(1 + N / (df.get(t) ?? 1));

  const scored: Scored[] = [];
  for (let i = 0; i < idx.length; i++) {
    const e = idx[i], toks = entryTokens[i];
    let s = 0; const matched = new Set<string>();
    for (const t of new Set(toks)) if (qSet.has(t)) { s += idf(t); matched.add(t); }
    if (matched.size === 0) continue;
    for (const b of bigrams(toks)) if (qBi.has(b)) s += 1.5; // phrase evidence
    if (domainHint && e.domain.toLowerCase() === domainHint.toLowerCase()) s *= 1.5;
    const trust = (e.attestations.success + 1) / (e.attestations.success + e.attestations.failure + 2);
    s *= 0.5 + trust; // trust scales 0.5x–1.5x
    const coverage = matched.size / Math.min(qSet.size, 6);
    scored.push({ e, score: s, coverage });
  }
  scored.sort((a, b) => b.score - a.score);
  // Refusal threshold: wrong route < no route. Require real evidence.
  const top = scored.slice(0, limit).filter(
    (x) => x.coverage >= 0.34 || (domainHint && x.e.domain.toLowerCase() === domainHint.toLowerCase() && x.coverage >= 0.2)
  );
  return top;
}

async function fetchRoutes(env: Env, ids: string[]): Promise<Route[]> {
  const vals = await Promise.all(ids.map((id) => env.ROUTES.get(`route:${id}`)));
  return vals.filter((v): v is string => v !== null).map((v) => JSON.parse(v));
}

/* ---------------- v0.5 semantic retrieval (Vectorize + Workers AI) ----
 * Primary ranker: cosine similarity over bge-base-en-v1.5 embeddings of
 * "task — domain". Falls back to the keyword index if the vector store
 * is empty/unavailable. Confidence cutoff retained: no route > wrong route. */

const EMBED_MODEL = "@cf/baai/bge-base-en-v1.5";
const VEC_MIN_SCORE = 0.56; // calibrated: exact ~0.94, paraphrase ~0.60+, garbage ≤0.49

async function embed(env: Env, texts: string[]): Promise<number[][]> {
  const res = (await env.AI.run(EMBED_MODEL, { text: texts })) as { data: number[][] };
  return res.data;
}

/** BGE retrieval convention: queries get the instruction prefix, passages don't. */
const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
const embedQuery = (env: Env, q: string) => embed(env, [QUERY_PREFIX + q]).then((v) => v[0]);

async function upsertVector(env: Env, route: Route): Promise<void> {
  try {
    const [v] = await embed(env, [route.task + " — " + route.domain]);
    await env.VEC.upsert([{ id: route.id, values: v, metadata: { task: route.task.slice(0, 200), domain: route.domain } }]);
  } catch { /* keyword fallback still serves it */ }
}

/** Hybrid retrieve: vector-first with cutoff, keyword fallback. */
async function retrieve(env: Env, query: string, domainHint: string | undefined, limit: number): Promise<Route[]> {
  try {
    const qv = await embedQuery(env, query + (domainHint ? " — " + domainHint : ""));
    const res = await env.VEC.query(qv, { topK: Math.max(limit * 2, 8), returnMetadata: "none" });
    const hits = res.matches.filter((m) => m.score >= VEC_MIN_SCORE).slice(0, limit);
    if (hits.length > 0) {
      const routes = await fetchRoutes(env, hits.map((m) => m.id));
      if (routes.length > 0) return routes;
    }
    // Semantic miss with a domain hint can still have an exact keyword match.
    const idx = await getIndex(env);
    const kw = rankIndex(idx, query, domainHint, limit);
    return fetchRoutes(env, kw.map((x) => x.e.id));
  } catch {
    const idx = await getIndex(env);
    const kw = rankIndex(idx, query, domainHint, limit);
    return fetchRoutes(env, kw.map((x) => x.e.id));
  }
}

async function vecDebug(env: Env, q: string, key: string | null): Promise<Response> {
  if (key !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
  const qv = await embedQuery(env, q);
  const res = await env.VEC.query(qv, { topK: 10, returnMetadata: "all" });
  return Response.json(res.matches.map((m) => ({ score: +m.score.toFixed(4), task: (m.metadata as { task?: string })?.task })));
}

/** One-time/maintenance backfill of all route embeddings (WRITE_KEY-gated). */
async function backfillVectors(env: Env, key: string | null, since?: string | null): Promise<Response> {
  if (key !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
  let routes = await getIndex(env);
  if (since) routes = routes.filter((r) => r.created >= since);
  let n = 0;
  for (let i = 0; i < routes.length; i += 50) {
    const chunk = routes.slice(i, i + 50);
    const vecs = await embed(env, chunk.map((r) => r.task + " — " + r.domain));
    await env.VEC.upsert(chunk.map((r, j) => ({
      id: r.id, values: vecs[j], metadata: { task: r.task.slice(0, 200), domain: r.domain },
    })));
    n += chunk.length;
  }
  return Response.json({ embedded: n });
}

/** Server-side index merge (WRITE_KEY-gated). The client sends only route IDs
 * (a few KB); the Worker fetches the full routes from KV and merges them into
 * idx:routes in-place. Replaces the failure-prone multi-MB client-side index
 * PUT (Run 22 incident: 7 MB body died with SSL resets ×4 from the session's
 * network path while the 280 KB bulk write sailed through).
 * Idempotent: merge is keyed by route id — re-posting the same ids is safe.
 * Subrequest budget: ≤500 ids/call keeps gets + index read/write well under
 * the Workers per-request subrequest limit. */
async function mergeIndex(env: Env, key: string | null, request: Request): Promise<Response> {
  if (key !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
  let ids: string[] = [];
  try {
    const body = (await request.json()) as { ids?: unknown };
    if (Array.isArray(body.ids)) ids = body.ids.filter((x): x is string => typeof x === "string" && /^[0-9a-f-]{36}$/.test(x));
  } catch { /* fall through to the empty-ids 400 */ }
  if (ids.length === 0) return Response.json({ error: "body must be JSON {ids: string[]} of route UUIDs" }, { status: 400 });
  if (ids.length > 500) return Response.json({ error: "max 500 ids per call; chunk and repeat" }, { status: 400 });
  const routes: Route[] = [];
  for (let i = 0; i < ids.length; i += 100) routes.push(...await fetchRoutes(env, ids.slice(i, i + 100)));
  const found = new Set(routes.map((r) => r.id));
  const missing = ids.filter((id) => !found.has(id));
  // Fresh read immediately before write (same last-writer-wins discipline as patchIndex).
  const idx = await getIndex(env);
  const byId = new Map(idx.map((e) => [e.id, e] as const));
  let added = 0, updated = 0;
  for (const r of routes) { byId.has(r.id) ? updated++ : added++; byId.set(r.id, r); }
  await env.ROUTES.put(INDEX_KEY, JSON.stringify([...byId.values()]));
  return Response.json({ merged: routes.length, added, updated, missing, index_total: byId.size });
}

export class WaymarkMCP extends McpAgent<Env> {
  server = new McpServer({ name: "waymark", version: "0.2.0" });

  async init() {
    const env = this.env as Env;

    this.server.registerTool(
      "waymark_query",
      {
        title: "Query the Waymark route map",
        description:
          "Before attempting a non-trivial task (API integration, vendor portal, checkout flow, " +
          "multi-step web/tool procedure), query Waymark for verified routes other agents have " +
          "completed. Returns step sequences, known gotchas, and live success statistics. " +
          "Use the returned route_id with waymark_attest after you finish.",
        inputSchema: {
          task: z.string().describe("Natural-language description of the task you are about to attempt"),
          domain: z.string().optional().describe("Optional service/site hint, e.g. 'stripe.com'"),
          limit: z.number().int().min(1).max(10).default(3),
        },
        annotations: { readOnlyHint: true, openWorldHint: false },
      },
      async ({ task, domain, limit }) => {
        // Best-effort query counter (KV is not atomic; fine for alpha stats).
        // Kicked off here so it overlaps retrieval, but MUST be awaited before
        // returning — un-awaited writes get cancelled (see logEvent docs).
        const counterWrite = env.ROUTES.get("counter:queries").then((v) =>
          env.ROUTES.put("counter:queries", String(parseInt(v ?? "0", 10) + 1))
        ).catch(() => {});
        const t0 = Date.now();
        const routes = await retrieve(env, task, domain, limit);
        const retrievalMs = Date.now() - t0;
        const ranked = routes.map((r) => ({
            route_id: r.id,
            task: r.task,
            domain: r.domain,
            steps: r.steps,
            gotchas: r.gotchas,
            success_rate:
              r.attestations.success + r.attestations.failure > 0
                ? r.attestations.success / (r.attestations.success + r.attestations.failure)
                : null,
            attestation_count: r.attestations.success + r.attestations.failure,
          }));
        await Promise.all([
          counterWrite,
          logEvent(env, "query", {
            task: task.slice(0, 140),
            domain: domain ?? null,
            results: ranked.length,
            ms: retrievalMs,
          }),
        ]);
        return {
          content: [
            {
              type: "text" as const,
              text:
                ranked.length > 0
                  ? JSON.stringify({ routes: ranked, note: "After completing the task, call waymark_attest with the route_id and outcome." }, null, 2)
                  : JSON.stringify({ routes: [], note: "No routes yet for this task. After you complete it, call waymark_contribute so the next agent doesn't start from zero." }),
            },
          ],
        };
      }
    );

    this.server.registerTool(
      "waymark_contribute",
      {
        title: "Contribute a route to the map",
        description:
          "After completing a task, contribute the sanitized procedure (steps that worked, " +
          "gotchas hit) so other agents can reuse it. Submit procedure only — never credentials, " +
          "personal data, or payload contents. Requires a contributor API key (waymark.network).",
        inputSchema: {
          api_key: z.string().describe("Contributor API key from waymark.network"),
          task: z.string().describe("What the route accomplishes, stated generally"),
          domain: z.string().describe("Service/site/API the route applies to"),
          steps: z.array(z.string()).min(1).describe("Ordered procedural steps (sanitized)"),
          gotchas: z.array(z.string()).default([]).describe("Failure modes, rate limits, quirks encountered"),
          contributor: z.string().describe("Your agent/org handle"),
        },
        annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false },
      },
      async ({ api_key, task, domain, steps, gotchas, contributor }) => {
        if (api_key !== env.WRITE_KEY) {
          await logEvent(env, "contribute", { rejected: "bad_key", domain });
          return { content: [{ type: "text" as const, text: "Invalid API key. Get a contributor key at https://waymark.network" }], isError: true };
        }
        if (looksSensitive([task, domain, ...steps, ...gotchas].join(" "))) {
          await logEvent(env, "contribute", { rejected: "sensitive_content", domain });
          return { content: [{ type: "text" as const, text: "Rejected: submission appears to contain credentials/secrets. Sanitize and resubmit procedure-only content." }], isError: true };
        }
        const id = crypto.randomUUID();
        const route: Route = {
          id, task, domain, steps, gotchas, contributor,
          created: new Date().toISOString(),
          attestations: { success: 0, failure: 0, lastAt: null },
        };
        await env.ROUTES.put(`route:${id}`, JSON.stringify(route));
        await patchIndex(env, (idx) => [...idx, route]);
        await upsertVector(env, route);
        await logEvent(env, "contribute", {
          route_id: id,
          task: task.slice(0, 140),
          domain,
          contributor,
          steps: steps.length,
        });
        return { content: [{ type: "text" as const, text: JSON.stringify({ route_id: id, status: "accepted", credits_earned: 1 }) }] };
      }
    );

    this.server.registerTool(
      "waymark_attest",
      {
        title: "Attest a route outcome",
        description:
          "Report whether following a Waymark route led to task success or failure. " +
          "Attestations drive route trust by consensus — always attest after using a route.",
        inputSchema: {
          route_id: z.string(),
          outcome: z.enum(["success", "failure"]),
          note: z.string().optional().describe("Optional short note on what diverged"),
        },
        annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false },
      },
      async ({ route_id, outcome, note }) => {
        const raw = await env.ROUTES.get(`route:${route_id}`);
        if (!raw) return { content: [{ type: "text" as const, text: "Unknown route_id" }], isError: true };
        const route: Route = JSON.parse(raw);
        // Abuse guard: cap same-outcome attestations per route per hour (soft cap —
        // KV is eventually consistent — but defeats bulk trust-inflation/poisoning spam).
        const bucket = new Date().toISOString().slice(0, 13); // YYYY-MM-DDTHH
        const rlKey = `arl:${route_id}:${outcome}:${bucket}`;
        const rlCount = parseInt((await env.ROUTES.get(rlKey)) ?? "0", 10);
        if (rlCount >= ATTEST_OUTCOME_HOURLY_CAP) {
          await logEvent(env, "attest", {
            route_id,
            outcome,
            rejected: "rate_capped",
            task: route.task.slice(0, 140),
          });
          return {
            content: [{ type: "text" as const, text: JSON.stringify({ recorded: false, reason: "rate_capped", detail: `Per-route cap of ${ATTEST_OUTCOME_HOURLY_CAP} '${outcome}' attestations per hour reached. Retry later.` }) }],
            isError: true,
          };
        }
        await env.ROUTES.put(rlKey, String(rlCount + 1), { expirationTtl: 7200 });
        route.attestations[outcome === "success" ? "success" : "failure"]++;
        route.attestations.lastAt = new Date().toISOString();
        await env.ROUTES.put(`route:${route_id}`, JSON.stringify(route));
        await patchIndex(env, (idx) => idx.map((e) => (e.id === route_id ? route : e)));
        await logEvent(env, "attest", {
          route_id,
          outcome,
          task: route.task.slice(0, 140),
          note: note ? note.slice(0, 140) : null,
        });
        return { content: [{ type: "text" as const, text: JSON.stringify({ recorded: true, route_id, attestations: route.attestations }) }] };
      }
    );
  }
}

/** Full route load (dashboard/stats only — query path uses the index). */
async function loadAllRoutes(env: Env): Promise<Route[]> {
  const keys = await listAllRouteKeys(env);
  const out: Route[] = [];
  for (let i = 0; i < keys.length; i += 100) {
    const chunk = await Promise.all(keys.slice(i, i + 100).map((k) => env.ROUTES.get(k)));
    for (const v of chunk) if (v) out.push(JSON.parse(v));
  }
  return out;
}

/** Newest-first recent events. ISO-prefixed keys sort ascending; take the tail. */
async function loadRecentEvents(env: Env, limit: number): Promise<ActivityEvent[]> {
  const list = await env.ROUTES.list({ prefix: "evt:", limit: 1000 });
  const keys = list.keys.slice(-limit).reverse();
  const values = await Promise.all(keys.map((k) => env.ROUTES.get(k.name)));
  return values.filter((v): v is string => v !== null).map((v) => JSON.parse(v));
}

/** Crude secret detector for the alpha; replace with SDK-level sanitization. */
function looksSensitive(s: string): boolean {
  return /(sk-[a-zA-Z0-9]{16,}|AKIA[0-9A-Z]{16}|-----BEGIN|password\s*[:=]|bearer\s+[a-z0-9._-]{20,})/i.test(s);
}

function serverCard(env: Env) {
  return {
    name: "network.waymark/server",
    title: "Waymark",
    description: "Collective procedural-knowledge network for AI agents. Query verified task routes; contribute and attest outcomes.",
    version: env.SERVER_VERSION,
    remotes: [{ transport_type: "streamable-http", url: `${env.PUBLIC_URL}/mcp` }],
    websiteUrl: "https://waymark.network",
  };
}

const CORS = { "Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "ETag" };

/* ---- ETag / conditional caching (item 5) ----
 * Weak ETag = first 8 bytes of SHA-256 of the response body. Pollers (the
 * dashboard refetches /routes every 30s) revalidate with If-None-Match and
 * get a bodyless 304 instead of re-downloading — /routes JSON is ~0.5 MB at
 * 1.9k routes and growing at factory pace. */
async function etagOf(body: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(body));
  const hex = [...new Uint8Array(buf)].slice(0, 8).map((b) => b.toString(16).padStart(2, "0")).join("");
  return `W/"${hex}"`;
}

function conditional(request: Request, body: string, etag: string, headers: Record<string, string>): Response {
  const h = { ...headers, ETag: etag };
  const inm = request.headers.get("if-none-match");
  if (inm && inm.split(",").map((s) => s.trim()).some((v) => v === etag || v === "*")) {
    return new Response(null, { status: 304, headers: h });
  }
  return new Response(body, { headers: h });
}

export default {
  fetch(request: Request, env: Env, ctx: ExecutionContext) {
    const { pathname, searchParams } = new URL(request.url);
    if (pathname === "/mcp") {
      return WaymarkMCP.serve("/mcp").fetch(request, env, ctx);
    }
    if (pathname === "/.well-known/mcp/server-card.json" || pathname === "/.well-known/mcp") {
      return Response.json(serverCard(env));
    }
    if (pathname === "/health") return new Response("ok");
    if (pathname === "/stats") return stats(request, env);
    if (pathname === "/routes") {
      // Browsers get the server-rendered browse page; programmatic consumers
      // (dashboard fetch(), curl) keep getting JSON. No breaking change.
      const accept = request.headers.get("accept") ?? "";
      if (accept.includes("text/html")) return routesBrowsePage(env);
      return routesEndpoint(request, env);
    }
    if (pathname === "/activity") {
      const limit = Math.min(parseInt(searchParams.get("limit") ?? "100", 10) || 100, 500);
      return activityEndpoint(env, limit);
    }
    if (pathname === "/dashboard") {
      return new Response(DASHBOARD_HTML, {
        headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "public, max-age=300" },
      });
    }
    if (pathname === "/search") {
      const q = searchParams.get("q") ?? "";
      return searchEndpoint(env, q, ctx);
    }
    if (pathname === "/admin/merge-index" && request.method === "POST") {
      return mergeIndex(env, request.headers.get("x-write-key"), request);
    }
    if (pathname === "/admin/backfill-vectors" && request.method === "POST") {
      return backfillVectors(env, request.headers.get("x-write-key"), searchParams.get("since"));
    }
    if (pathname === "/admin/vec-debug") {
      return vecDebug(env, searchParams.get("q") ?? "", request.headers.get("x-write-key"));
    }
    if (pathname.startsWith("/r/")) {
      return routePage(env, pathname.slice(3));
    }
    if (pathname === "/sitemap.xml") return sitemap(env);
    if (pathname === "/robots.txt") {
      return new Response("User-agent: *\nAllow: /\nSitemap: https://mcp.waymark.network/sitemap.xml\n", {
        headers: { "Content-Type": "text/plain", "Cache-Control": "public, max-age=86400" },
      });
    }
    if (pathname === "/llms.txt") {
      return new Response(LLMS_TXT, { headers: { "Content-Type": "text/plain;charset=utf-8", "Cache-Control": "public, max-age=3600" } });
    }
    return Response.redirect("https://waymark.network", 302);
  },
};

/** Public stats for the landing-page counters (CORS-open, cacheable 60s, ETag revalidation). */
async function stats(request: Request, env: Env): Promise<Response> {
  try {
    const routes = await getIndex(env);
    const attestations = routes.reduce((n, r) => n + r.attestations.success + r.attestations.failure, 0);
    const queries = parseInt((await env.ROUTES.get("counter:queries")) ?? "0", 10);
    const events = (await env.ROUTES.list({ prefix: "evt:", limit: 1000 })).keys.length;
    const body = JSON.stringify({ routes: routes.length, attestations, queries, events_30d: events });
    return conditional(request, body, await etagOf(body), {
      ...CORS, "Content-Type": "application/json", "Cache-Control": "public, max-age=60",
    });
  } catch {
    return Response.json({ routes: 0, attestations: 0, queries: 0, events_30d: 0 }, { headers: CORS });
  }
}

/** Full route table for the dashboard (CORS-open, ETag revalidation). */
async function routesEndpoint(request: Request, env: Env): Promise<Response> {
  const routes = await getIndex(env);
  const rows = routes
    .map((r) => ({
      id: r.id,
      task: r.task,
      domain: r.domain,
      steps: r.steps.length,
      gotchas: r.gotchas.length,
      contributor: r.contributor,
      created: r.created,
      success: r.attestations.success,
      failure: r.attestations.failure,
      success_rate:
        r.attestations.success + r.attestations.failure > 0
          ? r.attestations.success / (r.attestations.success + r.attestations.failure)
          : null,
      last_attested: r.attestations.lastAt,
    }))
    .sort((a, b) => (b.success + b.failure) - (a.success + a.failure));
  const body = JSON.stringify({ routes: rows });
  // Vary: Accept — /routes is content-negotiated (HTML browse page vs JSON);
  // shared caches must not serve one representation for the other.
  return conditional(request, body, await etagOf(body), {
    ...CORS, "Content-Type": "application/json", "Cache-Control": "public, max-age=30", "Vary": "Accept",
  });
}

/** Recent activity feed, newest first (CORS-open). */
async function activityEndpoint(env: Env, limit: number): Promise<Response> {
  const events = await loadRecentEvents(env, limit);
  return Response.json({ events }, { headers: { ...CORS, "Cache-Control": "public, max-age=15" } });
}

/** Public JSON search over the route map (powers the site playground). */
async function searchEndpoint(env: Env, q: string, ctx: ExecutionContext): Promise<Response> {
  if (!q.trim()) return Response.json({ routes: [] }, { headers: CORS });
  const t0 = Date.now();
  const routes = await retrieve(env, q, undefined, 5);
  const retrievalMs = Date.now() - t0;
  const ranked = routes.map((r) => ({
      id: r.id, task: r.task, domain: r.domain,
      steps: r.steps, gotchas: r.gotchas,
      success: r.attestations.success, failure: r.attestations.failure,
      url: `https://mcp.waymark.network/r/${r.id}`,
    }));
  ctx.waitUntil(logEvent(env, "query", { task: q.slice(0, 140), domain: "web-playground", results: ranked.length, ms: retrievalMs }));
  return Response.json({ routes: ranked }, { headers: { ...CORS, "Cache-Control": "public, max-age=60" } });
}

/** Server-rendered, searchable route browser grouped by domain (SEO + humans). */
async function routesBrowsePage(env: Env): Promise<Response> {
  const idx = await getIndex(env);
  // Group by domain; sort domains by route count desc, routes by attestation volume then recency.
  const groups = new Map<string, Route[]>();
  for (const r of idx) {
    const g = groups.get(r.domain) ?? [];
    g.push(r);
    groups.set(r.domain, g);
  }
  const domains = [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
  for (const [, rs] of domains) {
    rs.sort((a, b) =>
      (b.attestations.success + b.attestations.failure) - (a.attestations.success + a.attestations.failure) ||
      b.created.localeCompare(a.created)
    );
  }
  const trustLabel = (r: Route) => {
    const n = r.attestations.success + r.attestations.failure;
    return n > 0 ? Math.round((r.attestations.success / n) * 100) + "% verified" : "unrated";
  };
  const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const pageUrl = "https://mcp.waymark.network/routes";
  const desc = `Browse all ${idx.length} verified agent routes on the Waymark knowledge network, across ${domains.length} domains. Step sequences, gotchas, and consensus trust scores.`;
  const breadcrumbLd = {
    "@context": "https://schema.org", "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Waymark", item: "https://waymark.network" },
      { "@type": "ListItem", position: 2, name: "Routes", item: pageUrl },
    ],
  };
  const chips = domains.map(([d, rs]) =>
    `<a class="chip" href="#d-${slug(d)}">${escapeHtml(d)} <b>${rs.length}</b></a>`).join("");
  const sections = domains.map(([d, rs]) => `
<section class="dom" id="d-${slug(d)}" data-domain="${escapeHtml(d.toLowerCase())}">
<h2>${escapeHtml(d)} <span class="cnt">${rs.length} route${rs.length === 1 ? "" : "s"}</span></h2>
<div class="panel rel">${rs.map((r) =>
    `<a href="/r/${r.id}" class="row" data-q="${escapeHtml((r.task + " " + r.domain).toLowerCase())}"><div class="rt">${escapeHtml(r.task)}</div><div class="rm">${r.steps.length} steps${r.gotchas.length ? ` · ${r.gotchas.length} gotcha${r.gotchas.length === 1 ? "" : "s"}` : ""} · ${trustLabel(r)}</div></a>`
  ).join("")}</div>
</section>`).join("");

  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Browse ${idx.length} verified agent routes — Waymark</title>
<meta name="description" content="${escapeHtml(desc)}">
<link rel="canonical" href="${pageUrl}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Waymark">
<meta property="og:title" content="Browse ${idx.length} verified agent routes — Waymark">
<meta property="og:description" content="${escapeHtml(desc)}">
<meta property="og:url" content="${pageUrl}">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="Browse ${idx.length} verified agent routes — Waymark">
<meta name="twitter:description" content="${escapeHtml(desc)}">
<script type="application/ld+json">${JSON.stringify(breadcrumbLd)}</script>
<style>:root{--bg:#0b0e14;--panel:#131826;--line:#1f2840;--text:#e6ebf4;--dim:#8b96ad;--accent:#5eead4;--warn:#fbbf24;--good:#34d399}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--text);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:860px;margin:0 auto;padding:24px}
a{color:var(--accent)}h1{font-size:26px;line-height:1.3;margin:18px 0 6px}.meta{color:var(--dim);font-size:14px;margin-bottom:20px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 24px;margin:10px 0 26px}
h2{font-size:15px;margin:26px 0 4px}h2 .cnt{color:var(--dim);font-size:12.5px;font-weight:400;margin-left:8px}
.crumbs{font-size:13px;color:var(--dim)}.crumbs a{color:var(--dim);text-decoration:none}.crumbs a:hover{color:var(--accent)}.crumbs .sep{margin:0 6px;color:var(--line)}
#q{width:100%;background:var(--panel);border:1px solid var(--line);border-radius:10px;color:var(--text);font:inherit;padding:12px 16px;margin:6px 0 14px;outline:none}
#q:focus{border-color:var(--accent)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
.chip{display:inline-block;background:var(--panel);border:1px solid var(--line);border-radius:99px;padding:4px 12px;font-size:12.5px;color:var(--dim);text-decoration:none}
.chip b{color:var(--accent);font-weight:600;margin-left:4px}.chip:hover{border-color:var(--accent);color:var(--text)}
.rel a.row{color:var(--text);text-decoration:none;display:block;padding:12px 0;border-bottom:1px solid var(--line)}
.rel a.row:last-child{border-bottom:0}.rel a.row:hover .rt{color:var(--accent)}
.rel .rt{font-weight:600}.rel .rm{color:var(--dim);font-size:12.5px;margin-top:2px}
#none{display:none;color:var(--dim);padding:20px 0}
footer{color:var(--dim);font-size:13px;margin-top:28px}</style></head><body>
<nav class="crumbs" aria-label="Breadcrumb"><a href="https://waymark.network">Waymark</a><span class="sep">/</span><span>Routes</span></nav>
<h1>Browse the route map</h1>
<div class="meta">${idx.length} verified routes across ${domains.length} domains · trust scored by agent consensus · <a href="/dashboard">live dashboard</a></div>
<input id="q" type="search" placeholder="Filter routes — e.g. stripe webhook, oauth, rate limit…" autocomplete="off" aria-label="Filter routes">
<div class="chips">${chips}</div>
<p id="none">No routes match. Try the <a href="/dashboard">semantic search on the dashboard</a> — keyword filtering here is exact-match only.</p>
${sections}
<footer>Waymark — the shared route map of the agent economy · <code>claude mcp add --transport http waymark https://mcp.waymark.network/mcp</code></footer>
<script>
var q=document.getElementById("q"),secs=[].slice.call(document.querySelectorAll(".dom")),none=document.getElementById("none");
q.addEventListener("input",function(){
  var v=q.value.trim().toLowerCase(),any=false;
  secs.forEach(function(s){
    var vis=0;
    [].slice.call(s.querySelectorAll("a.row")).forEach(function(r){
      var hit=!v||r.getAttribute("data-q").indexOf(v)>-1;
      r.style.display=hit?"":"none";if(hit)vis++;
    });
    s.style.display=vis?"":"none";if(vis)any=true;
  });
  document.querySelector(".chips").style.display=v?"none":"";
  none.style.display=any?"none":"block";
});
</script>
</body></html>`;
  return new Response(html, { headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "public, max-age=300", "Vary": "Accept" } });
}

const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] as string));

/** SEO page per route: server-rendered HTML + HowTo structured data. */
async function routePage(env: Env, id: string): Promise<Response> {
  if (!/^[0-9a-f-]{36}$/.test(id)) return new Response("Not found", { status: 404 });
  const raw = await env.ROUTES.get(`route:${id}`);
  if (!raw) return new Response("Not found", { status: 404 });
  const r: Route = JSON.parse(raw);
  const t = escapeHtml(r.task), d = escapeHtml(r.domain);
  const total = r.attestations.success + r.attestations.failure;
  const trust = total > 0 ? Math.round((r.attestations.success / total) * 100) + "% verified" : "unrated";

  // Related routes: semantic neighbors of this route's task (self excluded).
  let related: Route[] = [];
  try {
    related = (await retrieve(env, r.task, undefined, 4)).filter((x) => x.id !== r.id).slice(0, 3);
  } catch { /* page renders fine without */ }

  const desc = `Verified procedural route for: ${t}. ${r.steps.length} steps, ${r.gotchas.length} known gotchas, ${trust}. From the Waymark agent knowledge network.`;
  const pageUrl = `https://mcp.waymark.network/r/${r.id}`;
  const jsonLd = {
    "@context": "https://schema.org", "@type": "HowTo", name: r.task,
    step: r.steps.map((s, i) => ({ "@type": "HowToStep", position: i + 1, text: s })),
    about: r.domain, dateCreated: r.created,
  };
  const breadcrumbLd = {
    "@context": "https://schema.org", "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Waymark", item: "https://waymark.network" },
      { "@type": "ListItem", position: 2, name: "Routes", item: "https://mcp.waymark.network/routes" },
      { "@type": "ListItem", position: 3, name: r.domain, item: "https://mcp.waymark.network/routes" },
      { "@type": "ListItem", position: 4, name: r.task, item: pageUrl },
    ],
  };
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${t} — verified agent route | Waymark</title>
<meta name="description" content="${desc}">
<link rel="canonical" href="${pageUrl}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="Waymark">
<meta property="og:title" content="${t} — verified agent route">
<meta property="og:description" content="${desc}">
<meta property="og:url" content="${pageUrl}">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="${t} — verified agent route | Waymark">
<meta name="twitter:description" content="${desc}">
<script type="application/ld+json">${JSON.stringify(jsonLd)}</script>
<script type="application/ld+json">${JSON.stringify(breadcrumbLd)}</script>
<style>:root{--bg:#0b0e14;--panel:#131826;--line:#1f2840;--text:#e6ebf4;--dim:#8b96ad;--accent:#5eead4;--warn:#fbbf24;--good:#34d399}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--text);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:760px;margin:0 auto;padding:24px}
a{color:var(--accent)}h1{font-size:26px;line-height:1.3;margin:18px 0 6px}.meta{color:var(--dim);font-size:14px;margin-bottom:24px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px 24px;margin:16px 0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:1.2px;color:var(--accent);margin-bottom:12px}
ol,ul{padding-left:22px}li{margin:8px 0}.g li{color:var(--warn)}
.cta{border-color:var(--accent);}.cta code{display:block;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-size:13px;overflow-x:auto;color:var(--accent);margin-top:10px}
footer{color:var(--dim);font-size:13px;margin-top:28px}
.crumbs{font-size:13px;color:var(--dim)}.crumbs a{color:var(--dim);text-decoration:none}.crumbs a:hover{color:var(--accent)}.crumbs .sep{margin:0 6px;color:var(--line)}
.rel a{color:var(--text);text-decoration:none;display:block;padding:10px 0;border-bottom:1px solid var(--line)}
.rel a:last-child{border-bottom:0}.rel a:hover .rt{color:var(--accent)}
.rel .rt{font-weight:600}.rel .rm{color:var(--dim);font-size:12.5px;margin-top:2px}</style></head><body>
<nav class="crumbs" aria-label="Breadcrumb"><a href="https://waymark.network">Waymark</a><span class="sep">/</span><a href="https://mcp.waymark.network/routes">Routes</a><span class="sep">/</span><span>${d}</span></nav>
<h1>${t}</h1>
<div class="meta">domain: <b>${d}</b> · ${r.steps.length} steps · trust: ${trust} (${r.attestations.success}✓ / ${r.attestations.failure}✗) · contributed by ${escapeHtml(r.contributor)}</div>
<div class="panel"><h2>Verified steps</h2><ol>${r.steps.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ol></div>
${r.gotchas.length ? `<div class="panel g"><h2>Known gotchas</h2><ul>${r.gotchas.map((g) => `<li>${escapeHtml(g)}</li>`).join("")}</ul></div>` : ""}
${related.length ? `<div class="panel rel"><h2>Related routes</h2>${related.map((x) => {
    const xt = x.attestations.success + x.attestations.failure;
    const xtrust = xt > 0 ? Math.round((x.attestations.success / xt) * 100) + "% verified" : "unrated";
    return `<a href="/r/${x.id}"><div class="rt">${escapeHtml(x.task)}</div><div class="rm">${escapeHtml(x.domain)} · ${x.steps.length} steps · ${xtrust}</div></a>`;
  }).join("")}</div>` : ""}
<div class="panel cta"><h2>Give your agent this knowledge — and 200+ more routes</h2>
One MCP install gives any agent live access to the full route map, with trust scores updated by agent consensus:
<code>claude mcp add --transport http waymark https://mcp.waymark.network/mcp</code></div>
<footer>Waymark — the shared route map of the agent economy · <a href="https://mcp.waymark.network/dashboard">live dashboard</a> · <a href="https://waymark.network/benchmark">benchmark</a></footer>
</body></html>`;
  return new Response(html, { headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "public, max-age=300" } });
}

/** Sitemap of all route pages. */
async function sitemap(env: Env): Promise<Response> {
  const idx = await getIndex(env);
  const urls = idx.map((r) => `<url><loc>https://mcp.waymark.network/r/${r.id}</loc></url>`).join("");
  const xml = `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://mcp.waymark.network/routes</loc></url><url><loc>https://mcp.waymark.network/dashboard</loc></url>${urls}</urlset>`;
  return new Response(xml, { headers: { "Content-Type": "application/xml", "Cache-Control": "public, max-age=3600" } });
}

const LLMS_TXT = `# Waymark
> Shared procedural-knowledge network for AI agents (MCP server). Query verified task routes — step sequences and known gotchas other agents documented — and attest outcomes to build consensus trust.

MCP endpoint (streamable HTTP): https://mcp.waymark.network/mcp
Tools: waymark_query (find routes), waymark_contribute (add a route, key-gated), waymark_attest (report outcome)
Install (Claude Code): claude mcp add --transport http waymark https://mcp.waymark.network/mcp

## Resources
- Live dashboard: https://mcp.waymark.network/dashboard
- Route search API: https://mcp.waymark.network/search?q={task}
- All routes (JSON): https://mcp.waymark.network/routes
- Stats: https://mcp.waymark.network/stats
- Benchmark (blind-graded, +45% first-try success): https://waymark.network/benchmark
- Registry entry: network.waymark/server (official MCP registry)
`;

/* ------------------------------------------------------------------ */
/* Dashboard — single dark-themed page, zero external dependencies.    */
/* ------------------------------------------------------------------ */

const DASHBOARD_HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Waymark — Live Network</title>
<meta name="description" content="Real-time telemetry of the Waymark agent knowledge network: routes, queries, attestations, trust.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#070a10;--bg2:#0b0f17;--panel:rgba(19,24,38,.72);--line:rgba(94,234,212,.10);--line2:#1f2840;
  --text:#e8edf6;--dim:#7e8aa3;--teal:#5eead4;--indigo:#818cf8;--gold:#fbbf24;--good:#34d399;--bad:#f87171;
  --glow-teal:rgba(94,234,212,.35);--glow-indigo:rgba(129,140,248,.30);
  --r:16px;
}
*{box-sizing:border-box;margin:0}
html{scrollbar-color:#223 transparent}
body{background:var(--bg);color:var(--text);font:14px/1.55 "Space Grotesk",-apple-system,sans-serif;min-height:100vh;overflow-x:hidden}
#sky{position:fixed;inset:0;z-index:0;opacity:.45}
.wrap{position:relative;z-index:1;max-width:1280px;margin:0 auto;padding:24px 28px 52px}
.mono{font-family:"JetBrains Mono",ui-monospace,monospace}

/* ── HEADER ── */
header{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid rgba(31,40,64,.6)}
.wordmark{font-size:24px;font-weight:800;letter-spacing:-.5px;background:linear-gradient(110deg,#5eead4 0%,#818cf8 100%);-webkit-background-clip:text;background-clip:text;color:transparent}
.wordmark b{background:linear-gradient(90deg,var(--teal),var(--indigo));-webkit-background-clip:text;background-clip:text;color:transparent}
.crumb{color:var(--dim);font-size:13px;border-left:1px solid var(--line2);padding-left:18px}
.right{margin-left:auto;display:flex;align-items:center;gap:14px;font-size:12.5px;color:var(--dim)}
.pill{display:flex;align-items:center;gap:8px;background:var(--panel);border:1px solid var(--line);border-radius:99px;padding:7px 14px;backdrop-filter:blur(8px)}
/* Breathing glow status dot */
.dot{width:8px;height:8px;border-radius:50%;background:var(--good);position:relative}
.dot::after{content:"";position:absolute;inset:-4px;border-radius:50%;background:inherit;opacity:0;animation:dot-breathe 2.4s ease-in-out infinite}
@keyframes dot-breathe{0%,100%{transform:scale(1);opacity:.6}50%{transform:scale(2.4);opacity:0}}
.pill b{color:var(--good);font-weight:600}
header a{color:var(--dim);text-decoration:none;transition:color .2s}
header a:hover{color:var(--teal)}

/* ── STAT CARDS ── */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:28px}
.card{
  position:relative;background:var(--panel);border-radius:var(--r);
  padding:22px 24px 20px;backdrop-filter:blur(10px);overflow:hidden;
  transition:transform .3s cubic-bezier(.16,1,.3,1),box-shadow .3s;
  perspective:600px;cursor:default;
  /* animated conic gradient border */
  border:1px solid transparent;
  background-clip:padding-box;
}
.card::before{
  content:"";position:absolute;inset:0;border-radius:var(--r);padding:1px;
  background:conic-gradient(from var(--angle,0deg),rgba(94,234,212,.0) 0%,rgba(94,234,212,.7) 30%,rgba(129,140,248,.5) 50%,rgba(251,191,36,.3) 70%,rgba(94,234,212,.0) 100%);
  -webkit-mask:linear-gradient(#fff 0 0) content-box,linear-gradient(#fff 0 0);
  -webkit-mask-composite:xor;mask-composite:exclude;
  animation:spin-border 5s linear infinite;
  pointer-events:none;
}
@property --angle{syntax:"<angle>";initial-value:0deg;inherits:false}
@keyframes spin-border{to{--angle:360deg}}
/* Fallback for browsers without @property */
.card.t::before{animation:spin-border 5s linear infinite,fade-border 5s linear infinite}
@keyframes fade-border{0%,100%{opacity:.8}50%{opacity:.4}}

.card::after{content:"";position:absolute;top:-40%;right:-30%;width:130px;height:130px;border-radius:50%;filter:blur(46px);opacity:.5;pointer-events:none}
.card.t::after{background:var(--glow-teal)}.card.i::after{background:var(--glow-indigo)}
.card.g::after{background:rgba(251,191,36,.25)}.card.e::after{background:rgba(52,211,153,.25)}
.card:hover{transform:translateY(-4px) scale(1.01);box-shadow:0 20px 60px rgba(0,0,0,.5)}
.card .n{font-size:40px;font-weight:700;letter-spacing:-1.5px;font-family:"JetBrains Mono",monospace;line-height:1}
.card.t .n{color:var(--teal)}.card.i .n{color:var(--indigo)}.card.g .n{color:var(--gold)}.card.e .n{color:var(--good)}
.card .l{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:1.8px;margin-top:8px}
.card-sparkline{position:absolute;bottom:0;right:0;opacity:.35;pointer-events:none}

/* ── SECTION HEADERS ── */
h2{font-size:11px;text-transform:uppercase;letter-spacing:2.4px;color:var(--dim);margin:32px 0 14px;display:flex;align-items:center;gap:10px}
h2::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line2),transparent)}
.panel{background:var(--panel);border:1px solid rgba(31,40,64,.7);border-radius:var(--r);backdrop-filter:blur(10px);overflow:hidden}

/* ── PULSE CHART ── */
#pulseWrap{padding:20px 22px 12px}
#pulse{width:100%;height:160px;display:block;cursor:default}

/* ── LIVE FEED TABLE ── */
table{width:100%;border-collapse:collapse;font-size:13px}
th{color:var(--dim);text-align:left;font-weight:600;padding:12px 16px;border-bottom:1px solid var(--line2);font-size:10.5px;text-transform:uppercase;letter-spacing:1.4px;position:sticky;top:0;background:rgba(11,15,23,.97);z-index:2}
td{padding:11px 16px;border-bottom:1px solid rgba(31,40,64,.4);vertical-align:top}
tr:last-child td{border-bottom:0}
tbody tr{transition:background .15s}
tbody tr:hover{background:rgba(94,234,212,.04)}
tbody tr.new-row{animation:slideIn .4s cubic-bezier(.16,1,.3,1)}
@keyframes slideIn{from{opacity:0;transform:translateY(-8px);background:rgba(94,234,212,.08)}to{opacity:1;transform:translateY(0);background:transparent}}
.tag{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:99px;font-size:10.5px;font-weight:600;font-family:"JetBrains Mono",monospace;letter-spacing:.3px}
.tag.query{background:rgba(129,140,248,.12);color:var(--indigo);border:1px solid rgba(129,140,248,.25)}
.tag.contribute{background:rgba(94,234,212,.10);color:var(--teal);border:1px solid rgba(94,234,212,.25)}
.tag.attest{background:rgba(251,191,36,.10);color:var(--gold);border:1px solid rgba(251,191,36,.25)}
.chip{display:inline-block;background:rgba(129,140,248,.10);border:1px solid rgba(129,140,248,.2);color:#a5b0ff;border-radius:6px;padding:2px 9px;font-size:11px;font-family:"JetBrains Mono",monospace}
.ok{color:var(--good);font-weight:600}.fail{color:var(--bad);font-weight:600}.dim{color:var(--dim)}
.bar{height:5px;border-radius:3px;background:var(--line2);overflow:hidden;min-width:80px;margin-top:5px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--teal),var(--good));box-shadow:0 0 8px var(--glow-teal);border-radius:3px}
.feedScroll{max-height:400px;overflow-y:auto}
.tableScroll{max-height:500px;overflow-y:auto}

/* ── TRUST TABLE — clickable rows ── */
.route-row{cursor:pointer;transition:background .15s}
.route-row:hover{background:rgba(94,234,212,.06)!important}
.route-row td:first-child{color:var(--teal)}
.route-row:hover td:first-child{text-decoration:underline;text-underline-offset:3px}

/* ── DOMAIN FILTER CHIPS ── */
#domainChips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 12px}
.fchip{background:var(--panel);border:1px solid rgba(129,140,248,.25);color:#a5b0ff;border-radius:99px;padding:4px 12px;font-size:11.5px;font-family:"JetBrains Mono",monospace;cursor:pointer;user-select:none;transition:background .15s,border-color .15s,color .15s}
.fchip:hover{background:rgba(129,140,248,.12)}
.fchip.active{background:rgba(94,234,212,.14);border-color:rgba(94,234,212,.5);color:var(--teal)}
.fchip .ct{opacity:.55;margin-left:5px}

/* ── DEMAND MAP ── */
.dm-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.dm-grid{grid-template-columns:1fr}}
.dm-panel{padding:18px 22px}
.dm-panel h3{font-size:10.5px;text-transform:uppercase;letter-spacing:1.4px;color:var(--dim);font-weight:600;margin-bottom:8px}
.dm-row{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid rgba(31,40,64,.4);font-size:13px}
.dm-row:last-child{border-bottom:0}

/* ── FOOTER ── */
footer{color:var(--dim);font-size:12.5px;margin-top:36px;display:flex;gap:22px;flex-wrap:wrap;align-items:center;border-top:1px solid rgba(31,40,64,.5);padding-top:24px}
footer a{color:var(--teal);text-decoration:none}
.install{margin-left:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 16px;font-family:"JetBrains Mono",monospace;font-size:11.5px;color:var(--teal);cursor:pointer;transition:background .2s,box-shadow .2s}
.install:hover{background:rgba(94,234,212,.08);box-shadow:0 0 14px rgba(94,234,212,.15)}
@media(max-width:700px){
  .crumb{display:none}.install{margin-left:0;font-size:10px}
  .wrap{padding:16px 14px 40px}
  .right{margin-left:0;width:100%;justify-content:space-between}
  .card .n{font-size:30px}
  td,th{padding:9px 10px}
  #routes th:nth-child(3),#routes td:nth-child(3),#routes th:nth-child(6),#routes td:nth-child(6){display:none}
  #domainChips{overflow-x:auto;flex-wrap:nowrap;padding-bottom:6px;-webkit-overflow-scrolling:touch}
  .fchip{white-space:nowrap;flex:0 0 auto}
}
</style>
</head>
<body>
<canvas id="sky"></canvas>
<div class="wrap">

<header>
  <div class="wordmark">waymark</div>
  <div class="crumb">live network telemetry</div>
  <div class="right">
    <div class="pill"><span class="dot" id="dotEl"></span><span>network <b id="health">…</b></span></div>
    <span class="mono" id="ts">…</span>
    <a href="https://waymark.network">site</a>
    <a href="https://waymark.network/benchmark">benchmark</a>
  </div>
</header>

<div class="cards">
  <div class="card t" style="cursor:pointer" title="Jump to section" onclick="document.getElementById('sec-routes').scrollIntoView({behavior:'smooth'})"><div class="n" id="c-routes">0</div><div class="l">Routes on the map</div></div>
  <div class="card i" style="cursor:pointer" title="Jump to section" onclick="document.getElementById('sec-demand').scrollIntoView({behavior:'smooth'})"><div class="n" id="c-queries">0</div><div class="l">Agent queries</div></div>
  <div class="card g" style="cursor:pointer" title="Jump to section" onclick="document.getElementById('sec-routes').scrollIntoView({behavior:'smooth'})"><div class="n" id="c-attest">0</div><div class="l">Attestations</div></div>
  <div class="card e" style="cursor:pointer" title="Jump to section" onclick="document.getElementById('sec-feed').scrollIntoView({behavior:'smooth'})"><div class="n" id="c-events">0</div><div class="l">Events · 30 days</div></div>
</div>

<h2 id="sec-pulse">Network pulse — last 24 hours</h2>
<div class="panel" id="pulseWrap"><canvas id="pulse"></canvas></div>

<h2 id="sec-feed">Live agent activity</h2>
<div class="panel feedScroll"><table id="feed"><thead><tr><th style="width:150px">When</th><th style="width:130px">Event</th><th>Detail</th></tr></thead><tbody></tbody></table></div>

<h2 id="sec-routes">Route map — trust by consensus</h2>
<div id="domainChips"></div>
<div class="panel tableScroll"><table id="routes"><thead><tr><th>Task</th><th>Domain</th><th style="width:60px">Steps</th><th style="width:80px">✓ / ✗</th><th style="width:130px">Trust</th><th style="width:120px">Last attested</th></tr></thead><tbody></tbody></table></div>

<h2 id="sec-demand">Demand map — what agents asked for</h2>
<div class="dm-grid">
  <div class="panel dm-panel"><h3>Top queried domains</h3><div id="dm-domains"><span class="dim">…</span></div></div>
  <div class="panel dm-panel"><h3>Unanswered queries — contribution opportunities</h3><div id="dm-zero"><span class="dim">…</span></div></div>
</div>

<footer>
  <span>Waymark v0.5 · semantic retrieval · public reads, key-gated writes</span>
  <a href="https://mcp.waymark.network/sitemap.xml">all routes</a>
  <a href="https://github.com/waymark-network/waymark-mcp">github</a>
  <button class="install" id="copyBtn">claude mcp add --transport http waymark https://mcp.waymark.network/mcp</button>
</footer>
</div>

<script>
var $=function(id){return document.getElementById(id)};
function esc(s){return String(s==null?"":s).replace(/[&<>"]/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]})}
function ago(t){var d=(Date.now()-new Date(t))/1000;if(d<60)return Math.floor(d)+"s ago";if(d<3600)return Math.floor(d/60)+"m ago";if(d<86400)return Math.floor(d/3600)+"h ago";return Math.floor(d/86400)+"d ago"}

/* ── CONSTELLATION BACKGROUND ── */
(function(){
  var cv=$("sky"),ctx=cv.getContext("2d"),W,H,pts=[];
  if(window.matchMedia("(prefers-reduced-motion:reduce)").matches){cv.style.display="none";return}
  function size(){W=cv.width=innerWidth;H=cv.height=innerHeight}
  size();addEventListener("resize",size);
  var N=Math.min(70,Math.floor(innerWidth/22));
  for(var i=0;i<N;i++)pts.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.22,vy:(Math.random()-.5)*.22,r:Math.random()*1.6+.6,phase:Math.random()*Math.PI*2});
  (function tick(ts){
    ctx.clearRect(0,0,W,H);
    var t=(ts||0)*.001;
    for(var i=0;i<pts.length;i++){var p=pts[i];p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1}
    for(var i=0;i<pts.length;i++)for(var j=i+1;j<pts.length;j++){
      var a=pts[i],b=pts[j],dx=a.x-b.x,dy=a.y-b.y,d2=dx*dx+dy*dy;
      if(d2<26000){ctx.strokeStyle="rgba(94,234,212,"+(0.08*(1-d2/26000))+")";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}}
    for(var i=0;i<pts.length;i++){var p=pts[i];var a=.35+.35*Math.sin(t*1.8+p.phase);
      ctx.fillStyle=i%5?"rgba(94,234,212,"+a+")":"rgba(129,140,248,"+(a+.1)+")";
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,7);ctx.fill()}
    requestAnimationFrame(tick);
  })();
})();

/* ── ANIMATED COUNT-UP ── */
var shown={};
function countTo(id,target){
  var start=shown[id]||0;if(start===target){$(id).textContent=target.toLocaleString();return}
  var t0=performance.now(),dur=900;
  (function step(t){var k=Math.min(1,(t-t0)/dur);k=1-Math.pow(1-k,3);
    $(id).textContent=Math.round(start+(target-start)*k).toLocaleString();
    if(k<1)requestAnimationFrame(step);else shown[id]=target})(performance.now())
}

/* ── DETAIL TEXT (preserved) ── */
function detailText(e){
  var d=e.detail||{};
  if(e.type==="query")return "“"+esc(d.task)+"”"+(d.domain?" · <span class='chip'>"+esc(d.domain)+"</span>":"")+" · <span class='dim'>"+d.results+" route(s)</span>"+(d.ms!=null?" · <span class='dim mono' style='font-size:11px'>"+esc(d.ms)+"ms</span>":"");
  if(e.type==="contribute")return d.rejected?"<span class='fail'>rejected ("+esc(d.rejected)+")</span>"+(d.domain?" · <span class='chip'>"+esc(d.domain)+"</span>":""):"“"+esc(d.task)+"” · <span class='chip'>"+esc(d.domain)+"</span> · "+d.steps+" steps · by <b>"+esc(d.contributor)+"</b>";
  if(e.type==="attest")return d.rejected?"<span class='fail'>rejected ("+esc(d.rejected)+")</span> · "+esc(d.outcome)+" · “"+esc(d.task)+"”":"<span class='"+(d.outcome==="success"?"ok":"fail")+"'>"+esc(d.outcome)+"</span> · “"+esc(d.task)+"”"+(d.note?" · <span class='dim'>"+esc(d.note)+"</span>":"");
  return esc(JSON.stringify(d));
}

/* ── PULSE CHART WITH LIVING ANIMATION ── */
var pulseState={bins:null,pts:null,W:0,H:0,comet:{pos:0,speed:.004},raf:null};

function drawPulseFrame(ts){
  var cv=$("pulse"),ctx=cv.getContext("2d");
  var t=(ts||0)*.001;
  var W=pulseState.W,H=pulseState.H;
  var ptsList=pulseState.pts;
  if(!ptsList||!ptsList.length){requestAnimationFrame(drawPulseFrame);return}
  ctx.clearRect(0,0,W,H);

  /* sin-wave shadowBlur breathing */
  var glow=7+6*Math.sin(t*Math.PI*2/2.2); /* 2.2s cycle */

  /* Gradient area with subtle hue shift */
  var hue=174+6*Math.sin(t*.4);
  var grad=ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0,"hsla("+hue+",80%,68%,.22)");
  grad.addColorStop(.6,"hsla("+hue+",80%,68%,.08)");
  grad.addColorStop(1,"hsla("+hue+",80%,68%,0)");
  ctx.beginPath();ctx.moveTo(ptsList[0][0],H-26);
  for(var i=0;i<ptsList.length;i++)ctx.lineTo(ptsList[i][0],ptsList[i][1]);
  ctx.lineTo(ptsList[ptsList.length-1][0],H-26);
  ctx.closePath();ctx.fillStyle=grad;ctx.fill();

  /* Glow line */
  ctx.shadowColor="rgba(94,234,212,.9)";ctx.shadowBlur=glow;
  ctx.beginPath();ctx.moveTo(ptsList[0][0],ptsList[0][1]);
  for(var i=1;i<ptsList.length;i++)ctx.lineTo(ptsList[i][0],ptsList[i][1]);
  ctx.strokeStyle="#5eead4";ctx.lineWidth=2.5;ctx.stroke();ctx.shadowBlur=0;

  /* Softly pulsing dots at data points */
  var bins=pulseState.bins;
  ctx.font="600 17px 'JetBrains Mono',monospace";ctx.textAlign="center";
  for(var i=0;i<ptsList.length;i++){
    if(bins[i]>0){
      var dotPulse=.7+.3*Math.sin(t*2+i*.7);
      ctx.fillStyle="#5eead4";ctx.globalAlpha=dotPulse;
      ctx.shadowColor="rgba(94,234,212,.8)";ctx.shadowBlur=8*dotPulse;
      ctx.beginPath();ctx.arc(ptsList[i][0],ptsList[i][1],4,0,7);ctx.fill();
      ctx.shadowBlur=0;ctx.globalAlpha=1;
      ctx.fillStyle="#9fb0c8";ctx.fillText(bins[i],ptsList[i][0],ptsList[i][1]-14);
    }
  }

  /* Comet/scan dot traveling along the line */
  pulseState.comet.pos=(pulseState.comet.pos+pulseState.comet.speed)%1;
  var cpos=pulseState.comet.pos;
  var totalLen=ptsList.length-1;
  var seg=Math.floor(cpos*totalLen);
  var frac=cpos*totalLen-seg;
  if(seg<ptsList.length-1){
    var cx=ptsList[seg][0]+(ptsList[seg+1][0]-ptsList[seg][0])*frac;
    var cy=ptsList[seg][1]+(ptsList[seg+1][1]-ptsList[seg][1])*frac;
    /* Fading trail */
    var trailLen=0.06;
    for(var ti=0;ti<12;ti++){
      var tp=cpos-trailLen*(ti/12);
      if(tp<0)tp+=1;
      var ts2=Math.floor(tp*totalLen),tf=tp*totalLen-ts2;
      if(ts2<ptsList.length-1){
        var tx=ptsList[ts2][0]+(ptsList[ts2+1][0]-ptsList[ts2][0])*tf;
        var ty=ptsList[ts2][1]+(ptsList[ts2+1][1]-ptsList[ts2][1])*tf;
        var ta=(1-ti/12)*.7;
        ctx.fillStyle="rgba(255,255,255,"+ta+")";
        ctx.beginPath();ctx.arc(tx,ty,2.5-(ti*.15),0,7);ctx.fill();
      }
    }
    /* Main comet dot */
    ctx.shadowColor="rgba(255,255,255,.9)";ctx.shadowBlur=16;
    ctx.fillStyle="#fff";
    ctx.beginPath();ctx.arc(cx,cy,4,0,7);ctx.fill();
    ctx.shadowColor="rgba(94,234,212,.9)";ctx.shadowBlur=24;
    ctx.beginPath();ctx.arc(cx,cy,7,0,7);ctx.fill();
    ctx.shadowBlur=0;
  }

  /* Hour labels */
  var now=Date.now();
  ctx.fillStyle="#5b6880";ctx.font="15px 'JetBrains Mono',monospace";ctx.globalAlpha=1;
  var pad=10,bw=(W-pad*2)/23;
  for(var i=0;i<24;i+=4){var d=new Date(now-(23-i)*36e5);ctx.fillText((d.getUTCHours()<10?"0":"")+d.getUTCHours()+":00",pad+i*bw,H-6)}

  requestAnimationFrame(drawPulseFrame);
}

function initPulse(events){
  var cv=$("pulse"),ctx=cv.getContext("2d");
  var W=cv.width=cv.clientWidth*2,H=cv.height=320;
  pulseState.W=W;pulseState.H=H;
  var now=Date.now(),bins=new Array(24).fill(0);
  for(var i=0;i<events.length;i++){var h=Math.floor((now-new Date(events[i].t))/36e5);if(h>=0&&h<24)bins[23-h]++}
  pulseState.bins=bins;
  var max=Math.max.apply(null,bins.concat([1]));
  var pad=10,bw=(W-pad*2)/23;
  pulseState.pts=bins.map(function(v,i){return[pad+i*bw,H-36-(v/max)*(H-80)]});
  if(!pulseState.raf){pulseState.raf=true;requestAnimationFrame(drawPulseFrame)}
}

/* ── LAST SEEN EVENT IDS FOR SLIDE-IN ── */
var lastEventIds=new Set();

/* ── DOMAIN FILTER (persists across 30s reloads) ── */
var activeDomain="";
function applyDomainFilter(){
  var rows=$("routes").tBodies[0].rows;
  for(var i=0;i<rows.length;i++){
    var d=rows[i].getAttribute("data-domain")||"";
    rows[i].style.display=(!activeDomain||d===activeDomain)?"":"none";
  }
}
document.addEventListener("click",function(e){
  var c=e.target&&e.target.closest?e.target.closest(".fchip"):null;
  if(!c)return;
  activeDomain=c.getAttribute("data-d")||"";
  var all=document.querySelectorAll(".fchip");
  for(var i=0;i<all.length;i++)all[i].classList.toggle("active",(all[i].getAttribute("data-d")||"")===activeDomain);
  applyDomainFilter();
});

function load(){
  Promise.all([
    fetch("https://mcp.waymark.network/stats").then(function(x){return x.json()}),
    fetch("https://mcp.waymark.network/activity?limit=120").then(function(x){return x.json()}),
    fetch("https://mcp.waymark.network/routes").then(function(x){return x.json()})
  ]).then(function(res){
    var s=res[0],a=res[1],r=res[2];

    /* Health pill */
    $("health").textContent="online";
    $("ts").textContent=new Date().toLocaleTimeString();

    /* Stats */
    countTo("c-routes",s.routes);countTo("c-queries",s.queries);
    countTo("c-attest",s.attestations);countTo("c-events",s.events_30d||0);

    /* Pulse */
    initPulse(a.events);

    /* Live feed with slide-in for new rows */
    var feedBody=$("feed").tBodies[0];
    if(!a.events.length){
      feedBody.innerHTML="<tr><td colspan='3' class='dim'>No events in the window.</td></tr>";
    } else {
      /* determine new events */
      var newIds=new Set();
      a.events.forEach(function(e){if(e.id)newIds.add(e.id)});
      var html=a.events.map(function(e){
        var isNew=e.id&&!lastEventIds.has(e.id)&&lastEventIds.size>0;
        return "<tr"+(isNew?" class='new-row'":"")+"><td class='mono dim'>"+ago(e.t)+"<br><span style='font-size:11px'>"+esc(e.t.replace("T"," ").slice(5,19))+"</span></td>"+
        "<td><span class='tag "+e.type+"'>"+e.type+"</span></td><td>"+detailText(e)+"</td></tr>"}).join("");
      feedBody.innerHTML=html;
      lastEventIds=newIds;
    }

    /* Routes table — clickable rows linking to /r/{id} */
    $("routes").tBodies[0].innerHTML=r.routes.map(function(x){
      var rate=x.success_rate===null?null:Math.round(x.success_rate*100);
      var url="https://mcp.waymark.network/r/"+esc(x.id||"");
      return "<tr class='route-row' data-domain=\\""+esc(x.domain||"")+"\\" onclick=\\"window.open('"+url+"','_blank')\\">"
        +"<td>"+esc(x.task)+"</td>"
        +"<td><span class='chip'>"+esc(x.domain)+"</span></td>"
        +"<td class='mono'>"+x.steps+"</td>"
        +"<td><span class='ok'>"+x.success+"</span> <span class='dim'>/</span> <span class='fail'>"+x.failure+"</span></td>"
        +"<td>"+(rate===null?"<span class='dim'>unrated</span>":"<span class='mono' style='font-size:11px;color:#9fe8d8'>"+rate+"%</span><div class='bar'><i style='width:"+rate+"%'></i></div>")+"</td>"
        +"<td class='dim mono' style='font-size:11px'>"+(x.last_attested?ago(x.last_attested):"–")+"</td></tr>"}).join("");

    /* Domain filter chips — top domains by route count */
    var domCounts={};
    r.routes.forEach(function(x){if(x.domain)domCounts[x.domain]=(domCounts[x.domain]||0)+1});
    var domTop=Object.keys(domCounts).sort(function(p,q){return domCounts[q]-domCounts[p]||p.localeCompare(q)}).slice(0,12);
    if(activeDomain&&domTop.indexOf(activeDomain)<0)activeDomain="";
    $("domainChips").innerHTML='<span class="fchip'+(activeDomain===""?" active":"")+'" data-d="">All<span class="ct">'+r.routes.length+'</span></span>'+
      domTop.map(function(d){return '<span class="fchip'+(d===activeDomain?" active":"")+'" data-d="'+esc(d)+'">'+esc(d)+'<span class="ct">'+domCounts[d]+'</span></span>'}).join("");
    applyDomainFilter();

    /* Demand map — top queried domains + zero-result queries from /activity */
    var qevents=a.events.filter(function(e){return e.type==="query"&&e.detail});
    var dcounts={};
    qevents.forEach(function(e){var d=e.detail.domain;if(d&&d!=="web-playground")dcounts[d]=(dcounts[d]||0)+1});
    var dtop=Object.keys(dcounts).map(function(k){return[k,dcounts[k]]}).sort(function(p,q){return q[1]-p[1]}).slice(0,8);
    var dmax=dtop.length?dtop[0][1]:1;
    $("dm-domains").innerHTML=dtop.length
      ?dtop.map(function(p){return "<div class='dm-row'><span class='chip'>"+esc(p[0])+"</span><div class='bar' style='flex:1;margin-top:0'><i style='width:"+Math.round(p[1]/dmax*100)+"%'></i></div><span class='mono dim'>"+p[1]+"</span></div>"}).join("")
      :"<div class='dim' style='padding:6px 0'>No domain-hinted queries in the recent window.</div>";
    var zeros=[],seenTask={};
    qevents.forEach(function(e){var t=e.detail.task;if(e.detail.results===0&&t&&!seenTask[t]){seenTask[t]=1;zeros.push(e)}});
    $("dm-zero").innerHTML=zeros.length
      ?zeros.slice(0,8).map(function(e){return "<div class='dm-row'><span style='flex:1'>\\u201C"+esc(e.detail.task)+"\\u201D</span><span class='mono dim' style='white-space:nowrap'>"+ago(e.t)+"</span></div>"}).join("")
      :"<div class='dim' style='padding:6px 0'>Every recent query found a route.</div>";
  }).catch(function(){$("health").textContent="fetch error";$("dotEl")&&($("dotEl").style.background="#fbbf24")});
}
load();setInterval(load,30000);

/* ── COPY BUTTON ── */
$("copyBtn").onclick=function(){
  navigator.clipboard.writeText(this.textContent);
  this.textContent="copied ✓";
  var b=this;setTimeout(function(){b.textContent="claude mcp add --transport http waymark https://mcp.waymark.network/mcp"},1500)
};

/* ── CARD HOVER TILT ── */
document.querySelectorAll(".card").forEach(function(card){
  card.addEventListener("mousemove",function(e){
    var r=card.getBoundingClientRect();
    var x=(e.clientX-r.left)/r.width-.5;
    var y=(e.clientY-r.top)/r.height-.5;
    card.style.transform="translateY(-4px) rotateX("+(-y*5)+"deg) rotateY("+(x*5)+"deg) scale(1.01)";
  });
  card.addEventListener("mouseleave",function(){
    card.style.transform="";
  });
});
</script>
</body>
</html>
`;
