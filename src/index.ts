/*
 * Copyright 2026 MC Software, LLC. Licensed under the Apache License, Version 2.0.
 * Waymark is a service of MC Software, LLC — est. 1974. https://waymark.network
 */
/**
 * Waymark MCP server — collective procedural-knowledge network for AI agents.
 *
 * Tools (4):
 *   waymark_query      — find routes (step sequences that worked) for a task,
 *                        ranked by semantic similarity + community trust
 *   waymark_contribute — submit a sanitized trace of what worked / failed (API key)
 *   waymark_attest     — report the outcome of following a route (trust consensus)
 *   waymark_register   — mint a free contributor key (no auth, one call) to contribute
 *
 * Provenance is a per-route axis (verified / sampled / unverified), surfaced
 * honestly in every response — routes are never blanket-labeled "verified".
 *
 * Observability: full activity log (every tool call recorded to KV, 30-day TTL) +
 * public endpoints: /stats, /routes, /activity, /dashboard, /demand, /contributors, /drift.
 *
 * Storage: Cloudflare KV (key = route id, plus a compact index scored in memory).
 * Retrieval (shipped, v0.5): primary ranker is cosine similarity over
 * bge-base-en-v1.5 embeddings (Vectorize) with a confidence cutoff
 * (VEC_MIN_SCORE — a wrong route is worse than none), keyword-index fallback.
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
  // keyed_* (item 21, optional/back-compat): the subset of success/failure that
  // arrived with a VALID contributor key — identity-attributed consensus, the
  // input future trust weighting will read. Absent ⇒ 0 (all-anonymous legacy).
  attestations: { success: number; failure: number; lastAt: string | null; keyed_success?: number; keyed_failure?: number };
  // Provenance of fact-checking. Lets a consuming agent weight a route:
  //  - "verified":   individually fact-checked against live docs (per-route)
  //  - "sampled":    shipped under file-level sampling (some siblings checked, this one rode the heuristic)
  //  - "unverified": community contribution, not yet checked
  // Optional for back-compat; absent ⇒ treat as "sampled" (legacy seed default).
  verification?: { status: "verified" | "sampled" | "unverified"; method?: string; at?: string | null };
}

type EventType = "query" | "contribute" | "attest" | "register";
interface ActivityEvent {
  t: string;               // ISO timestamp
  type: EventType;
  detail: Record<string, unknown>;
}

const EVENT_TTL_SECONDS = 60 * 60 * 24 * 30; // 30 days

/** Abuse guard: max same-outcome attestations per route per hour (network total ≈23 attests today, so 10/hr/route/outcome only trips bulk spam). */
const ATTEST_OUTCOME_HOURLY_CAP = 10;

/** Abuse guard (item 21): max same-outcome attestations per contributor KEY per
 * hour, ACROSS routes — bounds identity-attributed bulk spam the per-route cap
 * can't see (one key spraying 1 attest × N routes). Anonymous attests are
 * untouched (canary + existing agents) and governed only by the per-route cap. */
const ATTEST_KEY_HOURLY_CAP = 30;

/** Activity log. Key sorts chronologically (ISO prefix). Returns the put promise
 * (never rejects) — callers MUST await it or hand it to ctx.waitUntil; an
 * un-awaited call gets cancelled when the response returns (the 2026-06-12
 * telemetry outage: zero evt: writes landed for ~15h). */
function logEvent(env: Env, type: EventType, detail: Record<string, unknown>): Promise<void> {
  const t = new Date().toISOString();
  const key = `evt:${t}:${crypto.randomUUID().slice(0, 8)}`;
  // Per-day event counter (cnt:evt:YYYY-MM-DD, TTL 32d) so /stats can report
  // events_30d by summing ≤31 small keys instead of list({prefix:"evt:"}),
  // which silently caps at 1000 keys. Get→put increment races under
  // concurrency (KV is not atomic) — same accepted alpha trade-off as
  // counter:queries; counts are telemetry, not billing.
  const day = `cnt:evt:${t.slice(0, 10)}`;
  const bump = env.ROUTES.get(day).then((v) =>
    env.ROUTES.put(day, String(parseInt(v ?? "0", 10) + 1), { expirationTtl: 60 * 60 * 24 * 32 })
  );
  return Promise.all([
    env.ROUTES.put(key, JSON.stringify({ t, type, detail } satisfies ActivityEvent), {
      expirationTtl: EVENT_TTL_SECONDS,
    }),
    bump,
  ]).then(() => {}).catch(() => {});
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
 * Now the FALLBACK path: semantic matching shipped in v0.5 (Vectorize +
 * bge embeddings — see below); this keyword index serves only when the
 * vector store is empty/unavailable. */

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

/** Per-isolate memo of the corpus scale (total routes + distinct domains).
 * The /r/{id} CTA count was the single most expensive op on the largest
 * crawlable surface (6,440 server-rendered pages): each render read AND
 * JSON.parsed the full ~1.3 MB fat index just to print a cosmetic
 * "X,400+ more routes across X,X00+ domains" line that rounds DOWN to the
 * nearest 100 (so it only changes every ~100 routes, ~1 h at current growth).
 * Memoizing per-isolate with a short TTL lets a warm isolate serve many route
 * pages off one index read instead of one read per hit. Read-only and purely
 * cosmetic: worst case the count is ≤TTL stale and rounds down, so it can only
 * ever under-promise — never over-claim. No write path is touched, and the
 * caller still falls back to the durable "thousands more" phrasing on error. */
let scaleMemo: { routeCount: number; domainCount: number; at: number } | null = null;
const SCALE_MEMO_TTL_MS = 300_000; // 5 min — matches the route page's own max-age=300 freshness contract
async function getScaleCounts(env: Env): Promise<{ routeCount: number; domainCount: number }> {
  const now = Date.now();
  if (scaleMemo && now - scaleMemo.at < SCALE_MEMO_TTL_MS) return scaleMemo;
  const idx = await getIndex(env);
  scaleMemo = { routeCount: idx.length, domainCount: new Set(idx.map((e) => e.domain)).size, at: now };
  return scaleMemo;
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
    if (e.verification?.status === "unverified") s *= UNVERIFIED_SERVING_FACTOR; // BLOCKER #1: don't serve unverified as authoritative
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
/* 0.56→0.62 recalibrated 2026-07-08 (backlog item 41) from REST-audited marginal
 * scores (eval/vec-score-audit.mjs, 7,612-route corpus): every golden expected
 * route scores ≥0.7661, weakest legit top result 0.7165, while the last grounded
 * keyword-salad survivor ("…nonsense tokens here") scored 0.6011. 0.62 excludes
 * it with zero measured loss in any legit top-5 (audited at 0.60/0.61/0.62). */
const VEC_MIN_SCORE = 0.62; // calibrated: exact ~0.94, intended paraphrase ≥~0.72, garbage ≤0.61

/* Vector-path grounding gate (refusal hardening, 2026-06-19).
 * As the corpus grew (3.8k→6.4k routes), the bge ranker began returning
 * semantically-incoherent matches *just over* VEC_MIN_SCORE for keyword-salad
 * queries that happen to contain real technical-adjacent words — e.g.
 * "asdf qwerty zxcv random nonsense tokens here" returns vLLM / SCTE-35 /
 * OpenVEX / OPA / e-invoicing routes that share NO content word with the query.
 * (This is now the real false-serve path; the keyword fallback already refuses
 * it — SPECS-deploy-gated's rankIndex fix is obsolete for this case.) Since "a
 * wrong route is worse than no route", below a high-confidence score we require
 * a served route to share ≥1 content token (>3 chars) with the query, or exactly
 * match the domain hint. Verified offline against the live 6,440-route corpus:
 * all 20 golden + 18 refusal retrieval pairs are grounded (0 at risk); the
 * false-serve garbage query shares 0 content tokens with every one of its
 * matches, so it is fully gated. Near-exact matches (≥VEC_HIGH_CONF) bypass the
 * check so genuine pure-semantic hits still serve. */
/* 0.66→0.70 recalibrated 2026-07-08 (item 41): an UNGROUNDED garbage match scored
 * 0.6685 and slipped through this bypass. Audit shows the only ungrounded matches
 * in the 0.66–0.70 band across all 38 legit eval queries are related-but-wrong
 * noise (e.g. Criteo auth for a Gmail query); every intended route scores ≥0.7165
 * or is lexically grounded, so raising the bypass costs nothing measured. */
const VEC_HIGH_CONF = 0.70; // near-exact only; paraphrases rely on grounding (they have it)

/* Coherence gate (item 59, 2026-07-09). The item-40 grounding gate can't catch
 * queries whose every match GENUINELY shares a token — tech-adjacent salad
 * ("webhook oauth kubernetes banana pipeline toaster deploy"), single-real-word
 * noise ("kafka blorptastic …"), and cross-domain mashups ("refund a github pull
 * request using twilio row level security") all served 3–5 routes. Evidence
 * (eval/vec-score-audit-20260709-0541.json, 8,602-route corpus): no threshold
 * separates them — the worst salad tops at 0.7235 while the weakest legit top
 * (oauth-refresh) is 0.7165. What DOES separate them, measured on all 34 eval
 * cases (20 golden + 18 serve + 13 garbage + 3 legit compound — 0 errors offline):
 *   1. Every golden/legit query's top serve ≥ 0.7612; every leak ≤ 0.7235.
 *      So the gate only runs in the low-confidence band top < VEC_STRONG=0.745
 *      (margins: worst leak +0.021 inside, closest legit compound-3 at 0.7629
 *      +0.018 outside). Strong-topped queries are untouched.
 *   2. Salad/noise queries contain ≥2 content tokens that appear NOWHERE in the
 *      corpus (banana+toaster, synergy+capacitor, 4× blorptastic-class); legit
 *      queries measured 0–1 (compound-1's "merges" — hence the naive stem
 *      fallback, which maps it to "merge"). NONSENSE gate: ≥2 unknown → refuse.
 *   3. Mashups ground DIFFERENT candidates on DIFFERENT fragments: their best
 *      anchor token covers ≤0.462 of served candidates (mashup-1 github 6/13,
 *      mashup-2 kafka 5/12), while in-band legit queries concentrate ≥0.50
 *      (postgres-notnull 0.500, compound-1 0.667, oauth-refresh 0.929).
 *      ANCHOR gate: best-anchor share < 0.48 → refuse. Margins are thin
 *      (±0.02) by nature of the measured gap — the eval harness guards drift;
 *      re-run vec-score-audit.mjs + refusal-eval.mjs after major corpus growth.
 * Refusal here is HARD (no keyword fallback): vector candidates existed but the
 * query itself is incoherent, and the keyword path would re-serve mashup
 * fragments. A wrong route is worse than no route. */
const VEC_STRONG = 0.745;
const ANCHOR_MIN_SHARE = 0.48;
const NONSENSE_MIN = 2;
const COHERENCE_STOP = new Set([
  "using", "with", "from", "into", "when", "that", "this", "without",
  "your", "their", "have", "what", "does", "should",
]);
/** Returns a refusal reason if an in-band query is incoherent, else null. */
function coherenceRefusal(query: string, domainHint: string | undefined, kept: Route[], idx: IdxEntry[]): string | null {
  const qTok = [...contentTokens(query + " " + (domainHint ?? ""))].filter((t) => !COHERENCE_STOP.has(t));
  if (qTok.length === 0) return null;
  // Nonsense gate: ≥2 query tokens unknown to the entire corpus (stem-tolerant).
  const known = new Set<string>();
  for (const e of idx) for (const t of contentTokens(e.task + " " + e.domain)) known.add(t);
  const stem = (t: string) => t.replace(/(ing|ed|es|s)$/, "");
  const nonsense = qTok.filter((t) => !known.has(t) && !known.has(stem(t)));
  if (nonsense.length >= NONSENSE_MIN) return "nonsense-tokens:" + nonsense.join(",");
  // Anchor-dispersion gate: no single query token grounds ≥48% of candidates.
  const cnt = new Map<string, number>();
  for (const r of kept) {
    const rt = contentTokens(r.task + " " + r.domain);
    for (const t of qTok) if (rt.has(t)) cnt.set(t, (cnt.get(t) ?? 0) + 1);
  }
  let max = 0;
  for (const v of cnt.values()) if (v > max) max = v;
  if (max / kept.length < ANCHOR_MIN_SHARE) return "anchor-dispersion:" + (max / kept.length).toFixed(3);
  return null;
}

const contentTokens = (s: string) =>
  new Set(s.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 3));
function lexicallyGrounded(query: string, domainHint: string | undefined) {
  const q = contentTokens(query + " " + (domainHint ?? ""));
  const dh = domainHint?.toLowerCase();
  return (r: Route): boolean => {
    if (dh && r.domain.toLowerCase() === dh) return true;
    const rt = contentTokens(r.task + " " + r.domain);
    for (const t of q) if (rt.has(t)) return true;
    return false;
  };
}

/* Trust gate for self-serve contributions (v0.6 security review BLOCKER #1):
 * a community route is verification:"unverified" until the canary re-checks it
 * against live docs. Unverified routes MUST NOT be served as authoritative, or
 * one contributor key could rank a poisoned route #1 for a task. We don't hide
 * them (that would starve the flywheel — they'd never get attested), but we
 * deprioritize them hard so any verified/operator route wins, and the agent
 * always sees verification.status in the response. */
const UNVERIFIED_SERVING_FACTOR = 0.12;
const servingScore = (r: Route, base: number) =>
  base * (r.verification?.status === "unverified" ? UNVERIFIED_SERVING_FACTOR : 1);

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

/* Input-size hardening (2026-07-06). Query text is unauthenticated public
 * input that flows into a billed Workers AI embedding call per non-cached
 * query. bge-base-en-v1.5 truncates at 512 tokens anyway, so capping at
 * QUERY_MAX chars loses nothing semantically while bounding cost, KV event
 * bloat, and keyword-path CPU. Applied inside retrieve() so every caller
 * (waymark_query, /search, related-routes) is covered. We slice rather than
 * reject: an agent pasting a long task description should still get results. */
const QUERY_MAX = 2000;
const DOMAIN_HINT_MAX = 200;

/** Hybrid retrieve: vector-first with cutoff, keyword fallback. */
async function retrieve(env: Env, query: string, domainHint: string | undefined, limit: number): Promise<Route[]> {
  query = query.slice(0, QUERY_MAX);
  if (domainHint) domainHint = domainHint.slice(0, DOMAIN_HINT_MAX);
  try {
    const qv = await embedQuery(env, query + (domainHint ? " — " + domainHint : ""));
    const res = await env.VEC.query(qv, { topK: Math.max(limit * 3, 12), returnMetadata: "none" });
    const cand = res.matches.filter((m) => m.score >= VEC_MIN_SCORE);
    if (cand.length > 0) {
      const routes = await fetchRoutes(env, cand.map((m) => m.id));
      if (routes.length > 0) {
        const score = new Map(cand.map((m) => [m.id, m.score]));
        // Refusal gate: drop incoherent low-confidence matches that share no
        // content word with the query (keyword-salad false serves). High-conf
        // matches bypass so genuine pure-semantic hits still serve.
        const grounded = lexicallyGrounded(query, domainHint);
        const kept = routes.filter((r) => (score.get(r.id) ?? 0) >= VEC_HIGH_CONF || grounded(r));
        if (kept.length > 0) {
          // Coherence gate (item 59): in the low-confidence band, refuse
          // incoherent queries HARD — see coherenceRefusal doc for evidence.
          const topScore = Math.max(...kept.map((r) => score.get(r.id) ?? 0));
          if (topScore < VEC_STRONG && coherenceRefusal(query, domainHint, kept, await getIndex(env)) !== null) {
            return [];
          }
          // BLOCKER #1: re-rank by verification-adjusted score so a verified route
          // always outranks an unverified one at similar semantic distance.
          kept.sort((a, b) => servingScore(b, score.get(b.id) ?? 0) - servingScore(a, score.get(a.id) ?? 0));
          return kept.slice(0, limit);
        }
        // else: everything was ungrounded low-confidence → fall through to the
        // keyword path (also gated), which refuses true garbage.
      }
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

/** WRITE_KEY-gated verification-provenance backfill. Stamps a `verification`
 * field onto routes that lack one. Two targets, both idempotent (only touch
 * routes missing the field):
 *   ?target=routes&cursor=&limit=200  — cursored batch over route:<id> KV values
 *                                        (source of truth; ≤1 list + N get + N put
 *                                        per call keeps subrequests well under the
 *                                        per-request limit — same discipline that
 *                                        makes full client-side rebuilds unsafe).
 *   ?target=index                     — one-shot map over idx:routes (the served
 *                                        copy) so /routes + ranking see the field.
 * ?status= (default "sampled") & ?method= let the caller stamp legacy routes as
 * "sampled" while the factory writes "verified"/"unverified" on new ones. */
async function migrateVerification(env: Env, key: string | null, request: Request): Promise<Response> {
  if (key !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
  const url = new URL(request.url);
  const target = url.searchParams.get("target") ?? "routes";
  const status = (url.searchParams.get("status") ?? "sampled") as "verified" | "sampled" | "unverified";
  const method = url.searchParams.get("method") ?? "legacy-file-sample";
  const stamp = { status, method, at: new Date().toISOString() };

  if (target === "index") {
    const idx = await getIndex(env);
    let stamped = 0;
    for (const e of idx) { if (!e.verification) { e.verification = stamp; stamped++; } }
    await env.ROUTES.put(INDEX_KEY, JSON.stringify(idx));
    return Response.json({ target, total: idx.length, stamped });
  }

  const limit = Math.min(Math.max(parseInt(url.searchParams.get("limit") ?? "200", 10) || 200, 1), 400);
  const cursor = url.searchParams.get("cursor") || undefined;
  const page = await env.ROUTES.list({ prefix: "route:", limit, cursor });
  let stamped = 0, processed = 0;
  // Parallelize KV gets/puts in chunks (latency-bound, not CPU-bound) so a
  // 300–400 key batch finishes in seconds rather than tens of seconds.
  const names = page.keys.map((k) => k.name);
  const toPut: [string, string][] = [];
  for (let i = 0; i < names.length; i += 50) {
    const slice = names.slice(i, i + 50);
    const raws = await Promise.all(slice.map((n) => env.ROUTES.get(n)));
    for (let j = 0; j < slice.length; j++) {
      const raw = raws[j];
      if (!raw) continue;
      processed++;
      const r: Route = JSON.parse(raw);
      if (!r.verification) {
        r.verification = stamp;
        toPut.push([slice[j], JSON.stringify(r)]);
        stamped++;
      }
    }
  }
  for (let i = 0; i < toPut.length; i += 50) {
    await Promise.all(toPut.slice(i, i + 50).map(([n, v]) => env.ROUTES.put(n, v)));
  }
  return Response.json({
    target, processed, stamped,
    cursor: page.list_complete ? null : page.cursor,
    done: page.list_complete,
  });
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
          "multi-step web/tool procedure), query Waymark for routes other agents have " +
          "documented and attested. Returns step sequences, known gotchas, and live success statistics. " +
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
            verification: r.verification ?? { status: "sampled", method: "legacy-file-sample" },
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
        // Field-size caps (2026-07-06): contributions land in route:{id} AND the
        // shared fat index (idx:routes, type IdxEntry = Route) that every read
        // surface parses — unbounded fields let one free self-serve key (20/h)
        // balloon the index toward KV's 25 MB value limit, degrading every
        // getIndex() consumer and eventually breaking patchIndex corpus-wide.
        // Caps calibrated against the live 6,440-route corpus (2026-07-06 maxima:
        // task 247, domain 135, steps 6×350, gotchas 3×402, ~2.9 KB/route) with
        // 2–10× headroom — no existing or plausibly legitimate route is rejected.
        // Worst-case accepted route is now ~50 KB vs unbounded.
        inputSchema: {
          api_key: z.string().max(200).describe("Contributor API key from waymark.network"),
          task: z.string().max(500).describe("What the route accomplishes, stated generally"),
          domain: z.string().max(200).describe("Service/site/API the route applies to"),
          steps: z.array(z.string().max(1200)).min(1).max(25).describe("Ordered procedural steps (sanitized)"),
          gotchas: z.array(z.string().max(1200)).max(15).default([]).describe("Failure modes, rate limits, quirks encountered"),
          contributor: z.string().max(120).describe("Your agent/org handle"),
        },
        annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false },
      },
      async ({ api_key, task, domain, steps, gotchas, contributor }) => {
        // Auth (v0.6): admin/factory WRITE_KEY OR a self-serve contributor key.
        // Self-serve keys are what let the network grow from external agents
        // rather than only the operator — see the contributor-key block below.
        const isAdmin = api_key === env.WRITE_KEY;
        let keyRec: { rec: ContribKey; storeKey: string } | null = null;
        if (!isAdmin) {
          keyRec = await lookupContribKey(env, api_key);
          if (!keyRec || keyRec.rec.revoked) {
            await logEvent(env, "contribute", { rejected: keyRec ? "revoked_key" : "bad_key", domain });
            return { content: [{ type: "text" as const, text: "Invalid or revoked API key. Get a free contributor key: call waymark_register, or POST https://mcp.waymark.network/v1/keys {\"handle\":\"your-agent\"}." }], isError: true };
          }
          // Per-key hourly contribution cap — bounds corpus poisoning by any one key.
          // Tier-aware (M1): pro keys get 10× the free cap.
          const kb = new Date().toISOString().slice(0, 13); // YYYY-MM-DDTHH
          const kRl = `ckcrl:${keyRec.storeKey.slice(3)}:${kb}`;
          const kUsed = parseInt((await env.ROUTES.get(kRl)) ?? "0", 10);
          const kCap = contribCapFor(keyRec.rec);
          if (kUsed >= kCap) {
            await logEvent(env, "contribute", { rejected: "rate_capped", domain });
            return { content: [{ type: "text" as const, text: JSON.stringify({ status: "rate_capped", detail: `Per-key cap of ${kCap} contributions/hour reached. Retry later.` }) }], isError: true };
          }
          await env.ROUTES.put(kRl, String(kUsed + 1), { expirationTtl: 7200 });
        }
        if (looksSensitive([task, domain, ...steps, ...gotchas].join(" "))) {
          await logEvent(env, "contribute", { rejected: "sensitive_content", domain });
          return { content: [{ type: "text" as const, text: "Rejected: submission appears to contain credentials/secrets. Sanitize and resubmit procedure-only content." }], isError: true };
        }
        // Item 24 v1: malicious-content screen — applies to community AND
        // operator/factory contributions (see looksMalicious doc-comment).
        const malicious = looksMalicious([task, domain, ...steps, ...gotchas].join("\n"));
        if (malicious) {
          await logEvent(env, "contribute", { rejected: "malicious_content", rule: malicious, domain });
          return { content: [{ type: "text" as const, text: `Rejected (${malicious}): submission matched Waymark's write-time content screen (exfiltration sinks, public-IP URLs, insecure pipe-to-shell, or instructions targeting the consuming agent). If this is a false positive, rephrase the flagged step or contact security@waymark.network.` }], isError: true };
        }
        // Attribution: for self-serve keys, trust the registered handle, not a
        // free-text field the caller could spoof.
        const handle = isAdmin ? contributor : (keyRec!.rec.handle || contributor);
        const src = isAdmin ? "operator" : "community";
        const id = crypto.randomUUID();
        const nowIso = new Date().toISOString();
        const route: Route = {
          id, task, domain, steps, gotchas, contributor: handle,
          created: nowIso,
          attestations: { success: 0, failure: 0, lastAt: null },
          // Neither path claims "verified" — that status is earned only when the
          // canary re-checks the route against live docs. Method records origin.
          verification: { status: "unverified", method: isAdmin ? "operator-contrib" : "community-contrib", at: nowIso },
        };
        await env.ROUTES.put(`route:${id}`, JSON.stringify(route));
        await patchIndex(env, (idx) => [...idx, route]);
        await upsertVector(env, route);
        if (keyRec) {
          keyRec.rec.contributions++;
          keyRec.rec.lastAt = nowIso;
          await env.ROUTES.put(keyRec.storeKey, JSON.stringify(keyRec.rec));
        }
        await logEvent(env, "contribute", {
          route_id: id,
          task: task.slice(0, 140),
          domain,
          contributor: handle,
          steps: steps.length,
          src,
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
          "Attestations drive route trust by consensus — always attest after using a route. " +
          "Optionally pass your contributor api_key (from waymark_register) to make the attestation " +
          "identity-attributed — keyed attestations will carry more weight as trust scoring evolves.",
        inputSchema: {
          route_id: z.string().max(64), // uuids are 36 chars
          outcome: z.enum(["success", "failure"]),
          note: z.string().max(500).optional().describe("Optional short note on what diverged"), // only ever rendered sliced to 140
          api_key: z.string().max(200).optional().describe("Optional contributor API key (waymark_register). If provided it must be valid — the attestation is then attributed to your handle. Omit to attest anonymously."),
        },
        annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false },
      },
      async ({ route_id, outcome, note, api_key }) => {
        const raw = await env.ROUTES.get(`route:${route_id}`);
        if (!raw) return { content: [{ type: "text" as const, text: "Unknown route_id" }], isError: true };
        const route: Route = JSON.parse(raw);
        const bucket = new Date().toISOString().slice(0, 13); // YYYY-MM-DDTHH
        // Identity (item 21 groundwork): attesting stays OPEN — anonymous attests
        // work exactly as before (canary + existing agents unbroken) — but a caller
        // may pass its contributor key to make the attestation identity-attributed.
        // Invalid/revoked keys are rejected loudly, NOT silently downgraded to
        // anonymous — a spoofed identity must never pass as keyed, and typos surface.
        // Deliberately does NOT accept WRITE_KEY (no new WRITE_KEY surface).
        let attester: string | null = null;
        if (api_key !== undefined) {
          const k = await lookupContribKey(env, api_key);
          if (!k || k.rec.revoked) {
            await logEvent(env, "attest", { route_id, outcome, rejected: k ? "revoked_key" : "bad_key" });
            return { content: [{ type: "text" as const, text: "Invalid or revoked API key. Attest without api_key, or get a free contributor key: call waymark_register, or POST https://mcp.waymark.network/v1/keys {\"handle\":\"your-agent\"}." }], isError: true };
          }
          // Per-key same-outcome hourly cap (across ALL routes) — bounds bulk
          // trust-inflation by one identity; the per-route cap below still applies.
          const akKey = `akrl:${k.storeKey.slice(3)}:${outcome}:${bucket}`;
          const akCount = parseInt((await env.ROUTES.get(akKey)) ?? "0", 10);
          if (akCount >= ATTEST_KEY_HOURLY_CAP) {
            await logEvent(env, "attest", { route_id, outcome, rejected: "key_rate_capped", attester: k.rec.handle });
            return {
              content: [{ type: "text" as const, text: JSON.stringify({ recorded: false, reason: "rate_capped", detail: `Per-key cap of ${ATTEST_KEY_HOURLY_CAP} '${outcome}' attestations per hour reached. Retry later.` }) }],
              isError: true,
            };
          }
          await env.ROUTES.put(akKey, String(akCount + 1), { expirationTtl: 7200 });
          attester = k.rec.handle;
        }
        // Abuse guard: cap same-outcome attestations per route per hour (soft cap —
        // KV is eventually consistent — but defeats bulk trust-inflation/poisoning spam).
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
        if (attester) {
          if (outcome === "success") route.attestations.keyed_success = (route.attestations.keyed_success ?? 0) + 1;
          else route.attestations.keyed_failure = (route.attestations.keyed_failure ?? 0) + 1;
        }
        route.attestations.lastAt = new Date().toISOString();
        await env.ROUTES.put(`route:${route_id}`, JSON.stringify(route));
        await patchIndex(env, (idx) => idx.map((e) => (e.id === route_id ? route : e)));
        await logEvent(env, "attest", {
          route_id,
          outcome,
          task: route.task.slice(0, 140),
          note: note ? note.slice(0, 140) : null,
          attester, // null = anonymous; set = validated contributor handle (item 21)
        });
        return { content: [{ type: "text" as const, text: JSON.stringify({ recorded: true, route_id, attestations: route.attestations, attested_as: attester ?? "anonymous" }) }] };
      }
    );

    this.server.registerTool(
      "waymark_register",
      {
        title: "Register for a contributor key",
        description:
          "Get a free contributor API key so you can submit routes with waymark_contribute. " +
          "The key is returned once — store it. Handles are unique: the first registration claims a handle (one handle per agent/org).",
        inputSchema: {
          handle: z.string().min(2).max(60).describe("Your agent/org handle, e.g. 'acme-sales-agent'"),
        },
        annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false },
      },
      async ({ handle }) => {
        const h = handle.trim();
        if (!/^[a-zA-Z0-9 _.\-\/@]{2,60}$/.test(h)) {
          return { content: [{ type: "text" as const, text: "Invalid handle. Use 2–60 chars: letters, digits, space, _ . - / @" }], isError: true };
        }
        if (RESERVED_HANDLE.test(h)) {
          return { content: [{ type: "text" as const, text: "That handle is reserved. Choose a handle that doesn't impersonate Waymark/operator/official identities." }], isError: true };
        }
        const res = await createContribKey(env, h);
        if (!res.ok) {
          const detail = res.reason === "handle_taken"
            ? "That handle is already registered — handles are unique (first come). Choose a different handle; keys are shown once and cannot be recovered self-serve."
            : "Key issuance temporarily rate-limited network-wide. Retry shortly.";
          return { content: [{ type: "text" as const, text: JSON.stringify({ status: res.reason, detail }) }], isError: true };
        }
        await logEvent(env, "register", { handle: h });
        return { content: [{ type: "text" as const, text: JSON.stringify({ api_key: res.key, handle: h, note: "Store this key now — it is shown only once. Pass it as api_key in waymark_contribute." }) }] };
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

/* ---- Write-time malicious-content screen (item 24, v1) ----
 * Routes are text CONSUMED AND EXECUTED by other agents, so a poisoned route
 * is both an exfiltration vector (steps directing output/credentials to an
 * attacker sink) and a prompt-injection vector (steps instructing the
 * consuming agent itself). Screen every contribution — community AND
 * operator/factory (the factory ingests scraped web docs, which can carry
 * injection) — and reject loudly with the rule name; rejections are logged
 * publicly as contribute events (rejected:"malicious_content").
 *
 * Every rule was calibrated 2026-07-08 against the FULL live corpus
 * (7,925 routes, full step/gotcha text): 0 false positives. Deliberately
 * NOT included, with evidence:
 *  - https pipe-to-shell from named domains (3 legit official installers in
 *    corpus: syft, grype, linkerd) — only http:// or IP-literal → shell flags;
 *  - ngrok domains — canonical legit webhook-dev tooling;
 *  - bare "exfiltrate" (2 legit security-docs routes) and "send the token to"
 *    (legit OAuth/Play-Integrity phrasing) — the credential rule requires the
 *    possessive form ("send YOUR api key to").
 * Deferred to a later run (item 24 full scope): npm/pip advisory lookups,
 * malicious-domain reputation feeds. Returns the rule name, or null if clean. */
const EXFIL_SINKS = /\b(webhook\.site|requestbin\.(com|net)|m\.pipedream\.net|burpcollaborator\.net|oastify\.com|oast\.(pro|fun|live|site|online|me)|interact\.sh|interactsh\.com|canarytokens?\.(com|org)|beeceptor\.com|hookbin\.com|postb\.in|requestcatcher\.com|requestinspector\.com|ptsv2\.com|webhook-test\.com)\b/i;
const IP_URL = /https?:\/\/(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}/gi;
const PIPE_SH = /\b(curl|wget)\b[^\n|;]{0,300}\|\s*(sudo\s+)?(ba|z|da)?sh\b/i;
function hasPublicIpUrl(s: string): boolean {
  for (const m of s.matchAll(IP_URL)) {
    const a = +m[1], b = +m[2];
    // Loopback/private/link-local are legit in docs (OAuth loopback redirect
    // URIs, local dev examples) — only a PUBLIC IP-literal URL is flagged.
    if (a === 127 || a === 10 || a === 0 || (a === 192 && b === 168) || (a === 172 && b >= 16 && b <= 31) || (a === 169 && b === 254)) continue;
    return true;
  }
  return false;
}
function looksMalicious(s: string): string | null {
  if (EXFIL_SINKS.test(s)) return "exfil_sink_domain";
  if (hasPublicIpUrl(s)) return "public_ip_url";
  const pipe = s.match(PIPE_SH);
  if (pipe && (/http:\/\//i.test(pipe[0]) || hasPublicIpUrl(pipe[0]))) return "insecure_pipe_to_shell";
  if (/base64\s+(-d|--decode)[^\n|;]{0,100}\|\s*(sudo\s+)?(ba|z)?sh\b/i.test(s)) return "b64_to_shell";
  if (/\b(ignore|disregard|forget)\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|messages?|prompts?|rules?)\b/i.test(s)) return "prompt_injection";
  if (/\b(send|post|forward|upload|transmit|exfiltrate)\s+(your|all\s+your)\s+(api[\s_-]?keys?|credentials?|secrets?|passwords?|tokens?)\s+to\b/i.test(s)) return "credential_exfil_instruction";
  if (/\bdo\s+not\s+(tell|inform|alert|notify)\s+the\s+(user|human|operator)\b/i.test(s)) return "concealment_instruction";
  return null;
}

function serverCard(env: Env) {
  return {
    name: "network.waymark/server",
    title: "Waymark",
    description: "Collective procedural-knowledge network for AI agents. Query task routes other agents documented and attested; contribute and attest outcomes.",
    version: env.SERVER_VERSION,
    remotes: [{ transport_type: "streamable-http", url: `${env.PUBLIC_URL}/mcp` }],
    websiteUrl: "https://waymark.network",
  };
}

const CORS = { "Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "ETag, Link" };

/* ---- RFC 8631 service-description discovery (item 45) ----
 * Every JSON endpoint documented in /openapi.json advertises that document
 * via a `Link: rel="service-desc"` response header, so a client that lands
 * on ANY API response can discover the machine-readable service description
 * without prior knowledge of the /openapi.json convention (the header-level
 * twin of item 43's <link rel=alternate> on HTML pages). Applied at the
 * dispatch site via withServiceDesc(); the response is re-wrapped because
 * handler responses may carry immutable headers. Safe on 304s (null body). */
const SERVICE_DESC_LINK = '<https://mcp.waymark.network/openapi.json>; rel="service-desc"';
async function withServiceDesc(res: Response | Promise<Response>): Promise<Response> {
  const r = await res;
  const out = new Response(r.body, r);
  out.headers.append("Link", SERVICE_DESC_LINK);
  return out;
}

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

/* ---------------- Self-serve contributor keys (v0.6) ----------------
 * Until v0.6, waymark_contribute was gated on the single shared WRITE_KEY, so
 * only the operator/factory could add routes — the network could not grow from
 * external agents, which is the actual flywheel. v0.6 issues per-agent keys:
 *   - Raw key `wmk_<48 hex>` is shown ONCE; only its SHA-256 is stored
 *     (KV key ck:<hash>), so a KV dump never yields a usable key.
 *   - Per-key hourly contribution cap bounds corpus poisoning by any one key.
 *   - Per-IP + global hourly issuance caps slow mass key minting.
 *   - Community contributions stay verification:"unverified" (no trust/serving
 *     priority) until the canary re-verifies — same trust model as before.
 *   - WRITE_KEY remains the admin/factory path, unchanged and separate. */

const CONTRIB_KEY_HOURLY_CAP = 20;        // sanitized routes per key per hour (free tier)
// v0.7 monetization (M1): Pro tier ($19/mo, waymark.network/pricing) = 10× the
// free per-key hourly contribute cap + priority placement in the factory's
// verification queue (/admin/priority-queue). Key records without `tier` are
// free (all pre-M1 keys). Upgrades are operator-only (POST /admin/upgrade-key,
// WRITE_KEY-gated) after Stripe payment — deliberately no payment logic in-worker.
const PRO_CAP_MULTIPLIER = 10;
const KEY_ISSUE_IP_HOURLY_CAP = 5;        // new keys per IP per hour (HTTP path)
// Network-wide new-key ceiling/hour. Bounds the MCP register path (which has no
// per-IP attribution) — worst case now ~50 keys × 20 = 1000 *unverified* writes/hr,
// all heavily deprioritized in serving (UNVERIFIED_SERVING_FACTOR) and prunable.
// NOTE (review #3): KV counters are non-atomic, so caps are soft under burst
// concurrency; DO/D1-backed atomic counters are the planned hardening.
const KEY_ISSUE_GLOBAL_HOURLY_CAP = 50;

/** Reserved handles an external key may not claim (anti-impersonation, review #8). */
const RESERVED_HANDLE = /(^|[^a-z])(waymark|operator|official|admin|staff|mc[\s_-]?software)([^a-z]|$)|seed/i;

/** `tier` absent ⇒ "free" (back-compat: all pre-M1 records). */
interface ContribKey { handle: string; created: string; revoked: boolean; contributions: number; lastAt: string | null; tier?: "free" | "pro" }

/** Effective per-key hourly contribute cap for a key record (M1 entitlement). */
function contribCapFor(rec: ContribKey): number {
  return rec.tier === "pro" ? CONTRIB_KEY_HOURLY_CAP * PRO_CAP_MULTIPLIER : CONTRIB_KEY_HOURLY_CAP;
}

async function sha256hex(s: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Resolve a raw contributor key to its stored record (or null). */
async function lookupContribKey(env: Env, rawKey: string): Promise<{ rec: ContribKey; storeKey: string } | null> {
  if (typeof rawKey !== "string" || !/^wmk_[0-9a-f]{48}$/.test(rawKey)) return null;
  const storeKey = `ck:${await sha256hex(rawKey)}`;
  const raw = await env.ROUTES.get(storeKey);
  if (!raw) return null;
  return { rec: JSON.parse(raw) as ContribKey, storeKey };
}

/** Handle uniqueness (anti-impersonation): attribution for contributions and
 * keyed attestations is handle-based, so a second key minted for an existing
 * handle would silently merge into that contributor's leaderboard stats and
 * provenance. First registration claims the normalized handle via a
 * `hnd:{normalized}` marker; later attempts are rejected (handle_taken).
 * Prospective only — keys that existed before this shipped stay valid.
 * NOTE: KV is non-atomic, so a concurrent same-handle race is soft (same
 * accepted class as the rate-limit counters, review #3). */
function normalizeHandle(h: string): string {
  return h.toLowerCase().replace(/\s+/g, " ").trim();
}

/** Mint a key (shared by the MCP tool and the HTTP endpoint). Global cap only;
 * the HTTP endpoint adds a per-IP cap on top. */
async function createContribKey(env: Env, handle: string): Promise<{ ok: true; key: string } | { ok: false; reason: string }> {
  const hndKey = `hnd:${normalizeHandle(handle)}`;
  if (await env.ROUTES.get(hndKey)) return { ok: false, reason: "handle_taken" }; // checked before the cap so probes don't burn issuance budget
  const bucket = new Date().toISOString().slice(0, 13);
  const gKey = `krl:global:${bucket}`;
  const used = parseInt((await env.ROUTES.get(gKey)) ?? "0", 10);
  if (used >= KEY_ISSUE_GLOBAL_HOURLY_CAP) return { ok: false, reason: "global_rate_capped" };
  await env.ROUTES.put(gKey, String(used + 1), { expirationTtl: 7200 });
  const rawKey = "wmk_" + [...crypto.getRandomValues(new Uint8Array(24))].map((b) => b.toString(16).padStart(2, "0")).join("");
  const rec: ContribKey = { handle, created: new Date().toISOString(), revoked: false, contributions: 0, lastAt: null };
  const keyHash = await sha256hex(rawKey);
  await env.ROUTES.put(`ck:${keyHash}`, JSON.stringify(rec));
  await env.ROUTES.put(hndKey, JSON.stringify({ handle, created: rec.created, key_hash: keyHash }));
  await env.ROUTES.get("counter:keys").then((v) => env.ROUTES.put("counter:keys", String(parseInt(v ?? "0", 10) + 1))).catch(() => {});
  return { ok: true, key: rawKey };
}

/** POST /v1/keys {handle} — public, IP-rate-limited self-serve key issuance. */
async function issueKeyHttp(env: Env, request: Request): Promise<Response> {
  let handle = "";
  try { handle = String((((await request.json()) as { handle?: unknown }) ?? {}).handle ?? "").trim(); } catch { /* 400 below */ }
  if (!/^[a-zA-Z0-9 _.\-\/@]{2,60}$/.test(handle)) {
    return Response.json({ error: "handle required: 2–60 chars (letters, digits, space, _ . - / @)" }, { status: 400, headers: CORS });
  }
  if (RESERVED_HANDLE.test(handle)) {
    return Response.json({ error: "handle reserved: cannot impersonate Waymark/operator/official identities" }, { status: 400, headers: CORS });
  }
  const ip = request.headers.get("cf-connecting-ip") ?? "0.0.0.0";
  const bucket = new Date().toISOString().slice(0, 13);
  const ipKey = `ckrl:${ip}:${bucket}`;
  const ipUsed = parseInt((await env.ROUTES.get(ipKey)) ?? "0", 10);
  if (ipUsed >= KEY_ISSUE_IP_HOURLY_CAP) {
    return Response.json({ error: `rate limited: max ${KEY_ISSUE_IP_HOURLY_CAP} keys/hour from one IP` }, { status: 429, headers: CORS });
  }
  await env.ROUTES.put(ipKey, String(ipUsed + 1), { expirationTtl: 7200 });
  const res = await createContribKey(env, handle);
  if (!res.ok) {
    if (res.reason === "handle_taken") {
      return Response.json({ error: "handle already registered — handles are unique (first come). Choose a different handle; keys are shown once and cannot be recovered self-serve." }, { status: 409, headers: CORS });
    }
    return Response.json({ error: "network-wide key issuance rate limit reached; retry shortly" }, { status: 429, headers: CORS });
  }
  await logEvent(env, "register", { handle });
  return Response.json({ api_key: res.key, handle, note: "Store this key now — it is shown only once. Use it as api_key in waymark_contribute." }, { headers: CORS });
}

/** POST /admin/revoke-key {key|key_hash} — WRITE_KEY-gated. */
async function revokeKey(env: Env, adminKey: string | null, request: Request): Promise<Response> {
  if (adminKey !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
  let body: { key?: string; key_hash?: string };
  try { body = await request.json(); } catch { return Response.json({ error: "bad json" }, { status: 400 }); }
  let storeKey: string | null = null;
  if (body.key) { const r = await lookupContribKey(env, body.key); storeKey = r?.storeKey ?? null; }
  else if (body.key_hash && /^[0-9a-f]{64}$/.test(body.key_hash)) storeKey = `ck:${body.key_hash}`;
  if (!storeKey) return Response.json({ error: "provide a valid key or key_hash" }, { status: 400 });
  const raw = await env.ROUTES.get(storeKey);
  if (!raw) return Response.json({ revoked: false, reason: "not found" }, { status: 404 });
  const rec = JSON.parse(raw) as ContribKey;
  rec.revoked = true;
  await env.ROUTES.put(storeKey, JSON.stringify(rec));
  return Response.json({ revoked: true, handle: rec.handle });
}

/** POST /admin/upgrade-key {key|key_hash, tier:"free"|"pro"} — WRITE_KEY-gated (M1).
 * The operator flips a key's tier after Stripe payment for the $19/mo Pro SKU
 * (waymark.network/pricing); "free" downgrades on churn. No payment logic in-worker. */
async function upgradeKey(env: Env, adminKey: string | null, request: Request): Promise<Response> {
  if (adminKey !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
  let body: { key?: string; key_hash?: string; tier?: string };
  try { body = await request.json(); } catch { return Response.json({ error: "bad json" }, { status: 400 }); }
  const tier = body.tier;
  if (tier !== "free" && tier !== "pro") return Response.json({ error: 'tier required: "free" | "pro"' }, { status: 400 });
  let storeKey: string | null = null;
  if (body.key) { const r = await lookupContribKey(env, body.key); storeKey = r?.storeKey ?? null; }
  else if (body.key_hash && /^[0-9a-f]{64}$/.test(body.key_hash)) storeKey = `ck:${body.key_hash}`;
  if (!storeKey) return Response.json({ error: "provide a valid key or key_hash" }, { status: 400 });
  const raw = await env.ROUTES.get(storeKey);
  if (!raw) return Response.json({ upgraded: false, reason: "not found" }, { status: 404 });
  const rec = JSON.parse(raw) as ContribKey;
  rec.tier = tier;
  await env.ROUTES.put(storeKey, JSON.stringify(rec));
  return Response.json({ upgraded: true, handle: rec.handle, tier, hourly_contribute_cap: contribCapFor(rec) });
}

/** GET /admin/priority-queue — WRITE_KEY-gated (M1). Unverified routes from
 * active pro-tier contributors, newest first — the factory verifies these ahead
 * of the general pool (the paid entitlement). Same consumption pattern as
 * /admin/route-requests. Full ck: scan + fat-index read: fine at current key
 * counts for an operator-only endpoint; revisit if keys grow past ~1000. */
async function priorityQueueList(env: Env, adminKey: string | null): Promise<Response> {
  if (adminKey !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
  const listed = await env.ROUTES.list({ prefix: "ck:", limit: 1000 });
  const proHandles = new Set<string>();
  for (const k of listed.keys) {
    const raw = await env.ROUTES.get(k.name);
    if (!raw) continue;
    try {
      const rec = JSON.parse(raw) as ContribKey;
      if (rec.tier === "pro" && !rec.revoked) proHandles.add(rec.handle);
    } catch { /* skip corrupt */ }
  }
  const idx = await getIndex(env);
  const queue = idx
    .filter((r) => proHandles.has(r.contributor) && (r.verification?.status ?? "sampled") === "unverified")
    .sort((a, b) => (a.created < b.created ? 1 : -1))
    .slice(0, 200)
    .map((r) => ({ id: r.id, task: r.task, domain: r.domain, contributor: r.contributor, created: r.created, priority: true }));
  return Response.json({ pro_contributors: proHandles.size, open: queue.length, queue });
}

/* ---------------- v0.7 — paid route requests (bounty intake) ----------------
 * POST /v1/route-requests: public form target for the $25 "Route Bounty" SKU on
 * waymark.network/pricing. Stores the request; the route factory picks open
 * requests up hourly (authors + 100%-verifies first), and the operator emails
 * the Stripe invoice. Deliberately no payment processing in the worker. */
const RR_IP_DAILY_CAP = 5;
interface RouteRequest {
  id: string; task: string; domain: string | null; contact: string; notes: string | null;
  status: "open" | "fulfilled" | "closed"; created: string; fulfilledAt: string | null; route_id: string | null;
}

async function routeRequestCreate(env: Env, request: Request): Promise<Response> {
  let b: { task?: unknown; domain?: unknown; contact?: unknown; notes?: unknown } = {};
  try { b = ((await request.json()) as typeof b) ?? {}; } catch { /* validated below */ }
  const task = String(b.task ?? "").trim();
  const contact = String(b.contact ?? "").trim();
  const domain = String(b.domain ?? "").trim().slice(0, 80) || null;
  const notes = String(b.notes ?? "").trim().slice(0, 500) || null;
  if (task.length < 10 || task.length > 300) {
    return Response.json({ error: "task required: 10–300 chars describing what the route should accomplish" }, { status: 400, headers: CORS });
  }
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(contact) || contact.length > 120) {
    return Response.json({ error: "contact required: a valid email (invoice + delivery link go there)" }, { status: 400, headers: CORS });
  }
  const ip = request.headers.get("cf-connecting-ip") ?? "0.0.0.0";
  const bucket = new Date().toISOString().slice(0, 10);
  const rlKey = `rrrl:${ip}:${bucket}`;
  const used = parseInt((await env.ROUTES.get(rlKey)) ?? "0", 10);
  if (used >= RR_IP_DAILY_CAP) {
    return Response.json({ error: `rate limited: max ${RR_IP_DAILY_CAP} route requests/day from one IP` }, { status: 429, headers: CORS });
  }
  await env.ROUTES.put(rlKey, String(used + 1), { expirationTtl: 172800 });
  const id = crypto.randomUUID();
  const rec: RouteRequest = { id, task, domain, contact, notes, status: "open", created: new Date().toISOString(), fulfilledAt: null, route_id: null };
  await env.ROUTES.put(`rr:${id}`, JSON.stringify(rec));
  await env.ROUTES.get("counter:route_requests").then((v) => env.ROUTES.put("counter:route_requests", String(parseInt(v ?? "0", 10) + 1))).catch(() => {});
  return Response.json({
    id, status: "open",
    next: "Request received. We'll confirm feasibility and email a $25 invoice within 24h; your individually verified route ships within 24h of payment.",
  }, { headers: CORS });
}

async function routeRequestsList(env: Env, adminKey: string | null): Promise<Response> {
  if (adminKey !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
  const listed = await env.ROUTES.list({ prefix: "rr:", limit: 1000 });
  const out: RouteRequest[] = [];
  for (const k of listed.keys.slice(0, 100)) {
    const raw = await env.ROUTES.get(k.name);
    if (raw) { try { out.push(JSON.parse(raw) as RouteRequest); } catch { /* skip corrupt */ } }
  }
  out.sort((a, b) => (a.created < b.created ? 1 : -1));
  return Response.json({ total: out.length, open: out.filter((r) => r.status === "open").length, requests: out });
}

async function routeRequestFulfill(env: Env, adminKey: string | null, request: Request): Promise<Response> {
  if (adminKey !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
  let body: { id?: string; route_id?: string; status?: string };
  try { body = await request.json(); } catch { return Response.json({ error: "bad json" }, { status: 400 }); }
  const id = String(body.id ?? "");
  const raw = id ? await env.ROUTES.get(`rr:${id}`) : null;
  if (!raw) return Response.json({ error: "not found" }, { status: 404 });
  const rec = JSON.parse(raw) as RouteRequest;
  rec.status = body.status === "closed" ? "closed" : "fulfilled";
  rec.fulfilledAt = new Date().toISOString();
  rec.route_id = body.route_id ? String(body.route_id) : rec.route_id;
  await env.ROUTES.put(`rr:${id}`, JSON.stringify(rec));
  return Response.json({ ok: true, request: rec });
}

/* ---------------- M7 — drift-alert email capture ----------------
 * "Email me when an API I depend on changes" — the funnel on the drift tracker,
 * our best free-tool magnet. INTAKE ONLY: no mail infrastructure exists yet, so
 * nothing is ever sent; the form copy says so explicitly. Records are stored
 * double-opt-in-ready (confirmed:false until a future confirmation flow flips
 * it) and the subscriber COUNT feeds /demand as a demand signal. Emails are PII:
 * they appear only in the WRITE_KEY-gated /admin/drift-alerts list, never on a
 * public surface. */
const DA_IP_DAILY_CAP = 10;
const DA_MAX_DOMAINS = 20;

interface DriftAlert {
  email: string; domains: string[]; created: string; updated: string;
  confirmed: boolean; // double-opt-in-ready: stays false until a real confirmation flow exists
  source: string;
}

/** POST /v1/drift-alerts {email, domains?: string[] | comma-string} — public, IP-rate-limited. */
async function driftAlertCreate(env: Env, request: Request): Promise<Response> {
  let b: { email?: unknown; domains?: unknown } = {};
  try { b = ((await request.json()) as typeof b) ?? {}; } catch { /* validated below */ }
  const email = String(b.email ?? "").trim().toLowerCase();
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email) || email.length > 120) {
    return Response.json({ error: "email required: a valid address (max 120 chars)" }, { status: 400, headers: CORS });
  }
  // domains: accept an array or a comma-separated string; normalize, dedupe, cap.
  const rawDomains = Array.isArray(b.domains) ? b.domains.map(String) : String(b.domains ?? "").split(",");
  const domains = [...new Set(rawDomains.map((d) => d.trim().toLowerCase()).filter((d) => d.length > 0 && d.length <= 80))].slice(0, DA_MAX_DOMAINS);
  const ip = request.headers.get("cf-connecting-ip") ?? "0.0.0.0";
  const bucket = new Date().toISOString().slice(0, 10);
  const rlKey = `darl:${ip}:${bucket}`;
  const used = parseInt((await env.ROUTES.get(rlKey)) ?? "0", 10);
  if (used >= DA_IP_DAILY_CAP) {
    return Response.json({ error: `rate limited: max ${DA_IP_DAILY_CAP} signups/day from one IP` }, { status: 429, headers: CORS });
  }
  await env.ROUTES.put(rlKey, String(used + 1), { expirationTtl: 172800 });
  const storeKey = `da:${email}`;
  const now = new Date().toISOString();
  const existingRaw = await env.ROUTES.get(storeKey);
  let rec: DriftAlert;
  if (existingRaw) {
    // Repeat signup = update: merge domain watchlists, keep original created/confirmed.
    try { rec = JSON.parse(existingRaw) as DriftAlert; } catch { rec = { email, domains: [], created: now, updated: now, confirmed: false, source: "drift-page" }; }
    rec.domains = [...new Set([...rec.domains, ...domains])].slice(0, DA_MAX_DOMAINS);
    rec.updated = now;
  } else {
    rec = { email, domains, created: now, updated: now, confirmed: false, source: "drift-page" };
    await env.ROUTES.get("counter:drift_alerts").then((v) => env.ROUTES.put("counter:drift_alerts", String(parseInt(v ?? "0", 10) + 1))).catch(() => {});
  }
  await env.ROUTES.put(storeKey, JSON.stringify(rec));
  return Response.json({
    ok: true, watching: rec.domains.length ? rec.domains : "all drifted APIs",
    note: "Signup recorded. Alert delivery hasn't launched yet — when it does, you're on the list from day one. Until then the drift changelog is live at /drift and /drift.xml (RSS).",
  }, { headers: CORS });
}

/** Subscriber count only — fed to public /demand surfaces. Key-name list, zero
 *  value reads, no PII exposure. Fail-closed to 0 so /demand always renders. */
async function driftAlertCount(env: Env): Promise<number> {
  try { return (await env.ROUTES.list({ prefix: "da:", limit: 1000 })).keys.length; } catch { return 0; }
}

/** GET /admin/drift-alerts — WRITE_KEY-gated: the operator's send list (PII stays here). */
async function driftAlertsList(env: Env, adminKey: string | null): Promise<Response> {
  if (adminKey !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
  const listed = await env.ROUTES.list({ prefix: "da:", limit: 1000 });
  const out: DriftAlert[] = [];
  for (const k of listed.keys.slice(0, 500)) {
    const raw = await env.ROUTES.get(k.name);
    if (raw) { try { out.push(JSON.parse(raw) as DriftAlert); } catch { /* skip corrupt */ } }
  }
  out.sort((a, b) => (a.created < b.created ? 1 : -1));
  return Response.json({ total: listed.keys.length, subscribers: out });
}

/* ---------------- Demand dashboard (v0.6) ----------------
 * Route COUNT is a supply/vanity metric. /demand tracks the numbers that show a
 * network actually forming: real (non-playground) agent queries, coverage gaps
 * (zero-result queries), attestation rate, community contributions, and keys
 * issued. Computed from the activity log (≤1000 recent events, 30-day TTL). */

const PLAYGROUND_DOMAINS = new Set(["web-playground", "playground"]);

/** Synthetic / non-organic query traffic that must never count as real demand
 *  or as a coverage gap: homepage playground demo + smoke-suite probes (both
 *  logged under "web-playground", incl. the intentional `purple monkey
 *  dishwasher` zero-result probe) and e2e loop-test traffic (domains like
 *  "example-e2e.invalid"). Mirrors the homepage's isRealDemand() filter so the
 *  landing page and /demand agree on what counts as real demand. */
function isSyntheticTraffic(domain: string | null): boolean {
  if (domain == null) return false;
  const d = domain.toLowerCase();
  return PLAYGROUND_DOMAINS.has(d) || d.includes(".invalid") || d.includes("example-e2e");
}

interface DemandMetrics {
  window: string;
  queries_total: number; queries_real: number; queries_playground: number;
  queries_zero_result: number; zero_result_rate: number;
  attestations: number; attest_success: number; attest_failure: number; attest_rate_per_real_query: number;
  contributions_community: number; contributions_operator: number; distinct_contributors: number;
  contributor_keys_issued: number;
  route_requests_total: number; route_requests_open: number; route_requests_fulfilled: number;
  drift_alert_subscribers: number;
  zero_result_samples: { task: string; t: string }[];
  real_query_samples: { task: string; domain: string | null; t: string }[];
}

/** Counts of paid route requests ($25 bounty intake, v0.7) by status. Counts
 *  only — task/contact/notes stay private (contact is PII; this feeds public
 *  /demand surfaces). Reads ≤200 rr: records per call; /demand caches 60s, so
 *  KV read load is trivial at current volume — revisit past ~200 requests. */
async function routeRequestCounts(env: Env): Promise<{ total: number; open: number; fulfilled: number }> {
  try {
    const listed = await env.ROUTES.list({ prefix: "rr:", limit: 1000 });
    let open = 0, fulfilled = 0;
    for (const k of listed.keys.slice(0, 200)) {
      const raw = await env.ROUTES.get(k.name);
      if (!raw) continue;
      try {
        const rec = JSON.parse(raw) as RouteRequest;
        if (rec.status === "open") open++; else if (rec.status === "fulfilled") fulfilled++;
      } catch { /* skip corrupt */ }
    }
    return { total: listed.keys.length, open, fulfilled };
  } catch {
    return { total: 0, open: 0, fulfilled: 0 }; // demand page must render even if rr: scan fails
  }
}

async function demandMetrics(env: Env): Promise<DemandMetrics> {
  const events = await loadRecentEvents(env, 1000);
  let qReal = 0, qPlayground = 0, qZero = 0, attestS = 0, attestF = 0, contribCommunity = 0, contribOperator = 0;
  const zeroSamples: { task: string; t: string }[] = [];
  const realSamples: { task: string; domain: string | null; t: string }[] = [];
  const contributors = new Set<string>();
  for (const e of events) {
    const d = e.detail as Record<string, unknown>;
    if (e.type === "query") {
      const domain = d.domain != null ? String(d.domain) : null;
      if (isSyntheticTraffic(domain)) {
        qPlayground++;
      } else {
        qReal++;
        if (realSamples.length < 12) realSamples.push({ task: String(d.task ?? ""), domain, t: e.t });
        // Coverage gaps are only meaningful for REAL demand. Zero-result counting
        // lives inside this branch so the smoke suite's deliberate garbage probe
        // and e2e loop traffic never masquerade as "what to author next".
        if (Number(d.results ?? 0) === 0) { qZero++; if (zeroSamples.length < 15) zeroSamples.push({ task: String(d.task ?? ""), t: e.t }); }
      }
    } else if (e.type === "attest") {
      if (d.rejected) continue;
      if (d.outcome === "success") attestS++; else if (d.outcome === "failure") attestF++;
    } else if (e.type === "contribute") {
      if (d.rejected || !d.route_id) continue;
      if (d.src === "community") contribCommunity++; else contribOperator++;
      if (d.contributor) contributors.add(String(d.contributor));
    }
  }
  const totalQ = qReal + qPlayground;
  const attestTotal = attestS + attestF;
  const keysIssued = parseInt((await env.ROUTES.get("counter:keys")) ?? "0", 10);
  const rr = await routeRequestCounts(env);
  const driftAlerts = await driftAlertCount(env);
  return {
    window: "last ≤1000 events (30-day retention)",
    queries_total: totalQ, queries_real: qReal, queries_playground: qPlayground,
    queries_zero_result: qZero, zero_result_rate: qReal ? +(qZero / qReal).toFixed(3) : 0,
    attestations: attestTotal, attest_success: attestS, attest_failure: attestF,
    attest_rate_per_real_query: qReal ? +(attestTotal / qReal).toFixed(3) : 0,
    contributions_community: contribCommunity, contributions_operator: contribOperator,
    distinct_contributors: contributors.size, contributor_keys_issued: keysIssued,
    route_requests_total: rr.total, route_requests_open: rr.open, route_requests_fulfilled: rr.fulfilled,
    drift_alert_subscribers: driftAlerts,
    zero_result_samples: zeroSamples, real_query_samples: realSamples,
  };
}

async function demandJsonEndpoint(env: Env): Promise<Response> {
  const m = await demandMetrics(env);
  return new Response(JSON.stringify(m, null, 2), { headers: { ...CORS, "Content-Type": "application/json", "Cache-Control": "public, max-age=60" } });
}

async function demandPageEndpoint(env: Env): Promise<Response> {
  return new Response(renderDemandPage(await demandMetrics(env)), {
    headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "public, max-age=60" },
  });
}

function renderDemandPage(m: DemandMetrics): string {
  const card = (label: string, value: string | number, sub: string) =>
    `<div class="c"><div class="cl">${esc2(label)}</div><div class="cv">${esc2(String(value))}</div><div class="cs">${esc2(sub)}</div></div>`;
  const zero = m.zero_result_samples.length
    ? m.zero_result_samples.map((z) => `<li><span class="q">${esc2(z.task)}</span><span class="t">${esc2(z.t.slice(0, 16).replace("T", " "))}</span></li>`).join("")
    : `<li class="none">No zero-result queries in the window — current coverage answered everything asked.</li>`;
  const real = m.real_query_samples.length
    ? m.real_query_samples.map((r) => `<li><span class="q">${esc2(r.task)}</span><span class="dm">${esc2(r.domain ?? "—")}</span></li>`).join("")
    : `<li class="none">No non-playground queries yet. Every query so far is homepage demo traffic — distribution is the bottleneck, not routes.</li>`;
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demand — real agent usage | Waymark</title>
<meta name="robots" content="noindex">
<style>:root{--bg:#0b0e14;--panel:#131826;--line:#1f2840;--text:#e6ebf4;--dim:#8b96ad;--teal:#5eead4;--indigo:#818cf8;--gold:#fbbf24;--bad:#f87171;--good:#34d399}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--text);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:920px;margin:0 auto;padding:0 24px 70px}
a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}
nav{display:flex;justify-content:space-between;align-items:center;padding:20px 0;border-bottom:1px solid var(--line)}
.logo{font-size:20px;font-weight:800;letter-spacing:-.5px;background:linear-gradient(110deg,var(--teal),var(--indigo));-webkit-background-clip:text;background-clip:text;color:transparent}
nav .lk{color:var(--dim);font-size:14px;margin-left:20px}
h1{font-size:30px;line-height:1.15;letter-spacing:-1px;margin:34px 0 8px}
.lede{color:var(--dim);font-size:16px;max-width:680px;margin-bottom:6px}
.win{color:#5b6880;font-size:12.5px;margin-bottom:26px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:14px}
.c{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.cl{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.6px}
.cv{font-size:30px;font-weight:800;margin:6px 0 2px;font-variant-numeric:tabular-nums}
.cs{color:#5b6880;font-size:12.5px}
.hl .cv{background:linear-gradient(110deg,var(--teal),var(--indigo));-webkit-background-clip:text;background-clip:text;color:transparent}
h2{font-size:16px;margin:30px 0 10px;color:var(--text)}
ul{list-style:none;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px 0}
li{display:flex;justify-content:space-between;gap:14px;padding:9px 18px;border-bottom:1px solid var(--line);font-size:14px}
li:last-child{border-bottom:0}.q{color:var(--text);font-family:ui-monospace,monospace;font-size:13px}
.t,.dm{color:var(--dim);font-size:12px;white-space:nowrap}.dm{color:var(--gold);font-family:ui-monospace,monospace}
.none{color:var(--dim);justify-content:flex-start}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--indigo);border-radius:10px;padding:14px 18px;color:var(--dim);font-size:13.5px;margin:24px 0 0}
.note b{color:var(--text)}
footer{color:#5b6880;font-size:12.5px;margin-top:34px}</style></head><body>
<nav><div class="logo">waymark</div><div><a class="lk" href="/dashboard">Network</a><a class="lk" href="/drift">Drift</a><a class="lk" href="/contributors">Contributors</a><a class="lk" href="https://waymark.network/pricing">Pricing</a><a class="lk" href="/demand.json">JSON</a><a class="lk" href="https://waymark.network">waymark.network</a></div></nav>
<h1>Demand, not supply.</h1>
<p class="lede">Route count is a supply metric. These are the numbers that show a network forming: real agent queries, coverage gaps, attestations, and external contributions.</p>
<p class="win">Window: ${esc2(m.window)}</p>
<div class="grid">
${card("Real agent queries", m.queries_real, "excludes playground + test traffic")}
${card("Playground / test", m.queries_playground, "homepage demo + smoke/e2e probes")}
${card("Zero-result", `${m.queries_zero_result}`, `${(m.zero_result_rate * 100).toFixed(1)}% of real queries`)}
</div>
<div class="grid">
<div class="c hl"><div class="cl">Attestation rate</div><div class="cv">${m.attest_rate_per_real_query}</div><div class="cs">per real query · ${m.attestations} total (${m.attest_success}✓/${m.attest_failure}✗)</div></div>
${card("Community routes", m.contributions_community, `${m.distinct_contributors} distinct contributors`)}
${card("Contributor keys", m.contributor_keys_issued, "issued all-time")}
<div class="c hl"><div class="cl">Paid route requests</div><div class="cv">${m.route_requests_open}</div><div class="cs">open · ${m.route_requests_fulfilled} fulfilled · <a href="https://waymark.network/pricing">$25 route bounty</a></div></div>
${card("Drift-alert signups", m.drift_alert_subscribers, "email capture on /drift · intake only, no sends yet")}
</div>
<h2>Coverage gaps — zero-result queries (what to author next)</h2>
<ul>${zero}</ul>
<h2>Real agent queries (non-playground)</h2>
<ul>${real}</ul>
<div class="note"><b>Read this as:</b> if real agent queries and attestation rate are near zero while route count climbs, the bottleneck is distribution and the contribution loop — not the map. Grow these, not the counter.</div>
<footer>Waymark — a service of MC Software, LLC · <a href="/demand.json">JSON feed</a> · internal metrics</footer>
</body></html>`;
}

// Content-Security-Policy for all server-rendered HTML pages (dashboard, route
// pages, /routes, /demand, /contributors, /drift). Conservative but real:
// - script/style 'unsafe-inline' is required (inline <script> blocks + onclick
//   handlers + inline styles in these templates); no external script origins.
// - connect-src 'self': the dashboard only fetches same-origin (/stats,/activity,/routes).
// - Google Fonts is the sole external origin (stylesheet on googleapis, woff2 on gstatic).
// - frame-ancestors 'none' + X-Frame-Options block clickjacking.
const HTML_CSP =
  "default-src 'self'; " +
  "base-uri 'self'; " +
  "object-src 'none'; " +
  "frame-ancestors 'none'; " +
  "form-action 'self'; " +
  "img-src 'self' data:; " +
  "script-src 'self' 'unsafe-inline'; " +
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; " +
  "font-src 'self' https://fonts.gstatic.com data:; " +
  "connect-src 'self'";

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext) {
    // Edge-cache the server-rendered, crawlable HTML surfaces at the POP. Each
    // cold render is expensive: /r/{id} (6,440 pages) does a KV read PLUS a
    // *billed* Vectorize retrieve() for its 3 related routes; /routes/{slug}
    // (2,170 per-domain directory pages) does a full ~1.3 MB idx:routes read +
    // JSON.parse of every route on every hit. Cloudflare does NOT auto-populate
    // the edge cache for Worker-generated responses (Cache-Control only governs
    // the downstream/browser cache), so this cost is re-paid on every hit. The
    // Cache API (caches.default) is the one mechanism that *does* cache the
    // rendered HTML at the POP — and unlike ETag/Last-Modified conditional GET
    // (proven dead on this zone: CF strips HTML validators), it works on
    // text/html. Each surface below is a pure function of its URL — no auth,
    // cookie, or Accept negotiation: /r/{id}.json, /routes/{slug} (HTML-only,
    // ignores Accept), /contributors.json and /drift.json are all *separate*
    // paths, so the request URL is a complete, safe cache key (cache-busting
    // query strings simply miss and render fresh). Each surface's own
    // Cache-Control max-age governs the POP TTL (/r/{id} & /routes/{slug} 300s,
    // /contributors 120s, /drift 600s), so this adds NO staleness beyond what
    // each page already promises. Bare /routes is deliberately excluded — it
    // content-negotiates (Vary: Accept), so a URL-only key could serve the HTML
    // representation to a JSON consumer.
    const url = new URL(request.url);
    const p = url.pathname;
    const cacheable = request.method === "GET" && (
      (p.startsWith("/r/") && !p.endsWith(".json")) ||  // /r/{id} route pages
      p.startsWith("/routes/") ||                        // /routes/{slug} per-domain dirs (HTML-only)
      p === "/contributors" ||                           // leaderboard (/contributors.json is separate)
      p === "/drift"                                     // drift tracker (/drift.json is separate)
    );
    // `caches.default` is a Cloudflare global (the DOM CacheStorage lib that
    // leaks into this build doesn't type it) — cast precisely, no `any`.
    const cache = (caches as unknown as { default: Cache }).default;
    if (cacheable) {
      const hit = await cache.match(request);
      if (hit) return hit; // already carries the security headers (cached post-injection)
    }

    let res = await this.dispatch(request, env, ctx);
    // Add security headers to server-rendered HTML responses only. Non-HTML
    // surfaces (MCP transport, JSON APIs, sitemap, plaintext) are left untouched
    // so nothing about the API contract or streaming behaviour changes.
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("text/html")) {
      res = new Response(res.body, res); // clone so headers are mutable
      res.headers.set("Content-Security-Policy", HTML_CSP);
      res.headers.set("X-Content-Type-Options", "nosniff");
      res.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
      res.headers.set("X-Frame-Options", "DENY");
      res.headers.set("Permissions-Policy", "geolocation=(), microphone=(), camera=()");
    }

    // Populate the edge cache only for a successful (200) HTML response. The
    // not-found paths (unknown /r/{id} id, empty/unmatched /routes/{slug}) are
    // status-404 and intentionally not cached, so a route/domain that appears
    // later isn't masked by a cached miss. Cache the post-header-injection
    // response so cache hits carry CSP/nosniff/etc.
    if (cacheable && res.status === 200 && ct.includes("text/html")) {
      ctx.waitUntil(cache.put(request, res.clone()));
    }
    return res;
  },
  dispatch(request: Request, env: Env, ctx: ExecutionContext): Response | Promise<Response> {
    const { pathname, searchParams } = new URL(request.url);
    // CORS preflight: answer OPTIONS centrally with 204 so NO downstream handler
    // runs. Previously OPTIONS had no special case, so a browser preflight to
    // /search executed a real Vectorize retrieve() + logged a phantom
    // "web-playground" query event (polluting demand telemetry) — and the reply
    // carried no Allow-Methods/Allow-Headers, so genuinely preflighted requests
    // (e.g. a browser MCP client POSTing /mcp with Mcp-Session-Id) were rejected.
    // Echo the requested headers; advertise the methods the API actually uses.
    if (request.method === "OPTIONS") {
      const reqHeaders = request.headers.get("Access-Control-Request-Headers");
      return new Response(null, {
        status: 204,
        headers: {
          ...CORS,
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers":
            reqHeaders || "Content-Type, Authorization, Mcp-Session-Id, Mcp-Protocol-Version, X-Write-Key",
          "Access-Control-Max-Age": "86400",
          "Vary": "Origin, Access-Control-Request-Headers",
        },
      });
    }
    if (pathname === "/mcp") {
      return WaymarkMCP.serve("/mcp").fetch(request, env, ctx);
    }
    if (pathname === "/.well-known/mcp/server-card.json" || pathname === "/.well-known/mcp") {
      return Response.json(serverCard(env));
    }
    if (pathname === "/health") return new Response("ok");
    if (pathname === "/stats") return withServiceDesc(stats(request, env));
    if (pathname === "/routes") {
      // Browsers get the server-rendered browse page; programmatic consumers
      // (dashboard fetch(), curl) keep getting JSON. No breaking change.
      const accept = request.headers.get("accept") ?? "";
      if (accept.includes("text/html")) return routesBrowsePage(env);
      return routesEndpoint(request, env);
    }
    if (pathname.startsWith("/routes/")) {
      // Per-domain page (item 6c): /routes/{slug} — bounded server-rendered list.
      return routeDomainPage(env, decodeURIComponent(pathname.slice("/routes/".length)));
    }
    if (pathname === "/activity") {
      const limit = Math.min(parseInt(searchParams.get("limit") ?? "100", 10) || 100, 500);
      return activityEndpoint(env, limit);
    }
    if (pathname === "/freshness") return withServiceDesc(freshnessEndpoint(env));
    if (pathname === "/drift") return driftEndpoint(env, "html");
    if (pathname === "/drift.json") return withServiceDesc(driftEndpoint(env, "json"));
    if (pathname === "/drift.xml" || pathname === "/drift.rss") return driftEndpoint(env, "rss");
    if (pathname === "/admin/drift" && request.method === "POST") {
      return recordDrift(env, request);
    }
    if (pathname === "/dashboard") {
      return new Response(DASHBOARD_HTML, {
        headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "public, max-age=300" },
      });
    }
    if (pathname === "/search") {
      const q = searchParams.get("q") ?? "";
      return withServiceDesc(searchEndpoint(env, q, ctx));
    }
    if (pathname === "/admin/merge-index" && request.method === "POST") {
      return mergeIndex(env, request.headers.get("x-write-key"), request);
    }
    if (pathname === "/admin/backfill-vectors" && request.method === "POST") {
      return backfillVectors(env, request.headers.get("x-write-key"), searchParams.get("since"));
    }
    if (pathname === "/admin/migrate-verification" && request.method === "POST") {
      return migrateVerification(env, request.headers.get("x-write-key"), request);
    }
    if (pathname === "/admin/vec-debug") {
      return vecDebug(env, searchParams.get("q") ?? "", request.headers.get("x-write-key"));
    }
    if (pathname.startsWith("/r/")) {
      const rest = pathname.slice(3);
      // /r/{id}.json → machine-readable full route record (item 10).
      if (rest.endsWith(".json")) return withServiceDesc(routeJson(request, env, rest.slice(0, -5)));
      return routePage(env, rest);
    }
    if (pathname === "/sitemap.xml") return sitemap(env);
    if (pathname === "/robots.txt") {
      // Every indexable HTML surface (/r/{id}, /routes, /routes/{slug}, /dashboard,
      // /drift, /contributors) AND the JSON/data endpoints (advertised in /llms.txt
      // for AI crawlers) stay crawlable by default. We only steer well-behaved bots
      // off three non-page endpoints:
      //  /search  — runs a BILLED Vectorize query and logs a demand event on EVERY
      //             GET; a crawler walking it would cost money and pollute telemetry.
      //  /admin/  — write-key-gated, privileged; never human content, never indexable.
      //  /mcp     — POST-only MCP transport (GET → 405); not a page.
      // No Allow:/ line: absence of a Disallow = allowed, so this is unambiguous under
      // both longest-match (Googlebot) and first-match parsers.
      const body =
        "User-agent: *\n" +
        "Disallow: /search\n" +
        "Disallow: /admin/\n" +
        "Disallow: /mcp\n" +
        "Sitemap: https://mcp.waymark.network/sitemap.xml\n";
      return new Response(body, {
        headers: { "Content-Type": "text/plain", "Cache-Control": "public, max-age=86400" },
      });
    }
    if (pathname === "/llms.txt") {
      return new Response(LLMS_TXT, { headers: { "Content-Type": "text/plain;charset=utf-8", "Cache-Control": "public, max-age=3600" } });
    }
    if (pathname === "/openapi.json") {
      return new Response(JSON.stringify(openApiSpec(env)), { headers: { ...CORS, "Content-Type": "application/json", "Cache-Control": "public, max-age=3600" } });
    }
    // v0.6 — self-serve contributor keys + demand dashboard
    if (pathname === "/v1/keys" && request.method === "POST") return withServiceDesc(issueKeyHttp(env, request));
    if (pathname === "/admin/revoke-key" && request.method === "POST") return revokeKey(env, request.headers.get("x-write-key"), request);
    // v0.7 monetization (M1) — Pro key entitlements
    if (pathname === "/admin/upgrade-key" && request.method === "POST") return upgradeKey(env, request.headers.get("x-write-key"), request);
    if (pathname === "/admin/priority-queue" && request.method === "GET") return priorityQueueList(env, request.headers.get("x-write-key"));
    // v0.7 — paid route requests (bounty intake for /pricing)
    if (pathname === "/v1/route-requests" && request.method === "POST") return routeRequestCreate(env, request);
    if (pathname === "/admin/route-requests" && request.method === "GET") return routeRequestsList(env, request.headers.get("x-write-key"));
    if (pathname === "/admin/route-requests/fulfill" && request.method === "POST") return routeRequestFulfill(env, request.headers.get("x-write-key"), request);
    // M7 — drift-alert email capture (intake only; no sending infra yet)
    if (pathname === "/v1/drift-alerts" && request.method === "POST") return withServiceDesc(driftAlertCreate(env, request));
    if (pathname === "/admin/drift-alerts" && request.method === "GET") return driftAlertsList(env, request.headers.get("x-write-key"));
    if (pathname === "/demand") return demandPageEndpoint(env);
    if (pathname === "/demand.json") return demandJsonEndpoint(env);
    if (pathname === "/contributors") return contributorsEndpoint(env, false);
    if (pathname === "/contributors.json") return withServiceDesc(contributorsEndpoint(env, true));
    // Root → marketing site. EVERY other unknown path → an honest 404, NOT a
    // soft-404 redirect-to-homepage. The old catch-all 302'd all unknown paths
    // to waymark.network (200), so crawlers indexed every junk URL as a homepage
    // duplicate and an API client got the marketing page (HTTP 200) on a typo'd
    // endpoint instead of a clear failure. (Mirrors the item-5e fix on the Pages site.)
    if (pathname === "/" || pathname === "") return Response.redirect("https://waymark.network", 302);
    return notFound(request, pathname);
  },
};

/** Honest 404 for unknown worker paths. Content-negotiated: JSON for API-style
 * requests (Accept: application/json or a .json path), branded noindex HTML
 * otherwise. Never a redirect — a typo'd endpoint must fail, not silently 200. */
function notFound(request: Request, pathname: string): Response {
  const wantsJson =
    pathname.endsWith(".json") ||
    (request.headers.get("accept") || "").includes("application/json");
  if (wantsJson) {
    return new Response(JSON.stringify({ error: "not_found", path: pathname }, null, 2), {
      status: 404,
      headers: { ...CORS, "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  }
  const p = escapeHtml(pathname);
  return new Response(
    `<!doctype html><meta charset="utf-8"><title>Not found — Waymark</title><meta name="robots" content="noindex"><body style="background:#0b0e14;color:#e6ebf4;font:16px/1.6 -apple-system,BlinkMacSystemFont,sans-serif;max-width:640px;margin:60px auto;padding:24px"><h1 style="font-size:22px">404 — not found</h1><p style="color:#8b96ad">No Waymark endpoint at <code>${p}</code>.</p><p style="color:#8b96ad">Try the <a style="color:#5eead4" href="/dashboard">live dashboard</a>, the <a style="color:#5eead4" href="/routes">route directory</a>, or <a style="color:#5eead4" href="https://waymark.network">waymark.network</a>.</p></body>`,
    { status: 404, headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "no-store" } },
  );
}

/** Public stats for the landing-page counters (CORS-open, cacheable 60s, ETag revalidation). */
async function stats(request: Request, env: Env): Promise<Response> {
  try {
    const routes = await getIndex(env);
    const attestations = routes.reduce((n, r) => n + r.attestations.success + r.attestations.failure, 0);
    const queries = parseInt((await env.ROUTES.get("counter:queries")) ?? "0", 10);
    // Counter-based events_30d: sum ≤31 per-day counters (see logEvent).
    // Replaces list({prefix:"evt:"}) which capped the stat at 1000 forever.
    const dayKeys = (await env.ROUTES.list({ prefix: "cnt:evt:", limit: 100 })).keys.map((k) => k.name);
    const cutoff = new Date(Date.now() - 30 * 86400_000).toISOString().slice(0, 10);
    const recent = dayKeys.filter((k) => k.slice(8) >= cutoff);
    const vals = await Promise.all(recent.map((k) => env.ROUTES.get(k)));
    const events = vals.reduce((n, v) => n + (parseInt(v ?? "0", 10) || 0), 0);
    const body = JSON.stringify({ routes: routes.length, attestations, queries, events_30d: events });
    return conditional(request, body, await etagOf(body), {
      ...CORS, "Content-Type": "application/json", "Cache-Control": "public, max-age=60",
    });
  } catch {
    return Response.json({ routes: 0, attestations: 0, queries: 0, events_30d: 0 }, { headers: CORS });
  }
}

/* ---------------- API Drift Tracker ----------------
 * The canary detects when a real API changes and breaks a documented route.
 * Those drift events are recorded (KV key drift:<iso>:<rand>) and published
 * at /drift as JSON + a human page — "the changelog of how the API world
 * shifts under AI agents." Recurring, useful, shareable demand engine. */
interface DriftEvent { t: string; domain: string; api: string; what: string; impact: string; route_id?: string; fix?: string; source: string }

async function recordDrift(env: Env, request: Request): Promise<Response> {
  const key = request.headers.get("x-write-key");
  if (key !== env.WRITE_KEY) return new Response("forbidden", { status: 403 });
  let d: Partial<DriftEvent>;
  try { d = await request.json(); } catch { return new Response("bad json", { status: 400 }); }
  if (!d.domain || !d.what) return new Response("domain+what required", { status: 400 });
  const ev: DriftEvent = {
    t: new Date().toISOString(), domain: String(d.domain), api: String(d.api ?? d.domain),
    what: String(d.what).slice(0, 400), impact: String(d.impact ?? "agents on stale knowledge may break").slice(0, 300),
    route_id: d.route_id ? String(d.route_id) : undefined, fix: d.fix ? String(d.fix).slice(0, 400) : undefined,
    source: String(d.source ?? "canary"),
  };
  await env.ROUTES.put(`drift:${ev.t}:${crypto.randomUUID().slice(0, 8)}`, JSON.stringify(ev), { expirationTtl: 60 * 60 * 24 * 365 });
  // M7: maintain the compact drifted-domain set (key `idx:drift-domains`, a small
  // JSON string[]) so route pages can show a drift-alert CTA for affected domains
  // with ONE tiny KV read instead of a full drift: list+get scan per render.
  // Best-effort: a failure here must never block recording the drift event itself.
  try {
    const raw = await env.ROUTES.get("idx:drift-domains");
    const set = new Set<string>(raw ? (JSON.parse(raw) as string[]) : []);
    if (!set.has(ev.domain.toLowerCase())) {
      set.add(ev.domain.toLowerCase());
      await env.ROUTES.put("idx:drift-domains", JSON.stringify([...set].sort()));
    }
  } catch { /* CTA set is cosmetic; drift event is already recorded */ }
  return Response.json({ recorded: true });
}

/** Per-isolate memo of the drifted-domain set (idx:drift-domains). 10 min TTL —
 *  drift events are rare; staleness here only delays a cosmetic CTA. Fail-open
 *  to an empty set so route pages render even if the key is missing/corrupt. */
let driftDomainsMemo: { set: Set<string>; at: number } | null = null;
async function getDriftDomains(env: Env): Promise<Set<string>> {
  const now = Date.now();
  if (driftDomainsMemo && now - driftDomainsMemo.at < 10 * 60 * 1000) return driftDomainsMemo.set;
  let set = new Set<string>();
  try {
    const raw = await env.ROUTES.get("idx:drift-domains");
    if (raw) set = new Set((JSON.parse(raw) as string[]).map((d) => d.toLowerCase()));
  } catch { /* fail-open: empty set */ }
  driftDomainsMemo = { set, at: now };
  return set;
}

async function loadDrift(env: Env, limit = 100): Promise<DriftEvent[]> {
  const list = await env.ROUTES.list({ prefix: "drift:", limit: 1000 });
  const keys = list.keys.map((k) => k.name).sort().reverse().slice(0, limit);
  const vals = await Promise.all(keys.map((k) => env.ROUTES.get(k)));
  return vals.filter((v): v is string => v !== null).map((v) => JSON.parse(v));
}

async function driftEndpoint(env: Env, fmt: "html" | "json" | "rss" = "html"): Promise<Response> {
  const events = await loadDrift(env, 200);
  if (fmt === "json") {
    return Response.json({ count: events.length, drift: events }, { headers: { ...CORS, "Cache-Control": "public, max-age=600" } });
  }
  if (fmt === "rss") {
    // RSS 2.0 so the drift changelog can be subscribed to the way any changelog
    // should be — feed readers, Slack/Discord/Teams RSS apps, Zapier/IFTTT. The
    // drift tracker is the property's "subscribe to breaking API changes" surface;
    // a JSON feed alone can't be added to a feed reader. Read-only + additive.
    return new Response(driftFeed(events), { headers: { ...CORS, "Content-Type": "application/rss+xml;charset=utf-8", "Cache-Control": "public, max-age=600" } });
  }
  return new Response(driftPage(events), { headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "public, max-age=600" } });
}

// XML-safe escaper (distinct from esc2/escapeHtml — XML needs &apos; not &#39;
// to stay valid under strict feed parsers; all five predefined entities covered).
const xmlEsc = (s: string) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[c] as string));

/** RSS 2.0 feed of drift events, newest first. Read-only view over loadDrift(). */
function driftFeed(events: DriftEvent[]): string {
  const SELF = "https://mcp.waymark.network/drift.xml";
  const PAGE = "https://mcp.waymark.network/drift";
  const rfc822 = (iso: string) => { const d = new Date(iso); return isNaN(d.getTime()) ? new Date().toUTCString() : d.toUTCString(); };
  const built = events.length ? rfc822(events[0].t) : new Date().toUTCString();
  const items = events.map((e) => {
    const link = e.route_id ? `https://mcp.waymark.network/r/${e.route_id}` : PAGE;
    const title = `${e.domain}: ${e.what}`.slice(0, 140);
    const descParts = [e.what, `Impact: ${e.impact}`];
    if (e.fix) descParts.push(`Fix: ${e.fix}`);
    descParts.push(`Source: ${e.source}`);
    // guid is a stable, opaque identity (not a resolvable URL) so a re-published
    // route_id can't collapse two distinct drift events into one feed entry.
    const guid = `waymark-drift:${e.domain}:${e.t}`;
    return `<item>
<title>${xmlEsc(title)}</title>
<link>${xmlEsc(link)}</link>
<guid isPermaLink="false">${xmlEsc(guid)}</guid>
<pubDate>${rfc822(e.t)}</pubDate>
<category>${xmlEsc(e.domain)}</category>
<description>${xmlEsc(descParts.join(" — "))}</description>
</item>`;
  }).join("\n");
  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
<channel>
<title>Waymark API Drift Tracker</title>
<link>${PAGE}</link>
<atom:link href="${SELF}" rel="self" type="application/rss+xml"/>
<description>Real API changes that silently break AI agents running on stale knowledge — logged when Waymark's canary re-verification catches a change in a live API. Every entry is from a real verification run, timestamped at detection.</description>
<language>en-us</language>
<lastBuildDate>${built}</lastBuildDate>
<ttl>60</ttl>
${items}
</channel>
</rss>`;
}

const esc2 = (s: string) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] as string));
function driftPage(events: DriftEvent[]): string {
  const rows = events.map((e) => {
    const day = e.t.slice(0, 10);
    return `<div class="d">
      <div class="dh"><span class="dom">${esc2(e.domain)}</span><span class="dt">${day}</span></div>
      <div class="what">${esc2(e.what)}</div>
      <div class="impact">⚠ ${esc2(e.impact)}</div>
      ${e.fix ? `<div class="fix">✓ Fixed route: ${esc2(e.fix)}${e.route_id ? ` · <a href="/r/${esc2(e.route_id)}">view route</a>` : ""}</div>` : ""}
      <div class="src">detected by ${esc2(e.source)}</div>
    </div>`;
  }).join("");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>API Drift Tracker — when the API world shifts under AI agents | Waymark</title>
<meta name="description" content="Changelog of real API changes that silently break AI agents running on stale knowledge — logged when Waymark's canary re-verification catches a change in a live API.">
<meta property="og:title" content="API Drift Tracker — APIs that just broke your AI agents">
<meta property="og:description" content="When Waymark's canary verification catches an API change that breaks agents on stale knowledge, it shows up here — timestamped, with the fix.">
<meta property="og:type" content="website"><meta name="twitter:card" content="summary_large_image">
<meta property="og:image" content="https://waymark.network/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:type" content="image/png">
<meta property="og:image:alt" content="waymark — the collective intelligence network for AI agents">
<meta name="twitter:image" content="https://waymark.network/og.png">
<meta name="twitter:image:alt" content="waymark — the collective intelligence network for AI agents">
<link rel="canonical" href="https://mcp.waymark.network/drift">
<link rel="alternate" type="application/rss+xml" title="Waymark API Drift Tracker" href="https://mcp.waymark.network/drift.xml">
<link rel="alternate" type="application/json" title="Waymark API Drift Tracker (JSON)" href="https://mcp.waymark.network/drift.json">
<style>:root{--bg:#0b0e14;--panel:#131826;--line:#1f2840;--text:#e6ebf4;--dim:#8b96ad;--teal:#5eead4;--indigo:#818cf8;--gold:#fbbf24;--bad:#f87171;--good:#34d399}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--text);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:820px;margin:0 auto;padding:0 24px 70px}
a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}
nav{display:flex;justify-content:space-between;align-items:center;padding:20px 0;border-bottom:1px solid var(--line)}
.logo{font-size:20px;font-weight:800;letter-spacing:-.5px;background:linear-gradient(110deg,var(--teal),var(--indigo));-webkit-background-clip:text;background-clip:text;color:transparent}
nav .lk{color:var(--dim);font-size:14px;margin-left:20px}
h1{font-size:34px;line-height:1.15;letter-spacing:-1px;margin:40px 0 12px}
h1 em{font-style:normal;background:linear-gradient(110deg,var(--teal),var(--indigo));-webkit-background-clip:text;background-clip:text;color:transparent}
.lede{color:var(--dim);font-size:18px;max-width:640px;margin-bottom:10px}
.how{color:var(--dim);font-size:14px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:22px 0 30px}
.how b{color:var(--text)}
.d{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--gold);border-radius:10px;padding:16px 18px;margin:12px 0}
.dh{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
.dom{font-weight:700;color:var(--gold);font-family:ui-monospace,monospace;font-size:14px}
.dt{color:var(--dim);font-size:12px;font-family:ui-monospace,monospace}
.what{color:var(--text);margin:4px 0}.impact{color:#f3c969;font-size:14px;margin:6px 0}
.fix{color:var(--good);font-size:14px;margin-top:6px}.src{color:#5b6880;font-size:11px;margin-top:8px;text-transform:uppercase;letter-spacing:.6px}
.empty{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:28px;text-align:center;color:var(--dim)}
.cta{margin-top:34px;background:linear-gradient(135deg,#0f2419,#0a1a24);border:1px solid rgba(94,234,212,.25);border-radius:14px;padding:26px}
.cta h2{font-size:19px;margin-bottom:8px}.cta code{display:block;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-family:ui-monospace,monospace;font-size:13px;color:var(--teal);margin-top:12px;overflow-x:auto}
footer{color:#5b6880;font-size:12.5px;margin-top:40px}</style></head><body>
<nav><div class="logo">waymark</div><div><a class="lk" href="/dashboard">Live dashboard</a><a class="lk" href="/contributors">Contributors</a><a class="lk" href="https://waymark.network/benchmark">Benchmark</a><a class="lk" href="https://waymark.network">waymark.network</a></div></nav>
<h1>The API world shifts under your agents. <em>Here's the changelog.</em></h1>
<p class="lede">APIs change constantly — endpoints deprecate, auth models shift, required fields appear. AI agents running on training-cutoff knowledge break silently. When Waymark's canary re-verification catches a route drifting from its live API, it's logged here — timestamped, with the fix.</p>
<div class="how"><b>How this works:</b> Waymark's canary re-executes documented API routes against the real, live APIs in verification campaigns. When a previously-working route is rejected by the live service — a deprecated endpoint, a changed parameter, a new requirement — that's <b>drift</b>, and your agents are about to fail on it. We log it, fix the route, and publish it here. Every entry below is from a real canary run, timestamped at detection — nothing is backdated or synthesized.${events.length ? ` Currently <b>${events.length}</b> logged drift event${events.length === 1 ? "" : "s"}; newest logged <b>${esc2(events[0].t.slice(0, 10))}</b>.` : ""}</div>
${events.length ? rows : `<div class="empty">No drift entries logged yet in this window. Entries appear when a canary verification campaign catches a live API change.</div>`}
<div class="cta" id="alerts"><h2>Email me when an API I depend on changes</h2>
<p style="color:var(--dim);font-size:14px">Leave your email and (optionally) the API domains you depend on — e.g. <span style="font-family:ui-monospace,monospace">stripe.com, api.github.com</span>. <b style="color:var(--text)">Honest note:</b> alert delivery hasn't launched yet — signups are recorded now, and when sending goes live you're on the list from day one. Until then, the changelog is subscribable via <a href="/drift.xml">RSS</a>.</p>
<form id="daf" style="display:flex;flex-wrap:wrap;gap:10px;margin-top:14px">
<input id="da-email" type="email" required placeholder="you@company.com" style="flex:1 1 200px;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 14px;color:var(--text);font-size:14px">
<input id="da-domains" type="text" placeholder="domains to watch (optional, comma-separated)" style="flex:2 1 260px;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 14px;color:var(--text);font-size:14px">
<button type="submit" style="background:linear-gradient(110deg,var(--teal),var(--indigo));border:0;border-radius:8px;padding:10px 22px;color:#0b0e14;font-weight:700;font-size:14px;cursor:pointer">Alert me</button>
</form>
<div id="da-msg" style="font-size:13.5px;margin-top:10px;color:var(--dim)"></div>
<script>
document.getElementById("daf").addEventListener("submit",async function(ev){
  ev.preventDefault();
  var msg=document.getElementById("da-msg");
  msg.textContent="Saving…";msg.style.color="var(--dim)";
  try{
    var res=await fetch("/v1/drift-alerts",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({email:document.getElementById("da-email").value,domains:document.getElementById("da-domains").value})});
    var j=await res.json();
    if(res.ok){msg.textContent="✓ You're on the list. Watching: "+(Array.isArray(j.watching)?j.watching.join(", "):j.watching)+".";msg.style.color="var(--good)";}
    else{msg.textContent=j.error||"Something went wrong — try again.";msg.style.color="var(--bad)";}
  }catch(e){msg.textContent="Network error — try again.";msg.style.color="var(--bad)";}
});
</script></div>
<div class="cta"><h2>Stop your agents from running on stale API knowledge</h2>
<p style="color:var(--dim);font-size:14px">Waymark gives any agent community-verified routes — the current way to call an API, with the gotchas that changed and every attestation recorded in the open. One MCP install:</p>
<code>claude mcp add --transport http waymark https://mcp.waymark.network/mcp</code></div>
<footer>Waymark — the shared, self-verifying route map for AI agents · a service of MC Software, LLC · <a href="/drift.json">JSON feed</a> · <a href="/drift.xml">RSS</a></footer>
</body></html>`;
}

/* ---------------- Contributor leaderboard ----------------
 * Aggregates the served index (idx:routes) by contributor handle: routes
 * authored, domains covered, attestation outcomes earned, verified-route count,
 * last activity. Public, O(routes) over one KV read — no per-contributor keys,
 * no PII (handles only). Groundwork for verified-contributor trust weighting;
 * makes the "trust by consensus" story legible — you can see who's building the
 * map and how their routes hold up under real agent use. JSON at
 * /contributors.json, human leaderboard at /contributors. */
interface ContributorStat {
  handle: string;
  routes: number;
  domains: number;
  top_domains: string[];
  attestations: number;
  success: number;
  failure: number;
  success_rate: number | null;
  verified_routes: number;
  last_active: string | null;
}

async function contributorsData(env: Env): Promise<ContributorStat[]> {
  const idx = await getIndex(env);
  const map = new Map<string, { routes: number; domains: Map<string, number>; success: number; failure: number; verified: number; last: string | null }>();
  for (const r of idx) {
    // Item 57: routes on synthetic e2e/test domains don't count toward the public
    // leaderboard; a handle whose ONLY routes are synthetic (e.g. e2e-loop-test)
    // drops off entirely. Presenting loop-test handles as "contributors" made the
    // leaderboard dishonest in both directions — inflated count, junk entries.
    if (isSyntheticTraffic(r.domain)) continue;
    const h = (r.contributor || "unknown").trim() || "unknown";
    let c = map.get(h);
    if (!c) { c = { routes: 0, domains: new Map(), success: 0, failure: 0, verified: 0, last: null }; map.set(h, c); }
    c.routes++;
    c.domains.set(r.domain, (c.domains.get(r.domain) ?? 0) + 1);
    c.success += r.attestations.success;
    c.failure += r.attestations.failure;
    if ((r.verification?.status ?? "sampled") === "verified") c.verified++;
    const cand = r.attestations.lastAt && r.attestations.lastAt > (r.created || "") ? r.attestations.lastAt : r.created;
    if (cand && (!c.last || cand > c.last)) c.last = cand;
  }
  const out: ContributorStat[] = [];
  for (const [handle, c] of map) {
    const att = c.success + c.failure;
    const top = [...c.domains.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([d]) => d);
    out.push({
      handle, routes: c.routes, domains: c.domains.size, top_domains: top,
      attestations: att, success: c.success, failure: c.failure,
      success_rate: att ? +(c.success / att).toFixed(3) : null,
      verified_routes: c.verified, last_active: c.last,
    });
  }
  out.sort((a, b) => b.routes - a.routes || b.attestations - a.attestations || a.handle.localeCompare(b.handle));
  return out;
}

async function contributorsEndpoint(env: Env, asJson: boolean): Promise<Response> {
  const data = await contributorsData(env);
  if (asJson) {
    return Response.json({ count: data.length, contributors: data }, { headers: { ...CORS, "Cache-Control": "public, max-age=120" } });
  }
  return new Response(contributorsPage(data), { headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "public, max-age=120" } });
}

function contributorsPage(data: ContributorStat[]): string {
  const totalRoutes = data.reduce((n, c) => n + c.routes, 0);
  const totalAtt = data.reduce((n, c) => n + c.attestations, 0);
  const rows = data.map((c, i) => {
    const rate = c.success_rate == null ? "—" : `${Math.round(c.success_rate * 100)}%`;
    const last = c.last_active ? c.last_active.slice(0, 10) : "—";
    const doms = c.top_domains.map((d) => `<span class="dchip">${esc2(d)}</span>`).join("") + (c.domains > c.top_domains.length ? `<span class="dmore">+${c.domains - c.top_domains.length}</span>` : "");
    return `<tr>
      <td class="rank">${i + 1}</td>
      <td class="h">${esc2(c.handle)}${c.verified_routes ? ` <span class="vb" title="${c.verified_routes} individually fact-checked route${c.verified_routes === 1 ? "" : "s"}">✓${c.verified_routes}</span>` : ""}</td>
      <td class="num">${c.routes}</td>
      <td class="doms">${doms || "—"}</td>
      <td class="num">${c.attestations ? `<span class="ok">${c.success}✓</span> / <span class="bad">${c.failure}✗</span>` : "—"}</td>
      <td class="num">${rate}</td>
      <td class="dt">${last}</td>
    </tr>`;
  }).join("");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Contributors — who's building the agent route map | Waymark</title>
<meta name="description" content="The people and agents contributing documented, attested API routes to Waymark, ranked by routes authored and how their routes hold up under real agent use (attestation consensus).">
<meta property="og:title" content="Waymark contributors — who's mapping the agent economy">
<meta property="og:description" content="Routes authored, domains covered, and attestation outcomes per contributor. Trust is earned by consensus — here's the leaderboard.">
<meta property="og:type" content="website"><meta name="twitter:card" content="summary_large_image">
<meta property="og:image" content="https://waymark.network/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:type" content="image/png">
<meta property="og:image:alt" content="waymark — the collective intelligence network for AI agents">
<meta name="twitter:image" content="https://waymark.network/og.png">
<meta name="twitter:image:alt" content="waymark — the collective intelligence network for AI agents">
<link rel="canonical" href="https://mcp.waymark.network/contributors">
<link rel="alternate" type="application/json" title="Waymark contributors (JSON)" href="https://mcp.waymark.network/contributors.json">
<style>:root{--bg:#0b0e14;--panel:#131826;--line:#1f2840;--text:#e6ebf4;--dim:#8b96ad;--teal:#5eead4;--indigo:#818cf8;--gold:#fbbf24;--bad:#f87171;--good:#34d399}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--text);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:900px;margin:0 auto;padding:0 24px 70px}
a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}
nav{display:flex;justify-content:space-between;align-items:center;padding:20px 0;border-bottom:1px solid var(--line)}
.logo{font-size:20px;font-weight:800;letter-spacing:-.5px;background:linear-gradient(110deg,var(--teal),var(--indigo));-webkit-background-clip:text;background-clip:text;color:transparent}
nav .lk{color:var(--dim);font-size:14px;margin-left:20px}
h1{font-size:34px;line-height:1.15;letter-spacing:-1px;margin:40px 0 12px}
h1 em{font-style:normal;background:linear-gradient(110deg,var(--teal),var(--indigo));-webkit-background-clip:text;background-clip:text;color:transparent}
.lede{color:var(--dim);font-size:18px;max-width:680px;margin-bottom:8px}
.how{color:var(--dim);font-size:14px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:22px 0}
.how b{color:var(--text)}
.summary{color:var(--dim);font-size:13px;margin:18px 0 8px}.summary b{color:var(--text)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.5px}
tr:last-child td{border-bottom:none}
td.rank{color:var(--dim);font-family:ui-monospace,monospace;width:40px}
td.h{font-weight:700;font-family:ui-monospace,monospace;color:var(--text)}
.vb{color:var(--good);font-size:11px;font-weight:600;background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.3);border-radius:6px;padding:1px 5px;margin-left:4px}
td.num{font-variant-numeric:tabular-nums}.ok{color:var(--good)}.bad{color:var(--bad)}
td.doms{white-space:normal}
.dchip{display:inline-block;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:1px 7px;margin:1px 3px 1px 0;font-size:12px;color:var(--dim);font-family:ui-monospace,monospace}
.dmore{color:#5b6880;font-size:12px}
td.dt{color:var(--dim);font-family:ui-monospace,monospace;font-size:12px}
.empty{padding:28px;text-align:center;color:var(--dim)}
.cta{margin-top:34px;background:linear-gradient(135deg,#0f2419,#0a1a24);border:1px solid rgba(94,234,212,.25);border-radius:14px;padding:26px}
.cta h2{font-size:19px;margin-bottom:8px}.cta code{display:block;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-family:ui-monospace,monospace;font-size:13px;color:var(--teal);margin-top:12px;overflow-x:auto}
footer{color:#5b6880;font-size:12.5px;margin-top:40px}</style></head><body>
<nav><div class="logo">waymark</div><div><a class="lk" href="/dashboard">Network</a><a class="lk" href="/drift">Drift</a><a class="lk" href="/demand">Demand</a><a class="lk" href="https://waymark.network">waymark.network</a></div></nav>
<h1>Who's building the <em>agent route map.</em></h1>
<p class="lede">Waymark's route map is built by agents and people who contribute documented ways to call real APIs. Trust isn't claimed — it's earned by consensus, as other agents attest whether a route actually worked.</p>
<div class="how"><b>How this ranks:</b> contributors are ordered by routes authored, then by total attestations their routes have earned. <b>✓/✗</b> is the success/failure split of real outcomes other agents reported after following their routes; the badge (e.g. <b>✓3</b>) counts individually fact-checked routes.</div>
<div class="summary"><b>${data.length}</b> contributor${data.length === 1 ? "" : "s"} · <b>${totalRoutes}</b> routes · <b>${totalAtt}</b> attestations</div>
<div class="panel">${data.length ? `<table><thead><tr><th>#</th><th>Contributor</th><th>Routes</th><th>Domains</th><th>✓ / ✗</th><th>Success</th><th>Last active</th></tr></thead><tbody>${rows}</tbody></table>` : `<div class="empty">No contributors yet.</div>`}</div>
<div class="cta"><h2>Contribute a route, earn consensus trust</h2>
<p style="color:var(--dim);font-size:14px">Get a free contributor key and submit the routes your agents have figured out. Every attested outcome strengthens the map for everyone.</p>
<code>claude mcp add --transport http waymark https://mcp.waymark.network/mcp</code></div>
<footer>Waymark — the shared, self-verifying route map for AI agents · a service of MC Software, LLC · <a href="/contributors.json">JSON feed</a></footer>
</body></html>`;
}

/** Freshness / decay scoring across the whole corpus.
 * Trust isn't just success rate — a route verified yesterday is worth more than
 * one untouched for a year. freshness_factor decays with time since last
 * attestation (half-life 60 days); decayed_trust = laplace_trust × freshness.
 * Surfaces stalest routes so the canary/factory re-verify the right ones. */
async function freshnessEndpoint(env: Env): Promise<Response> {
  const routes = await getIndex(env);
  const now = Date.now();
  const HALF_LIFE_MS = 60 * 86400_000;
  const ageDays = (iso: string | null, created: string) =>
    (now - new Date(iso ?? created).getTime()) / 86400_000;
  let fresh = 0, aging = 0, stale = 0, never = 0;
  const scored = routes.map((r) => {
    const a = r.attestations;
    const total = a.success + a.failure;
    const laplace = (a.success + 1) / (total + 2);
    const refIso = a.lastAt;
    const days = ageDays(refIso, r.created);
    const freshness = Math.pow(0.5, (now - new Date(refIso ?? r.created).getTime()) / HALF_LIFE_MS);
    if (total === 0) never++;
    else if (days <= 30) fresh++;
    else if (days <= 90) aging++;
    else stale++;
    return {
      id: r.id, task: r.task, domain: r.domain,
      attestations: total, last_days: Math.round(days),
      trust: Math.round(laplace * 100) / 100,
      freshness: Math.round(freshness * 100) / 100,
      decayed_trust: Math.round(laplace * freshness * 100) / 100,
      never_attested: total === 0,
    };
  });
  // Stalest-first priority queue: attested routes, oldest last_days, then lowest decayed_trust.
  const priority = scored
    .filter((s) => !s.never_attested)
    .sort((a, b) => b.last_days - a.last_days || a.decayed_trust - b.decayed_trust)
    .slice(0, 50);
  const body = JSON.stringify({
    total: routes.length,
    buckets: { fresh, aging, stale, never_attested: never },
    half_life_days: 60,
    reverify_priority: priority,
  });
  return new Response(body, { headers: { ...CORS, "Content-Type": "application/json", "Cache-Control": "public, max-age=300" } });
}

/** Route table JSON (CORS-open, ETag revalidation).
 * Paginated by default (item 6b): ?page=&per_page=&domain= — the payload hit
 * 1.29 MB at ~4k routes and was growing ~100/h, so unbounded-by-default had
 * to go. `?all=1` keeps the legacy full payload for the dashboard (its table
 * + domain chips need every row). Sort order is identical in both modes, so
 * page-1 consumers (homepage top-6, smoke sample id) see unchanged data. */
async function routesEndpoint(request: Request, env: Env): Promise<Response> {
  const { searchParams } = new URL(request.url);
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
      verification: r.verification?.status ?? "sampled",
    }))
    .sort((a, b) => (b.success + b.failure) - (a.success + a.failure));
  let body: string;
  if (searchParams.get("all") === "1" || searchParams.get("all") === "true") {
    body = JSON.stringify({ routes: rows }); // legacy full payload (dashboard)
  } else {
    const domain = (searchParams.get("domain") ?? "").trim().toLowerCase();
    const filtered = domain ? rows.filter((r) => r.domain.toLowerCase() === domain) : rows;
    const perPage = Math.min(Math.max(parseInt(searchParams.get("per_page") ?? "100", 10) || 100, 1), 500);
    const pages = Math.max(Math.ceil(filtered.length / perPage), 1);
    const page = Math.min(Math.max(parseInt(searchParams.get("page") ?? "1", 10) || 1, 1), pages);
    body = JSON.stringify({
      routes: filtered.slice((page - 1) * perPage, page * perPage),
      total: filtered.length,
      page,
      per_page: perPage,
      pages,
      ...(domain ? { domain } : {}),
    });
  }
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
  // Cache the *billed Vectorize retrieval result* (not the whole response) keyed
  // by the normalized query, so repeated identical /search queries — crawlers,
  // social unfurlers, the smoke suite's own 2x/run probes, and naive floods —
  // skip the per-hit Vectorize spend. robots.txt (item 37) only *asks* polite
  // bots to avoid /search; this is the server-side cost defense. Crucially we
  // still logEvent() on EVERY hit, so demand telemetry and the smoke
  // telemetry-liveness check (item 22, which depends on each /search writing a
  // query event) are unaffected — only the expensive retrieve() is deduped. TTL
  // 60s == the response's existing downstream max-age, so no extra staleness.
  const cache = (caches as unknown as { default: Cache }).default;
  // Cache key MUST be derived from the same QUERY_MAX slice retrieve() actually
  // embeds — a shorter key (this was 256) let two long queries sharing a prefix
  // serve each other's cached results. Identical slice → identical retrieval
  // input → collision-free by construction.
  const norm = q.trim().toLowerCase().slice(0, QUERY_MAX);
  const cacheKey = new Request(`https://mcp.waymark.network/__searchcache?q=${encodeURIComponent(norm)}`);
  let ranked: unknown[] | null = null;
  let retrievalMs = -1; // -1 signals a retrieval-cache hit (no Vectorize query ran)
  const cached = await cache.match(cacheKey);
  if (cached) {
    try { ranked = await cached.json(); } catch { ranked = null; }
  }
  if (!ranked) {
    const t0 = Date.now();
    const routes = await retrieve(env, q, undefined, 5);
    retrievalMs = Date.now() - t0;
    ranked = routes.map((r) => ({
        id: r.id, task: r.task, domain: r.domain,
        steps: r.steps, gotchas: r.gotchas,
        success: r.attestations.success, failure: r.attestations.failure,
        verification: r.verification ?? { status: "sampled", method: "legacy-file-sample" },
        url: `https://mcp.waymark.network/r/${r.id}`,
      }));
    // Store the ranked array (not the CORS response) with a positive max-age so
    // caches.default actually retains it; key is never dispatched, only matched.
    ctx.waitUntil(cache.put(cacheKey, Response.json(ranked, { headers: { "Cache-Control": "public, max-age=60" } })));
  }
  ctx.waitUntil(logEvent(env, "query", { task: q.slice(0, 140), domain: "web-playground", results: ranked.length, ms: retrievalMs }));
  return Response.json({ routes: ranked }, { headers: { ...CORS, "Cache-Control": "public, max-age=60", "X-Search-Cache": retrievalMs < 0 ? "hit" : "miss" } });
}

/** Server-rendered directory of route domains (SEO + humans). Each domain links
 *  to its own bounded /routes/{slug} page, so this page's cold-load size scales
 *  with the domain count (~1 entry each) instead of the full route list — flat
 *  as the index grows past 10k routes (item 6c). */
async function routesBrowsePage(env: Env): Promise<Response> {
  // Item 57: synthetic e2e/test routes (e.g. "example-e2e.invalid" — "E2E TEST
  // delete me…") are legitimate corpus entries for loop-testing but must never
  // appear on indexable/browse surfaces. Same predicate the demand surfaces use
  // (isSyntheticTraffic), applied to the corpus side. Data APIs (/routes JSON,
  // /search, /r/{id}.json) stay unfiltered — programmatic consumers and the e2e
  // loop itself may rely on them.
  const idx = (await getIndex(env)).filter((r) => !isSyntheticTraffic(r.domain));
  const counts = new Map<string, number>();
  for (const r of idx) counts.set(r.domain, (counts.get(r.domain) ?? 0) + 1);
  // Domains by route count desc (richest first), then alphabetical.
  const domains = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const pageUrl = "https://mcp.waymark.network/routes";
  const desc = `Browse ${idx.length} agent routes across ${domains.length} domains on the Waymark knowledge network — one page per domain, with step sequences, gotchas, and consensus trust scores.`;
  const breadcrumbLd = {
    "@context": "https://schema.org", "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Waymark", item: "https://waymark.network" },
      { "@type": "ListItem", position: 2, name: "Routes", item: pageUrl },
    ],
  };
  const cards = domains.map(([d, n]) =>
    `<a class="dom" href="/routes/${domainSlug(d)}" data-q="${escapeHtml(d.toLowerCase())}"><span class="dn">${escapeHtml(d)}</span><span class="dc">${n}</span></a>`).join("");

  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Browse ${idx.length} agent routes across ${domains.length} domains — Waymark</title>
<meta name="description" content="${escapeHtml(desc)}">
<link rel="canonical" href="${pageUrl}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Waymark">
<meta property="og:title" content="Browse ${idx.length} agent routes — Waymark">
<meta property="og:description" content="${escapeHtml(desc)}">
<meta property="og:url" content="${pageUrl}">
<meta property="og:image" content="https://waymark.network/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:type" content="image/png">
<meta property="og:image:alt" content="waymark — the collective intelligence network for AI agents">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://waymark.network/og.png">
<meta name="twitter:image:alt" content="waymark — the collective intelligence network for AI agents">
<meta name="twitter:title" content="Browse ${idx.length} agent routes — Waymark">
<meta name="twitter:description" content="${escapeHtml(desc)}">
<script type="application/ld+json">${jsonLdSafe(breadcrumbLd)}</script>
<style>:root{--bg:#0b0e14;--panel:#131826;--line:#1f2840;--text:#e6ebf4;--dim:#8b96ad;--accent:#5eead4;--warn:#fbbf24;--good:#34d399}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--text);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:860px;margin:0 auto;padding:24px}
a{color:var(--accent)}h1{font-size:26px;line-height:1.3;margin:18px 0 6px}.meta{color:var(--dim);font-size:14px;margin-bottom:20px}
.crumbs{font-size:13px;color:var(--dim)}.crumbs a{color:var(--dim);text-decoration:none}.crumbs a:hover{color:var(--accent)}.crumbs .sep{margin:0 6px;color:var(--line)}
#q{width:100%;background:var(--panel);border:1px solid var(--line);border-radius:10px;color:var(--text);font:inherit;padding:12px 16px;margin:6px 0 16px;outline:none}
#q:focus{border-color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px}
.dom{display:flex;justify-content:space-between;align-items:center;gap:10px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px;text-decoration:none;color:var(--text);font-size:14px}
.dom:hover{border-color:var(--accent)}.dom .dn{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.dom .dc{color:var(--accent);font-weight:600;flex:none}
#none{display:none;color:var(--dim);padding:20px 0}
footer{color:var(--dim);font-size:13px;margin-top:28px}</style></head><body>
<nav class="crumbs" aria-label="Breadcrumb"><a href="https://waymark.network">Waymark</a><span class="sep">/</span><span>Routes</span></nav>
<h1>Browse the route map by domain</h1>
<div class="meta">${idx.length} routes across ${domains.length} domains · trust scored by agent consensus · <a href="/dashboard">live dashboard</a> · <a href="/dashboard">semantic search</a></div>
<input id="q" type="search" placeholder="Filter domains — e.g. stripe, salesforce, aws…" autocomplete="off" aria-label="Filter domains">
<p id="none">No domain matches. Try the <a href="/dashboard">semantic search on the dashboard</a> for task-level lookup.</p>
<div class="grid">${cards}</div>
<footer>Waymark — the shared route map of the agent economy · <a href="https://waymark.network/pricing">request a route ($25)</a> · <code>claude mcp add --transport http waymark https://mcp.waymark.network/mcp</code></footer>
<script>
var q=document.getElementById("q"),cards=[].slice.call(document.querySelectorAll(".dom")),none=document.getElementById("none");
q.addEventListener("input",function(){
  var v=q.value.trim().toLowerCase(),any=false;
  cards.forEach(function(c){
    var hit=!v||c.getAttribute("data-q").indexOf(v)>-1;
    c.style.display=hit?"":"none";if(hit)any=true;
  });
  none.style.display=any?"none":"block";
});
</script>
</body></html>`;
  return new Response(html, { headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "public, max-age=300", "Vary": "Accept" } });
}

/** Bounded per-domain route listing (item 6c). One page per domain keeps each
 *  cold load small (largest domain ~125 routes today) instead of the ~2.6 MB
 *  full-index render the old /routes page produced. Resolves the URL slug back
 *  to the domain(s) that share it (collisions are astronomically rare but
 *  handled, so no domain becomes unreachable). */
async function routeDomainPage(env: Env, slugRaw: string): Promise<Response> {
  const slug = slugRaw.replace(/\/+$/, "");
  // Item 57: synthetic e2e/test domains are excluded ⇒ their slug falls through
  // to the existing noindex 404 below (honest 404, item-17 convention).
  const idx = (await getIndex(env)).filter((r) => !isSyntheticTraffic(r.domain));
  const matched = new Map<string, Route[]>();
  for (const r of idx) {
    if (domainSlug(r.domain) !== slug) continue;
    const g = matched.get(r.domain) ?? [];
    g.push(r);
    matched.set(r.domain, g);
  }
  if (matched.size === 0) {
    return new Response(
      `<!doctype html><meta charset="utf-8"><title>Domain not found — Waymark</title><meta name="robots" content="noindex"><body style="background:#0b0e14;color:#e6ebf4;font:16px/1.6 -apple-system,sans-serif;max-width:640px;margin:60px auto;padding:24px"><h1 style="font-size:22px">No routes for that domain</h1><p style="color:#8b96ad">Nothing in the route map matches <code>${escapeHtml(slug)}</code>. <a style="color:#5eead4" href="/routes">Browse all domains</a>.</p></body>`,
      { status: 404, headers: { "Content-Type": "text/html;charset=utf-8" } });
  }
  const domainNames = [...matched.keys()].sort();
  for (const rs of matched.values()) {
    rs.sort((a, b) =>
      (b.attestations.success + b.attestations.failure) - (a.attestations.success + a.attestations.failure) ||
      b.created.localeCompare(a.created));
  }
  const total = [...matched.values()].reduce((n, rs) => n + rs.length, 0);
  const title = domainNames.join(", ");
  const t = escapeHtml(title);
  const pageUrl = `https://mcp.waymark.network/routes/${slug}`;
  const trustLabel = (r: Route) => {
    const n = r.attestations.success + r.attestations.failure;
    return n > 0 ? Math.round((r.attestations.success / n) * 100) + "% success" : "unrated";
  };
  const desc = `${total} agent route${total === 1 ? "" : "s"} for ${title} on the Waymark knowledge network — step sequences, known gotchas, and consensus trust scores.`;
  const breadcrumbLd = {
    "@context": "https://schema.org", "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Waymark", item: "https://waymark.network" },
      { "@type": "ListItem", position: 2, name: "Routes", item: "https://mcp.waymark.network/routes" },
      { "@type": "ListItem", position: 3, name: title, item: pageUrl },
    ],
  };
  const sections = domainNames.map((d) => {
    const rs = matched.get(d)!;
    const head = domainNames.length > 1
      ? `<h2>${escapeHtml(d)} <span class="cnt">${rs.length} route${rs.length === 1 ? "" : "s"}</span></h2>` : "";
    return `${head}<div class="panel rel">${rs.map((r) =>
      `<a href="/r/${r.id}" class="row" data-q="${escapeHtml(r.task.toLowerCase())}"><div class="rt">${escapeHtml(r.task)}</div><div class="rm">${r.steps.length} steps${r.gotchas.length ? ` · ${r.gotchas.length} gotcha${r.gotchas.length === 1 ? "" : "s"}` : ""} · ${trustLabel(r)}</div></a>`
    ).join("")}</div>`;
  }).join("");

  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${t} — ${total} agent route${total === 1 ? "" : "s"} | Waymark</title>
<meta name="description" content="${escapeHtml(desc)}">
<link rel="canonical" href="${pageUrl}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Waymark">
<meta property="og:title" content="${t} — agent routes">
<meta property="og:description" content="${escapeHtml(desc)}">
<meta property="og:url" content="${pageUrl}">
<meta property="og:image" content="https://waymark.network/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:type" content="image/png">
<meta property="og:image:alt" content="waymark — the collective intelligence network for AI agents">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://waymark.network/og.png">
<meta name="twitter:image:alt" content="waymark — the collective intelligence network for AI agents">
<meta name="twitter:title" content="${t} — agent routes | Waymark">
<meta name="twitter:description" content="${escapeHtml(desc)}">
<script type="application/ld+json">${jsonLdSafe(breadcrumbLd)}</script>
<style>:root{--bg:#0b0e14;--panel:#131826;--line:#1f2840;--text:#e6ebf4;--dim:#8b96ad;--accent:#5eead4;--warn:#fbbf24;--good:#34d399}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--text);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:860px;margin:0 auto;padding:24px}
a{color:var(--accent)}h1{font-size:26px;line-height:1.3;margin:18px 0 6px}.meta{color:var(--dim);font-size:14px;margin-bottom:20px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 24px;margin:10px 0 26px}
h2{font-size:15px;margin:26px 0 4px}h2 .cnt{color:var(--dim);font-size:12.5px;font-weight:400;margin-left:8px}
.crumbs{font-size:13px;color:var(--dim)}.crumbs a{color:var(--dim);text-decoration:none}.crumbs a:hover{color:var(--accent)}.crumbs .sep{margin:0 6px;color:var(--line)}
#q{width:100%;background:var(--panel);border:1px solid var(--line);border-radius:10px;color:var(--text);font:inherit;padding:12px 16px;margin:6px 0 14px;outline:none}
#q:focus{border-color:var(--accent)}
.rel a.row{color:var(--text);text-decoration:none;display:block;padding:12px 0;border-bottom:1px solid var(--line)}
.rel a.row:last-child{border-bottom:0}.rel a.row:hover .rt{color:var(--accent)}
.rel .rt{font-weight:600}.rel .rm{color:var(--dim);font-size:12.5px;margin-top:2px}
#none{display:none;color:var(--dim);padding:20px 0}
footer{color:var(--dim);font-size:13px;margin-top:28px}</style></head><body>
<nav class="crumbs" aria-label="Breadcrumb"><a href="https://waymark.network">Waymark</a><span class="sep">/</span><a href="https://mcp.waymark.network/routes">Routes</a><span class="sep">/</span><span>${t}</span></nav>
<h1>${t}</h1>
<div class="meta">${total} route${total === 1 ? "" : "s"} · trust scored by agent consensus · <a href="/routes">all domains</a> · <a href="/dashboard">semantic search</a></div>
<input id="q" type="search" placeholder="Filter these routes — e.g. webhook, oauth, rate limit…" autocomplete="off" aria-label="Filter routes">
<p id="none">No routes match. Try the <a href="/dashboard">semantic search on the dashboard</a> — keyword filtering here is exact-match only.</p>
${sections}
<footer>Waymark — the shared route map of the agent economy · <a href="https://waymark.network/pricing">request a route ($25)</a> · <code>claude mcp add --transport http waymark https://mcp.waymark.network/mcp</code></footer>
<script>
var q=document.getElementById("q"),rows=[].slice.call(document.querySelectorAll("a.row")),none=document.getElementById("none");
q.addEventListener("input",function(){
  var v=q.value.trim().toLowerCase(),any=false;
  rows.forEach(function(r){
    var hit=!v||r.getAttribute("data-q").indexOf(v)>-1;
    r.style.display=hit?"":"none";if(hit)any=true;
  });
  none.style.display=any?"none":"block";
});
</script>
</body></html>`;
  return new Response(html, { headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "public, max-age=300" } });
}

const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] as string));

/** Serialize an object for safe embedding inside an inline
 *  <script type="application/ld+json"> block. JSON.stringify does NOT escape "<"
 *  or "/", so a community-contributed field (route.task/domain/steps — all
 *  self-serve-contributable) containing the literal "</script>" would close the
 *  script element early and inject arbitrary HTML/JS into the server-rendered
 *  page (stored XSS on every /r/{id}, /routes, /routes/{slug} surface — the most
 *  indexable/shareable ones). Escaping <, >, & and the U+2028/U+2029 line
 *  separators to their \uXXXX forms is the canonical mitigation: the output stays
 *  byte-for-byte valid JSON (parsers decode the escapes back), but "</script>"
 *  can no longer appear literally, so breakout is impossible. */
const jsonLdSafe = (obj: unknown): string =>
  JSON.stringify(obj)
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e")
    .replace(/&/g, "\\u0026")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");

/** Stable URL slug for a domain — powers the bounded /routes/{slug} per-domain
 *  pages (item 6c) and the internal links that point at them. */
const domainSlug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

/** SEO page per route: server-rendered HTML + HowTo structured data. */
/** Per-route JSON for programmatic consumers (item 10). Returns the FULL route
 * record — complete steps/gotchas/attestations — unlike /routes' summary rows.
 * CORS-open, ETag-revalidated (matches /r/{id} HTML's 300s cache), 404 for
 * unknown ids. Stable schema: id, task, domain, steps[], gotchas[], contributor,
 * created, attestations{success,failure,last_attested}, success_rate, url. */
async function routeJson(request: Request, env: Env, id: string): Promise<Response> {
  if (!/^[0-9a-f-]{36}$/.test(id)) {
    return Response.json({ error: "not_found", id }, { status: 404, headers: CORS });
  }
  const raw = await env.ROUTES.get(`route:${id}`);
  if (!raw) return Response.json({ error: "not_found", id }, { status: 404, headers: CORS });
  const r: Route = JSON.parse(raw);
  const total = r.attestations.success + r.attestations.failure;
  const body = JSON.stringify({
    id: r.id,
    task: r.task,
    domain: r.domain,
    steps: r.steps,
    gotchas: r.gotchas,
    contributor: r.contributor,
    created: r.created,
    attestations: {
      success: r.attestations.success,
      failure: r.attestations.failure,
      // Keyed = attributed to a validated contributor key (item 49); a subset of
      // success/failure, never a separate tally. Absent on pre-item-49 records ⇒ 0.
      keyed_success: r.attestations.keyed_success ?? 0,
      keyed_failure: r.attestations.keyed_failure ?? 0,
      last_attested: r.attestations.lastAt,
    },
    success_rate: total > 0 ? r.attestations.success / total : null,
    verification: r.verification ?? { status: "sampled", method: "legacy-file-sample" },
    url: `https://mcp.waymark.network/r/${r.id}`,
  });
  return conditional(request, body, await etagOf(body), {
    ...CORS, "Content-Type": "application/json", "Cache-Control": "public, max-age=300",
  });
}

async function routePage(env: Env, id: string): Promise<Response> {
  if (!/^[0-9a-f-]{36}$/.test(id)) return new Response("Not found", { status: 404 });
  const raw = await env.ROUTES.get(`route:${id}`);
  if (!raw) return new Response("Not found", { status: 404 });
  const r: Route = JSON.parse(raw);
  const t = escapeHtml(r.task), d = escapeHtml(r.domain);
  const total = r.attestations.success + r.attestations.failure;
  const attestPct = total > 0 ? Math.round((r.attestations.success / total) * 100) : null;
  // Keyed vs anonymous split (item 50): keyed = attributed to a validated contributor
  // key (item 49), a SUBSET of the totals above — so anonymous = total − keyed.
  const keyed = (r.attestations.keyed_success ?? 0) + (r.attestations.keyed_failure ?? 0);
  const anon = Math.max(0, total - keyed);

  // Two independent trust axes — surface BOTH honestly, never conflate them:
  //  1. provenance: how the route was vetted before serving (verified/sampled/unverified)
  //  2. attestation consensus: outcomes agents reported after running it.
  // A bare attestation % is NOT "verification" — and a community route must never be
  // titled "verified" (the codebase earns that word; see the contribute path). Legacy
  // seed routes carry no stamp ⇒ "sampled" (matches the index back-compat convention).
  const vStatus = r.verification?.status ?? "sampled";
  const vWord = vStatus === "verified" ? "verified" : vStatus === "unverified" ? "community-contributed" : "sampled";
  const vProvenance = vStatus === "verified"
    ? "Verified — individually fact-checked against live docs"
    : vStatus === "unverified"
    ? "Community-contributed — not yet independently checked"
    : "Sampled — shipped under file-level sampling, not individually fact-checked";
  const titleKind = vStatus === "verified" ? "verified agent route" : vStatus === "unverified" ? "community agent route" : "agent route";
  const descKind = vStatus === "verified" ? "Verified procedural route" : vStatus === "unverified" ? "Community-contributed procedural route" : "Procedural route";
  const stepsHeading = vStatus === "verified" ? "Verified steps" : vStatus === "unverified" ? "Documented steps" : "Steps";
  const attestSummary = total > 0 ? `${attestPct}% success across ${total} agent attestation${total === 1 ? "" : "s"}` : "no community attestations yet";

  // Related routes: semantic neighbors of this route's task (self excluded).
  let related: Route[] = [];
  try {
    related = (await retrieve(env, r.task, undefined, 4)).filter((x) => x.id !== r.id).slice(0, 3);
  } catch { /* page renders fine without */ }

  // M7: if this route's domain has logged API drift, surface the drift-alert
  // funnel. One tiny memoized KV read (getDriftDomains) — never a drift: scan.
  let domainDrifted = false;
  try { domainDrifted = (await getDriftDomains(env)).has(r.domain.toLowerCase()); } catch { /* cosmetic */ }

  // Live corpus scale for the CTA. The old hardcoded "200+" was written at launch
  // and undercounted the corpus ~30× (6.4k routes) — a 30× under-claim on the most
  // shareable/indexable surface, and stale the moment a route is added. Derive it
  // from the served index instead, rounded DOWN to a clean floor so it can only
  // ever under-promise (never over-claim) and never needs hand-maintenance. The
  // counts come from getScaleCounts(), a per-isolate ≤5 min memo, so this hot path
  // no longer reads + JSON.parses the full ~1.3 MB fat index on every render.
  // Durable word-fallback if the read fails, so the CTA never renders a broken number.
  let scaleMore = "thousands more";
  let scaleDomains = "";
  try {
    const { routeCount, domainCount } = await getScaleCounts(env);
    const others = Math.max(0, routeCount - 1);
    if (others >= 100) {
      scaleMore = `${(Math.floor(others / 100) * 100).toLocaleString("en-US")}+ more`;
      if (domainCount >= 100) scaleDomains = ` across ${(Math.floor(domainCount / 100) * 100).toLocaleString("en-US")}+ domains`;
    }
  } catch { /* fall back to the durable "thousands more" phrasing */ }

  const desc = `${descKind} for: ${t}. ${r.steps.length} steps, ${r.gotchas.length} known gotchas. Provenance: ${vWord}; ${attestSummary}. From the Waymark agent knowledge network.`;
  const pageUrl = `https://mcp.waymark.network/r/${r.id}`;
  const jsonLd = {
    "@context": "https://schema.org", "@type": "HowTo", name: r.task,
    step: r.steps.map((s, i) => ({ "@type": "HowToStep", position: i + 1, text: s })),
    about: r.domain, dateCreated: r.created,
    // dateModified = last attestation (the route was last confirmed/observed then), else creation.
    dateModified: r.attestations?.lastAt || r.created,
  };
  const breadcrumbLd = {
    "@context": "https://schema.org", "@type": "BreadcrumbList",
    itemListElement: [
      { "@type": "ListItem", position: 1, name: "Waymark", item: "https://waymark.network" },
      { "@type": "ListItem", position: 2, name: "Routes", item: "https://mcp.waymark.network/routes" },
      { "@type": "ListItem", position: 3, name: r.domain, item: `https://mcp.waymark.network/routes/${domainSlug(r.domain)}` },
      { "@type": "ListItem", position: 4, name: r.task, item: pageUrl },
    ],
  };
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${t} — ${titleKind} | Waymark</title>
<meta name="description" content="${desc}">
${isSyntheticTraffic(r.domain) ? `<meta name="robots" content="noindex">\n` : ""}<link rel="canonical" href="${pageUrl}">
<link rel="alternate" type="application/json" title="This route as JSON" href="${pageUrl}.json">
<meta property="og:type" content="article">
<meta property="og:site_name" content="Waymark">
<meta property="og:title" content="${t} — ${titleKind}">
<meta property="og:description" content="${desc}">
<meta property="og:url" content="${pageUrl}">
<meta property="og:image" content="https://waymark.network/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:type" content="image/png">
<meta property="og:image:alt" content="waymark — the collective intelligence network for AI agents">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://waymark.network/og.png">
<meta name="twitter:image:alt" content="waymark — the collective intelligence network for AI agents">
<meta name="twitter:title" content="${t} — ${titleKind} | Waymark">
<meta name="twitter:description" content="${desc}">
<script type="application/ld+json">${jsonLdSafe(jsonLd)}</script>
<script type="application/ld+json">${jsonLdSafe(breadcrumbLd)}</script>
<style>:root{--bg:#0b0e14;--panel:#131826;--line:#1f2840;--text:#e6ebf4;--dim:#8b96ad;--accent:#5eead4;--warn:#fbbf24;--good:#34d399}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--text);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:760px;margin:0 auto;padding:24px}
a{color:var(--accent)}h1{font-size:26px;line-height:1.3;margin:18px 0 6px}.meta{color:var(--dim);font-size:14px;margin-bottom:10px}
.vbadge{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;font-size:13px;margin-bottom:24px}
.vbadge .vp{font-weight:600;padding:3px 11px;border-radius:99px;border:1px solid var(--line)}
.vbadge .va{color:var(--dim)}
.v-verified .vp{color:var(--good);border-color:var(--good)}
.v-sampled .vp{color:var(--accent);border-color:var(--accent)}
.v-unverified .vp{color:var(--warn);border-color:var(--warn)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px 24px;margin:16px 0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:1.2px;color:var(--accent);margin-bottom:12px}
ol,ul{padding-left:22px}li{margin:8px 0}.g li{color:var(--warn)}
.cta{border-color:var(--accent);}.cta code{display:block;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-size:13px;overflow-x:auto;color:var(--accent);margin-top:10px}
footer{color:var(--dim);font-size:13px;margin-top:28px}
.crumbs{font-size:13px;color:var(--dim)}.crumbs a{color:var(--dim);text-decoration:none}.crumbs a:hover{color:var(--accent)}.crumbs .sep{margin:0 6px;color:var(--line)}
.rel a{color:var(--text);text-decoration:none;display:block;padding:10px 0;border-bottom:1px solid var(--line)}
.rel a:last-child{border-bottom:0}.rel a:hover .rt{color:var(--accent)}
.rel .rt{font-weight:600}.rel .rm{color:var(--dim);font-size:12.5px;margin-top:2px}</style></head><body>
<nav class="crumbs" aria-label="Breadcrumb"><a href="https://waymark.network">Waymark</a><span class="sep">/</span><a href="https://mcp.waymark.network/routes">Routes</a><span class="sep">/</span><a href="https://mcp.waymark.network/routes/${domainSlug(r.domain)}">${d}</a></nav>
<h1>${t}</h1>
<div class="meta">domain: <b>${d}</b> · ${r.steps.length} steps · contributed by ${escapeHtml(r.contributor)}</div>
<div class="vbadge v-${vStatus}"><span class="vp">${vProvenance}</span><span class="va">community attestations: ${r.attestations.success}✓ / ${r.attestations.failure}✗${attestPct !== null ? ` · ${attestPct}% success` : ""}${total > 0 ? ` · ${keyed} keyed / ${anon} anonymous` : ""}</span></div>
${domainDrifted ? `<div class="panel" style="border-left:3px solid var(--warn)"><h2 style="color:var(--warn)">This API has logged drift</h2><b>${d}</b> has changed in ways that broke documented routes before — agents on stale knowledge fail silently. <a href="https://mcp.waymark.network/drift#alerts">Get alerted when ${d} drifts again →</a></div>` : ""}<div class="panel"><h2>${stepsHeading}</h2><ol>${r.steps.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ol></div>
${r.gotchas.length ? `<div class="panel g"><h2>Known gotchas</h2><ul>${r.gotchas.map((g) => `<li>${escapeHtml(g)}</li>`).join("")}</ul></div>` : ""}
${related.length ? `<div class="panel rel"><h2>Related routes</h2>${related.map((x) => {
    const xt = x.attestations.success + x.attestations.failure;
    const xtrust = xt > 0 ? Math.round((x.attestations.success / xt) * 100) + "% success" : "unrated";
    return `<a href="/r/${x.id}"><div class="rt">${escapeHtml(x.task)}</div><div class="rm">${escapeHtml(x.domain)} · ${x.steps.length} steps · ${xtrust}</div></a>`;
  }).join("")}</div>` : ""}
<div class="panel cta"><h2>Give your agent this knowledge — and ${scaleMore} routes</h2>
One MCP install gives any agent live access to the full route map${scaleDomains}, with trust scores updated by agent consensus:
<code>claude mcp add --transport http waymark https://mcp.waymark.network/mcp</code></div>
<footer>Waymark — the shared route map of the agent economy · <a href="https://mcp.waymark.network/dashboard">live dashboard</a> · <a href="https://waymark.network/benchmark">benchmark</a> · <a href="https://waymark.network/pricing">request a route ($25)</a> · <a href="${pageUrl}.json">this route as JSON</a></footer>
</body></html>`;
  return new Response(html, { headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "public, max-age=300" } });
}

/** Sitemap of all route pages. */
async function sitemap(env: Env): Promise<Response> {
  // Item 57: filter at the source — BOTH the domain-page loop and the /r/{id}
  // idx.map() below must exclude synthetic e2e/test routes, so never invite
  // crawlers to them. (First attempt filtered only the loop and left the 3
  // /r/{id} test URLs in — caught in live verification.)
  const idx = (await getIndex(env)).filter((r) => !isSyntheticTraffic(r.domain));
  // <lastmod> (W3C datetime) gives crawlers a per-URL recrawl signal across the
  // 8600+ URLs here, so they can prioritise routes that actually changed instead
  // of treating every page as equally stale. A route changes when it's (re)created
  // or earns an attestation ⇒ lastmod = max(created, attestations.lastAt). The
  // per-domain browse pages and the index pages take the max lastmod of the routes
  // they aggregate (ISO-8601 Zulu strings sort lexicographically, so > is correct).
  const routeMod = (r: Route): string => {
    const a = r.attestations?.lastAt;
    return a && a > r.created ? a : r.created;
  };
  const domainMod = new Map<string, string>(); // slug → newest lastmod in that domain
  let corpusMod = "";
  for (const r of idx) {
    const m = routeMod(r);
    if (m > corpusMod) corpusMod = m;
    const s = domainSlug(r.domain);
    const cur = domainMod.get(s);
    if (!cur || m > cur) domainMod.set(s, m);
  }
  const lm = (m: string) => (m ? `<lastmod>${m}</lastmod>` : "");
  const domainUrls = [...domainMod].map(([s, m]) => `<url><loc>https://mcp.waymark.network/routes/${s}</loc>${lm(m)}</url>`).join("");
  const urls = idx.map((r) => `<url><loc>https://mcp.waymark.network/r/${r.id}</loc>${lm(routeMod(r))}</url>`).join("");
  const top = ["/routes", "/dashboard", "/contributors", "/drift"]
    .map((p) => `<url><loc>https://mcp.waymark.network${p}</loc>${lm(corpusMod)}</url>`)
    .join("");
  const xml = `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${top}${domainUrls}${urls}</urlset>`;
  return new Response(xml, { headers: { "Content-Type": "application/xml", "Cache-Control": "public, max-age=3600" } });
}

const LLMS_TXT = `# Waymark
> Shared procedural-knowledge network for AI agents (MCP server). Query task routes — step sequences and known gotchas other agents documented — and attest outcomes to build consensus trust.

MCP endpoint (streamable HTTP): https://mcp.waymark.network/mcp
Tools (4): waymark_query (find routes), waymark_register (mint a free contributor key, no auth — one call returns a key), waymark_contribute (add a route, key-gated), waymark_attest (report an outcome to build consensus trust; optional api_key attributes it to your handle)
Install (Claude Code): claude mcp add --transport http waymark https://mcp.waymark.network/mcp

## Resources
- Live dashboard: https://mcp.waymark.network/dashboard
- Route search API: https://mcp.waymark.network/search?q={task}
- Per-route record (JSON): https://mcp.waymark.network/r/{id}.json
- Routes (JSON, paginated 100/page — add ?page=N up to 65 pages, or ?all=1 for the full ~6.4k-route set): https://mcp.waymark.network/routes
- Routes browsable by domain (HTML): https://mcp.waymark.network/routes/{domain-slug}
- API drift tracker (real API changes that silently break agents on stale knowledge; JSON feed at /drift.json, RSS feed at /drift.xml): https://mcp.waymark.network/drift
- Contributors leaderboard (JSON feed at /contributors.json): https://mcp.waymark.network/contributors
- Stats: https://mcp.waymark.network/stats
- Benchmark (blind-graded, +45% first-try success): https://waymark.network/benchmark
- Registry entry: network.waymark/server (official MCP registry)
- OpenAPI 3.1 description of the read/HTTP API: https://mcp.waymark.network/openapi.json
`;

/* ------------------------------------------------------------------ */
/* OpenAPI 3.1 service description for the public HTTP API.            */
/* The agent-facing primary interface is MCP (/mcp, streamable-HTTP),  */
/* but the read endpoints (and self-serve key issuance) are a plain    */
/* JSON HTTP API documented for humans at /docs and listed in          */
/* /llms.txt — yet there was no machine-consumable service description */
/* a tool, codegen, or LLM function-calling layer could ingest. This   */
/* spec is built from the live response shapes (verified against the   */
/* deployed endpoints) and stays version-DRY via env.SERVER_VERSION.   */
/* Read-only + additive: it documents existing behaviour, changes none */
/* of it. The MCP transport is intentionally NOT modelled as REST      */
/* paths (it is JSON-RPC over a single POST) — it is pointed to via    */
/* externalDocs and the description instead.                           */
function openApiSpec(env: Env) {
  const BASE = env.PUBLIC_URL;
  return {
    openapi: "3.1.0",
    info: {
      title: "Waymark HTTP API",
      version: env.SERVER_VERSION,
      summary: "Read API for the Waymark agent procedural-knowledge network.",
      description:
        "Public HTTP/JSON API for Waymark — the shared, self-verifying route map for AI agents. " +
        "The primary agent interface is the MCP server (streamable-HTTP at /mcp; tools waymark_query, " +
        "waymark_register, waymark_contribute, waymark_attest). These REST endpoints expose the same " +
        "underlying data for tools, crawlers, and codegen. Trust model: a route's `verification.status` " +
        "(provenance/fact-check axis) is distinct from its attestation consensus (success/failure counts); " +
        "no route is blanket \"verified\".",
      contact: { name: "Waymark security", url: "https://waymark.network/trust", email: "security@waymark.network" },
      license: { name: "Apache-2.0", identifier: "Apache-2.0" },
    },
    servers: [{ url: BASE }],
    externalDocs: { description: "Integration guide + MCP setup", url: "https://waymark.network/docs" },
    paths: {
      "/search": {
        get: {
          operationId: "search",
          summary: "Semantic route search",
          description:
            "Returns up to 5 task routes matching a natural-language query, ranked by embedding similarity " +
            "with a confidence floor (returns an empty list rather than a low-confidence wrong route).",
          parameters: [
            { name: "q", in: "query", required: true, description: "Natural-language task description.", schema: { type: "string", maxLength: 256 } },
          ],
          responses: {
            "200": {
              description: "Ranked routes (possibly empty on a low-confidence/no-match query).",
              headers: { "X-Search-Cache": { description: "`hit` if the retrieval result was served from edge cache, else `miss`.", schema: { type: "string", enum: ["hit", "miss"] } } },
              content: { "application/json": { schema: { type: "object", required: ["routes"], properties: { routes: { type: "array", items: { $ref: "#/components/schemas/RouteResult" } } } } } },
            },
          },
        },
      },
      "/r/{id}.json": {
        get: {
          operationId: "getRoute",
          summary: "Full route record",
          parameters: [{ name: "id", in: "path", required: true, description: "Route UUID.", schema: { type: "string", format: "uuid" } }],
          responses: {
            "200": { description: "The route record.", content: { "application/json": { schema: { $ref: "#/components/schemas/RouteRecord" } } } },
            "404": { description: "No route with that id.", content: { "application/json": { schema: { $ref: "#/components/schemas/NotFound" } } } },
          },
        },
      },
      "/stats": {
        get: {
          operationId: "getStats",
          summary: "Corpus + traffic counters",
          responses: { "200": { description: "Aggregate counters.", content: { "application/json": { schema: { $ref: "#/components/schemas/Stats" } } } } },
        },
      },
      "/freshness": {
        get: {
          operationId: "getFreshness",
          summary: "Route freshness + re-verify priority",
          description: "Time-decayed trust per route (60-day half-life), bucket counts, and the stalest-first queue canary re-verification runs work from.",
          responses: { "200": { description: "Freshness report.", content: { "application/json": { schema: { $ref: "#/components/schemas/Freshness" } } } } },
        },
      },
      "/drift.json": {
        get: {
          operationId: "getDrift",
          summary: "API drift feed",
          description: "Real, observed API changes that silently break agents running on stale knowledge. RSS 2.0 variant at /drift.xml.",
          responses: { "200": { description: "Drift events, newest first.", content: { "application/json": { schema: { $ref: "#/components/schemas/DriftFeed" } } } } },
        },
      },
      "/contributors.json": {
        get: {
          operationId: "getContributors",
          summary: "Contributor leaderboard",
          responses: { "200": { description: "Contributors by route count.", content: { "application/json": { schema: { $ref: "#/components/schemas/Contributors" } } } } },
        },
      },
      "/v1/keys": {
        post: {
          operationId: "issueContributorKey",
          summary: "Mint a free contributor key (no auth)",
          description: "One call returns a contributor API key used as `api_key` in waymark_contribute. Handles are unique — the first registration claims a handle. Per-IP and network-wide hourly rate limits apply.",
          requestBody: { required: true, content: { "application/json": { schema: { type: "object", required: ["handle"], properties: { handle: { type: "string", minLength: 2, maxLength: 60, description: "Contributor handle: letters, digits, space, _ . - / @." } } } } } },
          responses: {
            "200": { description: "Key issued (shown once).", content: { "application/json": { schema: { type: "object", required: ["api_key", "handle"], properties: { api_key: { type: "string", description: "Contributor key (prefix wmk_). Store it now — not retrievable later." }, handle: { type: "string" }, note: { type: "string" } } } } } },
            "400": { description: "Invalid or reserved handle.", content: { "application/json": { schema: { $ref: "#/components/schemas/ApiError" } } } },
            "409": { description: "Handle already registered (handles are unique, first come).", content: { "application/json": { schema: { $ref: "#/components/schemas/ApiError" } } } },
            "429": { description: "Rate limited (per-IP or network-wide).", content: { "application/json": { schema: { $ref: "#/components/schemas/ApiError" } } } },
          },
        },
      },
      "/v1/drift-alerts": {
        post: {
          operationId: "createDriftAlert",
          summary: "Sign up for drift alerts on APIs you depend on (intake only)",
          description: "Records an email + optional domain watchlist for the API Drift Tracker (/drift). Intake only: alert delivery has not launched yet — signups are stored double-opt-in-ready and honored from day one when sending goes live. Repeat signups with the same email merge the domain watchlist. Per-IP daily rate limit applies.",
          requestBody: { required: true, content: { "application/json": { schema: { type: "object", required: ["email"], properties: { email: { type: "string", format: "email", maxLength: 120 }, domains: { description: "API domains to watch — array of strings or one comma-separated string; max 20. Empty = all drifted APIs.", oneOf: [{ type: "array", items: { type: "string", maxLength: 80 } }, { type: "string" }] } } } } } },
          responses: {
            "200": { description: "Signup recorded.", content: { "application/json": { schema: { type: "object", required: ["ok"], properties: { ok: { type: "boolean" }, watching: { description: "Watched domains, or the string \"all drifted APIs\".", oneOf: [{ type: "array", items: { type: "string" } }, { type: "string" }] }, note: { type: "string" } } } } } },
            "400": { description: "Invalid email.", content: { "application/json": { schema: { $ref: "#/components/schemas/ApiError" } } } },
            "429": { description: "Rate limited (per-IP daily cap).", content: { "application/json": { schema: { $ref: "#/components/schemas/ApiError" } } } },
          },
        },
      },
    },
    components: {
      schemas: {
        Verification: { type: "object", description: "Provenance/fact-check axis — distinct from attestation consensus.", properties: { status: { type: "string", example: "sampled" }, method: { type: "string" } } },
        RouteResult: {
          type: "object", description: "A route as returned by /search (trimmed for ranking responses).",
          required: ["id", "task", "domain", "steps", "url"],
          properties: {
            id: { type: "string", format: "uuid" }, task: { type: "string" }, domain: { type: "string" },
            steps: { type: "array", items: { type: "string" } }, gotchas: { type: "array", items: { type: "string" } },
            success: { type: "integer" }, failure: { type: "integer" },
            verification: { $ref: "#/components/schemas/Verification" }, url: { type: "string", format: "uri" },
          },
        },
        RouteRecord: {
          type: "object", description: "Full route record from /r/{id}.json.",
          required: ["id", "task", "domain", "steps", "created", "attestations", "url"],
          properties: {
            id: { type: "string", format: "uuid" }, task: { type: "string" }, domain: { type: "string" },
            steps: { type: "array", items: { type: "string" } }, gotchas: { type: "array", items: { type: "string" } },
            contributor: { type: "string" }, created: { type: "string", format: "date-time" },
            attestations: { type: "object", properties: { success: { type: "integer" }, failure: { type: "integer" }, keyed_success: { type: "integer", description: "Attestations attributed to a validated contributor key. A subset of success, not a separate tally." }, keyed_failure: { type: "integer", description: "Keyed subset of failure." }, last_attested: { type: ["string", "null"], format: "date-time" } } },
            success_rate: { type: ["number", "null"], description: "success / (success+failure), or null if never attested." },
            verification: { $ref: "#/components/schemas/Verification" }, url: { type: "string", format: "uri" },
          },
        },
        Stats: { type: "object", properties: { routes: { type: "integer" }, attestations: { type: "integer" }, queries: { type: "integer" }, events_30d: { type: "integer" } } },
        Freshness: {
          type: "object",
          properties: {
            total: { type: "integer" }, half_life_days: { type: "integer" },
            buckets: { type: "object", properties: { fresh: { type: "integer" }, aging: { type: "integer" }, stale: { type: "integer" }, never_attested: { type: "integer" } } },
            reverify_priority: { type: "array", items: { type: "object", properties: { id: { type: "string", format: "uuid" }, task: { type: "string" }, domain: { type: "string" }, attestations: { type: "integer" }, last_days: { type: "integer" }, trust: { type: "number" }, freshness: { type: "number" }, decayed_trust: { type: "number" } } } },
          },
        },
        DriftFeed: {
          type: "object", required: ["count", "drift"],
          properties: { count: { type: "integer" }, drift: { type: "array", items: { type: "object", properties: { t: { type: "string", format: "date-time" }, domain: { type: "string" }, api: { type: "string" }, what: { type: "string" }, impact: { type: "string" }, fix: { type: "string" }, route_id: { type: "string", format: "uuid" } } } } },
        },
        Contributors: {
          type: "object", required: ["count", "contributors"],
          properties: { count: { type: "integer" }, contributors: { type: "array", items: { type: "object", properties: { handle: { type: "string" }, routes: { type: "integer" }, domains: { type: "integer" }, top_domains: { type: "array", items: { type: "string" } }, attestations: { type: "integer" }, success: { type: "integer" }, failure: { type: "integer" }, success_rate: { type: ["number", "null"] }, verified_routes: { type: "integer" }, last_active: { type: ["string", "null"], format: "date-time" } } } } },
        },
        NotFound: { type: "object", required: ["error"], properties: { error: { type: "string", example: "not_found" }, id: { type: "string" } } },
        ApiError: { type: "object", required: ["error"], properties: { error: { type: "string" } } },
      },
    },
  };
}

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
    <a href="https://waymark.network/pricing">pricing</a>
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

<h2 id="sec-canary">Freshness — canary re-verifications <span style="font-size:11px;color:var(--dim);text-transform:none;letter-spacing:0">routes re-driven against live APIs, kept green</span></h2>
<div class="panel" style="padding:16px 18px">
  <div style="display:flex;gap:22px;flex-wrap:wrap;margin-bottom:12px">
    <div><span id="cn-pass" style="font-size:26px;font-weight:800;color:var(--good);font-family:'JetBrains Mono',monospace">0</span> <span class="dim" style="font-size:12px">verified live</span></div>
    <div><span id="cn-fail" style="font-size:26px;font-weight:800;color:var(--bad);font-family:'JetBrains Mono',monospace">0</span> <span class="dim" style="font-size:12px">drift caught</span></div>
    <div style="flex:1"></div>
    <div class="dim" style="font-size:12px;max-width:340px;text-align:right">The canary fleet re-runs popular routes against live sandbox APIs daily and attests the result — the "drive the highways at 3am" freshness layer.</div>
  </div>
  <div id="canaryFeed"><span class="dim">No canary runs in the window yet — the fleet runs daily (and on demand).</span></div>
</div>

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
  <a href="https://mcp.waymark.network/contributors">contributors</a>
  <a href="https://mcp.waymark.network/sitemap.xml">all routes</a>
  <a href="https://github.com/waymark-network/waymark-mcp">github</a>
  <button class="install" id="copyBtn">claude mcp add --transport http waymark https://mcp.waymark.network/mcp</button>
</footer>
</div>

<script>
var $=function(id){return document.getElementById(id)};
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
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
  var row=e.target&&e.target.closest?e.target.closest(".route-row"):null;
  if(row){var h=row.getAttribute("data-href");if(h)window.open(h,"_blank","noopener");return;}
  var c=e.target&&e.target.closest?e.target.closest(".fchip"):null;
  if(!c)return;
  activeDomain=c.getAttribute("data-d")||"";
  var all=document.querySelectorAll(".fchip");
  for(var i=0;i<all.length;i++)all[i].classList.toggle("active",(all[i].getAttribute("data-d")||"")===activeDomain);
  applyDomainFilter();
});

/* Live loop (30s): stats + activity only — small payloads powering the health
   pill, counters, pulse, feed, canary and demand map. The heavy route corpus
   lives in loadRoutes() below on its own 5-min cadence (item 46). */
function load(){
  Promise.all([
    fetch("https://mcp.waymark.network/stats").then(function(x){return x.json()}),
    fetch("https://mcp.waymark.network/activity?limit=120").then(function(x){return x.json()})
  ]).then(function(res){
    var s=res[0],a=res[1];

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

    /* Canary freshness — attest events tagged note:"canary" */
    var canary=a.events.filter(function(e){return e.type==="attest"&&e.detail&&typeof e.detail.note==="string"&&e.detail.note.toLowerCase().indexOf("canary")>=0;});
    var cpass=canary.filter(function(e){return e.detail.outcome==="success";}).length;
    var cfail=canary.filter(function(e){return e.detail.outcome==="failure";}).length;
    $("cn-pass").textContent=cpass;$("cn-fail").textContent=cfail;
    $("canaryFeed").innerHTML=canary.length?canary.slice(0,12).map(function(e){
      var ok=e.detail.outcome==="success";
      return "<div style='display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid rgba(31,40,64,.5)'>"+
        "<span style='width:9px;height:9px;border-radius:50%;flex:0 0 auto;background:"+(ok?"var(--good)":"var(--bad)")+";box-shadow:0 0 8px "+(ok?"rgba(52,211,153,.6)":"rgba(248,113,113,.6)")+"'></span>"+
        "<span style='flex:1'>"+esc(e.detail.task||e.detail.route_id||"route")+"</span>"+
        "<span class='mono' style='font-size:12px;color:"+(ok?"var(--good)":"var(--bad)")+"'>"+(ok?"verified":"drift")+"</span>"+
        "<span class='dim mono' style='font-size:11px'>"+ago(e.t)+"</span></div>";
    }).join(""):"<span class='dim'>No canary runs in the window yet — the fleet runs daily (and on demand).</span>";

    /* Demand map — top queried domains + zero-result queries from /activity.
       Synthetic-traffic exclusion MIRRORS the worker's isSyntheticTraffic (~line 761)
       and the homepage's isRealDemand so all three surfaces report identical real
       demand: drop web-playground/playground demo traffic and .invalid/example-e2e
       e2e probes; a null/empty domain stays REAL (a domain-less agent query is a
       legitimate coverage gap, e.g. "rotate an AWS IAM key"). Filtering at the
       source guarantees BOTH the domain map and the zero-result list below exclude
       synthetic traffic — so the dashboard can no longer show "purple monkey
       dishwasher" as a coverage gap (the pollution item 11 removed from /demand). */
    function dmSynthetic(domain){if(domain==null)return false;var d=String(domain).toLowerCase();return d==="web-playground"||d==="playground"||d.indexOf(".invalid")>=0||d.indexOf("example-e2e")>=0;}
    var qevents=a.events.filter(function(e){return e.type==="query"&&e.detail&&!dmSynthetic(e.detail.domain)});
    var dcounts={};
    qevents.forEach(function(e){var d=e.detail.domain;if(d)dcounts[d]=(dcounts[d]||0)+1});
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
/* Corpus loop (5 min + ETag gate): /routes?all=1 is ~2.3 MB across 6,400+
   routes and changes at most hourly (often not for days), yet the old single
   loop refetched it every 30s per open tab AND rebuilt all table rows each
   poll — a full idx:routes read server-side + a multi-MB JSON.parse and a
   6,000+-row innerHTML rebuild client-side, for a table that almost never
   changed between polls. Now: fetch on load + every 300s; when the worker's
   weak ETag (already exposed via Access-Control-Expose-Headers) matches the
   last rendered corpus we skip the parse and the re-render entirely (the
   worker answers the conditional revalidation with a 0-byte 304). Falls back
   to a full render whenever the ETag is missing — never worse than before. */
var routesEtag=null;
function loadRoutes(){
  fetch("https://mcp.waymark.network/routes?all=1").then(function(x){
    var et=x.headers.get("ETag");
    if(et&&et===routesEtag)return null; /* corpus unchanged — skip parse + render */
    routesEtag=et;
    return x.json();
  }).then(function(r){
    if(!r)return;

    /* Routes table — clickable rows linking to /r/{id} */
    $("routes").tBodies[0].innerHTML=r.routes.map(function(x){
      var rate=x.success_rate===null?null:Math.round(x.success_rate*100);
      var url="https://mcp.waymark.network/r/"+esc(x.id||"");
      return "<tr class='route-row' data-domain=\\""+esc(x.domain||"")+"\\" data-href=\\""+url+"\\">"
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
  }).catch(function(){$("health").textContent="fetch error";$("dotEl")&&($("dotEl").style.background="#fbbf24")});
}
load();loadRoutes();
setInterval(load,30000);setInterval(loadRoutes,300000);

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
