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

/** Fire-and-forget activity log. Key sorts chronologically (ISO prefix). */
function logEvent(env: Env, type: EventType, detail: Record<string, unknown>): void {
  const t = new Date().toISOString();
  const key = `evt:${t}:${crypto.randomUUID().slice(0, 8)}`;
  env.ROUTES.put(key, JSON.stringify({ t, type, detail } satisfies ActivityEvent), {
    expirationTtl: EVENT_TTL_SECONDS,
  }).catch(() => {});
}

const tokenize = (s: string) =>
  s.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 2);

function keywordScore(query: string, route: Route): number {
  const q = new Set(tokenize(query));
  const r = tokenize(route.task + " " + route.domain);
  let hits = 0;
  for (const t of r) if (q.has(t)) hits++;
  if (hits === 0) return 0;
  // Trust-weighted: Laplace-smoothed success rate scales relevance.
  const a = route.attestations;
  const trust = (a.success + 1) / (a.success + a.failure + 2);
  return hits * trust;
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
        env.ROUTES.get("counter:queries").then((v) =>
          env.ROUTES.put("counter:queries", String(parseInt(v ?? "0", 10) + 1))
        ).catch(() => {});
        const routes = await loadAllRoutes(env);
        const q = task + (domain ? " " + domain : "");
        const ranked = routes
          .map((r) => ({ r, score: keywordScore(q, r) }))
          .filter((x) => x.score > 0)
          .sort((a, b) => b.score - a.score)
          .slice(0, limit)
          .map(({ r }) => ({
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
        logEvent(env, "query", {
          task: task.slice(0, 140),
          domain: domain ?? null,
          results: ranked.length,
        });
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
          logEvent(env, "contribute", { rejected: "bad_key", domain });
          return { content: [{ type: "text" as const, text: "Invalid API key. Get a contributor key at https://waymark.network" }], isError: true };
        }
        if (looksSensitive([task, domain, ...steps, ...gotchas].join(" "))) {
          logEvent(env, "contribute", { rejected: "sensitive_content", domain });
          return { content: [{ type: "text" as const, text: "Rejected: submission appears to contain credentials/secrets. Sanitize and resubmit procedure-only content." }], isError: true };
        }
        const id = crypto.randomUUID();
        const route: Route = {
          id, task, domain, steps, gotchas, contributor,
          created: new Date().toISOString(),
          attestations: { success: 0, failure: 0, lastAt: null },
        };
        await env.ROUTES.put(`route:${id}`, JSON.stringify(route));
        logEvent(env, "contribute", {
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
        route.attestations[outcome === "success" ? "success" : "failure"]++;
        route.attestations.lastAt = new Date().toISOString();
        await env.ROUTES.put(`route:${route_id}`, JSON.stringify(route));
        logEvent(env, "attest", {
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

/** Alpha-scale: KV list + parallel get. Replace with D1/Vectorize beyond ~5k routes. */
async function loadAllRoutes(env: Env): Promise<Route[]> {
  const list = await env.ROUTES.list({ prefix: "route:", limit: 1000 });
  const values = await Promise.all(list.keys.map((k) => env.ROUTES.get(k.name)));
  return values.filter((v): v is string => v !== null).map((v) => JSON.parse(v));
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

const CORS = { "Access-Control-Allow-Origin": "*" };

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
    if (pathname === "/stats") return stats(env);
    if (pathname === "/routes") return routesEndpoint(env);
    if (pathname === "/activity") {
      const limit = Math.min(parseInt(searchParams.get("limit") ?? "100", 10) || 100, 500);
      return activityEndpoint(env, limit);
    }
    if (pathname === "/dashboard") {
      return new Response(DASHBOARD_HTML, {
        headers: { "Content-Type": "text/html;charset=utf-8", "Cache-Control": "public, max-age=300" },
      });
    }
    return Response.redirect("https://waymark.network", 302);
  },
};

/** Public stats for the landing-page counters (CORS-open, cacheable 60s). */
async function stats(env: Env): Promise<Response> {
  try {
    const routes = await loadAllRoutes(env);
    const attestations = routes.reduce((n, r) => n + r.attestations.success + r.attestations.failure, 0);
    const queries = parseInt((await env.ROUTES.get("counter:queries")) ?? "0", 10);
    const events = (await env.ROUTES.list({ prefix: "evt:", limit: 1000 })).keys.length;
    return Response.json(
      { routes: routes.length, attestations, queries, events_30d: events },
      { headers: { ...CORS, "Cache-Control": "public, max-age=60" } }
    );
  } catch {
    return Response.json({ routes: 0, attestations: 0, queries: 0, events_30d: 0 }, { headers: CORS });
  }
}

/** Full route table for the dashboard (CORS-open). */
async function routesEndpoint(env: Env): Promise<Response> {
  const routes = await loadAllRoutes(env);
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
  return Response.json({ routes: rows }, { headers: { ...CORS, "Cache-Control": "public, max-age=30" } });
}

/** Recent activity feed, newest first (CORS-open). */
async function activityEndpoint(env: Env, limit: number): Promise<Response> {
  const events = await loadRecentEvents(env, limit);
  return Response.json({ events }, { headers: { ...CORS, "Cache-Control": "public, max-age=15" } });
}

/* ------------------------------------------------------------------ */
/* Dashboard — single dark-themed page, zero external dependencies.    */
/* ------------------------------------------------------------------ */

const DASHBOARD_HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Waymark — Activity Dashboard</title>
<style>
  :root{--bg:#0b0e14;--panel:#131826;--panel2:#0f1420;--line:#1f2840;--text:#e6ebf4;--dim:#8b96ad;
        --accent:#5eead4;--accent2:#818cf8;--good:#34d399;--bad:#f87171;--warn:#fbbf24;}
  *{box-sizing:border-box;margin:0}
  body{background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:24px;max-width:1200px;margin:0 auto}
  header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:20px}
  h1{font-size:22px;font-weight:700;letter-spacing:.3px}
  h1 .wm{color:var(--accent)}
  .sub{color:var(--dim);font-size:13px}
  .pill{margin-left:auto;font-size:12px;color:var(--dim)}
  .pill b{color:var(--good)}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
  .card .n{font-size:30px;font-weight:800;color:var(--accent)}
  .card:nth-child(2) .n{color:var(--accent2)} .card:nth-child(3) .n{color:var(--warn)} .card:nth-child(4) .n{color:var(--good)}
  .card .l{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.8px;margin-top:2px}
  h2{font-size:14px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin:22px 0 10px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{color:var(--dim);text-align:left;font-weight:600;padding:10px 14px;border-bottom:1px solid var(--line);font-size:11px;text-transform:uppercase;letter-spacing:.6px}
  td{padding:10px 14px;border-bottom:1px solid var(--panel2);vertical-align:top}
  tr:last-child td{border-bottom:0}
  .tag{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700}
  .tag.query{background:#1e2a4a;color:var(--accent2)}
  .tag.contribute{background:#123a32;color:var(--accent)}
  .tag.attest{background:#3a2f12;color:var(--warn)}
  .ok{color:var(--good);font-weight:700}.fail{color:var(--bad);font-weight:700}
  .dim{color:var(--dim)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
  .bar{height:6px;border-radius:3px;background:var(--line);overflow:hidden;min-width:70px}
  .bar i{display:block;height:100%;background:var(--good)}
  #spark{width:100%;height:90px;display:block}
  footer{color:var(--dim);font-size:12px;margin-top:24px;text-align:center}
  a{color:var(--accent)}
</style>
</head>
<body>
<header>
  <h1><span class="wm">Way</span>mark · Activity Dashboard</h1>
  <span class="sub">live system &amp; agent telemetry</span>
  <span class="pill">status <b id="health">…</b> · refreshed <span id="ts">…</span> · auto-refresh 30s</span>
</header>

<div class="cards">
  <div class="card"><div class="n" id="c-routes">–</div><div class="l">Routes on the map</div></div>
  <div class="card"><div class="n" id="c-queries">–</div><div class="l">Agent queries</div></div>
  <div class="card"><div class="n" id="c-attest">–</div><div class="l">Attestations</div></div>
  <div class="card"><div class="n" id="c-events">–</div><div class="l">Events (30 days)</div></div>
</div>

<h2>Activity — last 24h by hour</h2>
<div class="panel" style="padding:14px"><canvas id="spark"></canvas></div>

<h2>Live agent activity feed</h2>
<div class="panel"><table id="feed"><thead><tr><th>When (UTC)</th><th>Event</th><th>Detail</th></tr></thead><tbody></tbody></table></div>

<h2>Route map — trust table</h2>
<div class="panel"><table id="routes"><thead><tr><th>Task</th><th>Domain</th><th>Steps</th><th>✓ / ✗</th><th>Trust</th><th>Contributor</th><th>Last attested</th></tr></thead><tbody></tbody></table></div>

<footer>Waymark v0.2 · <a href="https://waymark.network">waymark.network</a> · MCP endpoint: <span class="mono">https://mcp.waymark.network/mcp</span> · public read, key-gated writes</footer>

<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const ago=t=>{const d=(Date.now()-new Date(t))/1000;if(d<60)return Math.floor(d)+"s ago";if(d<3600)return Math.floor(d/60)+"m ago";if(d<86400)return Math.floor(d/3600)+"h ago";return Math.floor(d/86400)+"d ago"};

function detailText(e){
  const d=e.detail||{};
  if(e.type==="query")return "“"+esc(d.task)+"”"+(d.domain?" · "+esc(d.domain):"")+" · <span class='dim'>"+d.results+" route(s) returned</span>";
  if(e.type==="contribute")return d.rejected?"<span class='fail'>rejected ("+esc(d.rejected)+")</span>"+(d.domain?" · "+esc(d.domain):""):"“"+esc(d.task)+"” · "+esc(d.domain)+" · "+d.steps+" steps · by <b>"+esc(d.contributor)+"</b>";
  if(e.type==="attest")return "<span class='"+(d.outcome==="success"?"ok":"fail")+"'>"+esc(d.outcome)+"</span> · “"+esc(d.task)+"”"+(d.note?" · "+esc(d.note):"");
  return esc(JSON.stringify(d));
}

function drawSpark(events){
  const cv=$("spark"),ctx=cv.getContext("2d");
  const W=cv.width=cv.clientWidth*2,H=cv.height=180;
  ctx.clearRect(0,0,W,H);
  const now=Date.now(),bins=new Array(24).fill(0);
  for(const e of events){const h=Math.floor((now-new Date(e.t))/36e5);if(h>=0&&h<24)bins[23-h]++}
  const max=Math.max(...bins,1),bw=W/24;
  ctx.fillStyle="#1f2840";ctx.fillRect(0,H-2,W,2);
  bins.forEach((v,i)=>{const bh=v/max*(H-30);ctx.fillStyle=v?"#5eead4":"#1f2840";ctx.fillRect(i*bw+3,H-2-bh,bw-6,Math.max(bh,2));
    if(v){ctx.fillStyle="#8b96ad";ctx.font="20px sans-serif";ctx.textAlign="center";ctx.fillText(v,i*bw+bw/2,H-8-bh)}});
}

async function load(){
  try{
    const [s,a,r,h]=await Promise.all([
      fetch("/stats").then(x=>x.json()),
      fetch("/activity?limit=100").then(x=>x.json()),
      fetch("/routes").then(x=>x.json()),
      fetch("/health").then(x=>x.text()).catch(()=>"down")
    ]);
    $("health").textContent=h==="ok"?"● online":"● "+h;
    $("c-routes").textContent=s.routes;$("c-queries").textContent=s.queries;
    $("c-attest").textContent=s.attestations;$("c-events").textContent=s.events_30d??0;
    $("ts").textContent=new Date().toLocaleTimeString();
    drawSpark(a.events);
    $("feed").tBodies[0].innerHTML=a.events.length?a.events.map(e=>
      "<tr><td class='mono dim'>"+esc(e.t.replace("T"," ").slice(0,19))+"<br>"+ago(e.t)+"</td>"+
      "<td><span class='tag "+e.type+"'>"+e.type+"</span></td><td>"+detailText(e)+"</td></tr>").join("")
      :"<tr><td colspan='3' class='dim'>No events logged yet — activity logging began with v0.2. Tool calls will appear here in real time.</td></tr>";
    $("routes").tBodies[0].innerHTML=r.routes.map(x=>{
      const rate=x.success_rate===null?null:Math.round(x.success_rate*100);
      return "<tr><td>"+esc(x.task)+"</td><td class='mono'>"+esc(x.domain)+"</td><td>"+x.steps+"</td>"+
      "<td><span class='ok'>"+x.success+"</span> / <span class='fail'>"+x.failure+"</span></td>"+
      "<td>"+(rate===null?"<span class='dim'>unrated</span>":"<div class='bar'><i style='width:"+rate+"%'></i></div><span class='dim'>"+rate+"%</span>")+"</td>"+
      "<td class='dim'>"+esc(x.contributor)+"</td><td class='dim'>"+(x.last_attested?ago(x.last_attested):"–")+"</td></tr>"}).join("");
  }catch(err){$("health").textContent="● fetch error";}
}
load();setInterval(load,30000);
</script>
</body>
</html>`;
