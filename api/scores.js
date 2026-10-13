// Leaderboard API for the Star Trace game (Vercel serverless function).
// Storage: Upstash Redis over its REST API (Vercel Marketplace integration sets the env vars).
//   lb:<diff>  sorted set  member = player id, score = best score
//   p:<id>     hash        sec (sha256 of the player's secret), nick, run:<diff> (JSON of the best run)
const crypto = require("crypto");

const REDIS_URL = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
const REDIS_TOKEN = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN;
const DIFFS = ["easy", "normal", "hyperdrive"];
const MAX_SCORE = 1000000;

async function redis(commands) {
  const res = await fetch(REDIS_URL.replace(/\/$/, "") + "/pipeline", {
    method: "POST",
    headers: { Authorization: `Bearer ${REDIS_TOKEN}`, "Content-Type": "application/json" },
    body: JSON.stringify(commands),
  });
  if (!res.ok) throw new Error("redis " + res.status);
  return (await res.json()).map(r => {
    if (r.error) throw new Error(r.error);
    return r.result;
  });
}

const cleanNick = s => String(s || "").replace(/[\u0000-\u001f\u007f<>]/g, "").replace(/\s+/g, " ").trim().slice(0, 12);
const sha256 = s => crypto.createHash("sha256").update(s).digest("hex");
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));

async function top(diff) {
  const [flat] = await redis([["ZREVRANGE", `lb:${diff}`, "0", "49", "WITHSCORES"]]);
  const ids = [], scores = [];
  for (let i = 0; i < flat.length; i += 2) { ids.push(flat[i]); scores.push(Number(flat[i + 1])); }
  if (!ids.length) return [];
  const details = await redis(ids.map(id => ["HMGET", `p:${id}`, "nick", `run:${diff}`]));
  return ids.map((id, i) => {
    const [nick, run] = details[i] || [];
    let r = {};
    try { r = JSON.parse(run || "{}"); } catch {}
    // Only a prefix of the id goes out, enough for a browser to spot its own row.
    return { id: id.slice(0, 12), nick: nick || "DRIVER", score: scores[i], acc: r.acc, combo: r.combo, fc: !!r.fc };
  });
}

async function submit(body, ip) {
  const id = String(body.playerId || ""), secret = String(body.secret || "");
  if (!/^[0-9a-f]{32}$/.test(id) || !/^[0-9a-f]{64}$/.test(secret)) return [400, { error: "bad_player" }];
  const diff = body.diff, nick = cleanNick(body.nick), score = Math.round(Number(body.score));
  if (!DIFFS.includes(diff) || !nick || !Number.isFinite(score) || score < 0 || score > MAX_SCORE) return [400, { error: "bad_input" }];

  // light per-IP limit: 20 submissions per 10 minutes
  const rl = `rl:${sha256(String(ip || "?")).slice(0, 16)}`;
  const [count] = await redis([["INCR", rl], ["EXPIRE", rl, "600", "NX"]]);
  if (count > 20) return [429, { error: "rate_limited" }];

  const [stored, prev] = await redis([["HGET", `p:${id}`, "sec"], ["ZSCORE", `lb:${diff}`, id]]);
  const hash = sha256(secret);
  if (stored && stored !== hash) return [403, { error: "not_owner" }];

  const previous = prev == null ? -1 : Number(prev);
  const improved = score > previous;
  const cmds = [["HSET", `p:${id}`, "sec", hash, "nick", nick]];
  if (improved) {
    cmds.push(["ZADD", `lb:${diff}`, String(score), id]);
    cmds.push(["HSET", `p:${id}`, `run:${diff}`, JSON.stringify({
      acc: Math.round(clamp(body.acc, 0, 100) * 100) / 100,
      combo: Math.round(clamp(body.combo, 0, 5000)), fc: body.fc === true, at: Date.now(),
    })]);
  }
  cmds.push(["ZREVRANK", `lb:${diff}`, id]);
  const out = await redis(cmds);
  const rank = out[out.length - 1];
  return [200, { improved, previous: Math.max(previous, 0), best: improved ? score : previous, rank: rank == null ? null : rank + 1 }];
}

module.exports = async (req, res) => {
  res.setHeader("Cache-Control", "no-store");
  if (!REDIS_URL || !REDIS_TOKEN) return res.status(503).json({ error: "not_configured" });
  try {
    if (req.method === "GET") {
      const diff = DIFFS.includes(req.query.diff) ? req.query.diff : "normal";
      return res.status(200).json({ diff, rows: await top(diff) });
    }
    if (req.method === "POST") {
      const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
      const ip = (req.headers["x-forwarded-for"] || "").split(",")[0].trim() || req.socket?.remoteAddress;
      const [status, payload] = await submit(body, ip);
      return res.status(status).json(payload);
    }
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: "method_not_allowed" });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "server_error" });
  }
};
