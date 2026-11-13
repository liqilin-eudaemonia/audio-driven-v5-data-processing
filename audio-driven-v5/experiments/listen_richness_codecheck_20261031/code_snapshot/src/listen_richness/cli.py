"""Explicit staged CLI. No service launch, GPU allocation or production mutation."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from functools import lru_cache
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time
import urllib.request

from . import VERSION
from .common import digest, file_hash, read, rows, safe_id, write, write_jsonl
from .extraction import MODEL, alias_map, contract, request, summarize, validate
from .selection import DEFAULT_POLICY, compact, select
from .statistics import snapshot


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


@contextmanager
def database(path):
    path = Path(path).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("This statistics DB is already being used") from None
        db = sqlite3.connect(path)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            old = db.execute("SELECT value FROM meta WHERE key='version'").fetchone()
            if old and old[0] != VERSION: raise ValueError("DB version differs; use a new output DB")
            db.execute("INSERT OR IGNORE INTO meta VALUES ('version',?)", (VERSION,))
            db.execute("""CREATE TABLE IF NOT EXISTS samples
                       (sid TEXT PRIMARY KEY, snapshot_sha TEXT NOT NULL, snapshot TEXT NOT NULL,
                        status TEXT NOT NULL, extraction TEXT, extraction_contract TEXT, attempts TEXT)""")
            db.execute("CREATE TABLE IF NOT EXISTS errors (key TEXT PRIMARY KEY, stage TEXT,error TEXT)")
            db.commit()
            code = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
            bind(db, "code_sha256", code)
            yield db
        finally: db.close()


def bind(db, key, value):
    encoded = dumps(value)
    old = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if old and old[0] != encoded:
        raise ValueError("Changed " + key + "; use a new DB to avoid mixing versions")
    db.execute("INSERT OR IGNORE INTO meta VALUES (?,?)", (key, encoded)); db.commit()


@lru_cache(maxsize=16)
def receipt(path, size, mtime):
    r = read(path)
    ds = {d["sample_id"]: d for d in r["decisions"]}
    if len(ds) != len(r["decisions"]): raise ValueError("Duplicate quality decisions")
    return r, ds


def indexed_sources(work):
    """Read the existing source index; never recursively scan result directories."""
    work = Path(work).resolve(); cfg = read(work / "config.json")
    gate = cfg.get("visual_quality_gate")
    for entry in rows(work / "ready.jsonl"):
        jid = safe_id(entry["job_id"])
        job = read(work / "jobs" / (jid + ".json"))
        if job["job_id"] != jid: raise ValueError("Job identity mismatch")
        ds = {}
        if gate:
            path = Path(gate["work"]) / "measure/completed" / (jid + ".json")
            if path.exists():
                st = path.stat(); r, ds = receipt(str(path), st.st_size, st.st_mtime_ns)
                if r["contract_sha256"] != gate["contract_sha256"] or r["source_result_sha256"] != job["source_result_sha256"]:
                    raise ValueError("Quality receipt/source contract mismatch")
        for row in job["rows"]:
            sid = safe_id(row["sample_id"])
            # Production sharding uses raw sample-ID bytes, not canonical JSON hashing.
            import hashlib
            path = work / "caption/results" / hashlib.sha256(sid.encode()).hexdigest()[:2] / (sid + ".json")
            yield dict(caption_file=str(path), quality_decision=ds.get(sid), expected_source=row["source"])


def prepare(args):
    counts = Counter(); seen = set()
    stream = indexed_sources(args.work) if args.work else rows(args.manifest)
    with database(args.db) as db:
        bind(db, "input", {"work": str(Path(args.work).resolve())} if args.work else
             {"manifest": str(Path(args.manifest).resolve()), "sha256": file_hash(args.manifest)})
        for item in stream:
            if args.limit and counts["attempted"] >= args.limit: break
            path = Path(item["caption_file"])
            if not path.exists(): counts["not_yet_captioned"] += 1; continue
            key = str(path.resolve())
            if key in seen: counts["duplicate_index_entry"] += 1; continue
            seen.add(key); counts["attempted"] += 1
            try:
                raw = path.read_bytes(); record = json.loads(raw)
                import hashlib
                sha = hashlib.sha256(raw).hexdigest()
                if "expected_source" in item and record.get("original_source", record["source"]) != item["expected_source"]:
                    raise ValueError("Caption/source index mismatch")
                s = snapshot(record, path, sha, item.get("quality_decision")); sid = safe_id(s["sample_id"])
                sh = digest(s); old = db.execute("SELECT snapshot_sha FROM samples WHERE sid=?", (sid,)).fetchone()
                if old:
                    if old[0] != sh: raise ValueError("Existing snapshot changed; use a new DB")
                    counts["reused"] += 1
                else:
                    status = "pending" if s["eligible_for_extraction"] else "excluded"
                    db.execute("INSERT INTO samples VALUES (?,?,?,?,NULL,NULL,NULL)", (sid, sh, dumps(s), status))
                    counts[status] += 1
                db.execute("DELETE FROM errors WHERE key=?", (key,))
            except Exception as exc:
                counts["errors"] += 1
                db.execute("INSERT OR REPLACE INTO errors VALUES (?,?,?)", (key, "prepare", str(exc)))
            if counts["attempted"] % 50 == 0: db.commit()
        db.commit()
    report = dict(version=VERSION, counts=dict(counts), limit=args.limit, full_scope_requested=args.limit == 0,
                  source_media_decoded=False, production_modified=False)
    write(Path(args.db).with_suffix(".prepare.json"), report)
    return report


def pending(db, ids=None, retry_failed=False):
    states = ("pending", "failed") if retry_failed else ("pending",)
    query = "SELECT sid,snapshot_sha,snapshot FROM samples WHERE status IN (%s) ORDER BY sid" % ",".join("?" for _ in states)
    for sid, sh, snap in db.execute(query, states):
        if ids is None or sid in ids: yield sid, sh, json.loads(snap)


def selected_ids(path):
    if not path: return None
    data = read(path)
    ids = data["sample_ids"]
    if not isinstance(ids, list) or len(set(ids)) != len(ids): raise ValueError("Invalid pilot sample IDs")
    return {safe_id(s) for s in ids}


def check_ids(db, ids):
    if ids is None: return
    for sid in ids:
        if db.execute("SELECT 1 FROM samples WHERE sid=?", (sid,)).fetchone() is None:
            raise ValueError("Pilot ID missing from statistics DB: " + sid)


def pilot(args):
    """Deterministic bounded-memory stratification over already prepared records."""
    import heapq
    buckets = defaultdict(list)
    def offer(kind, sid):
        rank = int(digest([args.seed, sid]), 16)
        heap = buckets[kind]; heapq.heappush(heap, (-rank, sid))
        if len(heap) > args.count: heapq.heappop(heap)
    with database(args.db) as db:
        for sid, _, s in pending(db):
            offer("random", sid)
            if s["nod"]["candidate_interval_count"]: offer("nod_candidates", sid)
            offer("silent" if s["audio"]["waveform_full_silence"] else "non_silent", sid)
            if (s["face"]["face_short_side_p10"] or 0) < 224: offer("small_face", sid)
            if s["crop_revision"] == "legacy_recorded_crop": offer("legacy_crop", sid)
    chosen = {}; quota = max(1, args.count // 6)
    for k in ("nod_candidates", "silent", "non_silent", "small_face", "legacy_crop"):
        added = 0
        for _, sid in sorted(buckets[k], reverse=True):
            if len(chosen) >= args.count or added >= quota: break
            if sid not in chosen: chosen[sid] = k; added += 1
    for _, sid in sorted(buckets["random"], reverse=True):
        if len(chosen) >= args.count: break
        chosen.setdefault(sid, "random")
    result = dict(schema="listen-richness-pilot-v1", seed=args.seed, requested=args.count,
                  sample_ids=sorted(chosen), strata=chosen,
                  note="Sampled from this DB's pending records only, not necessarily the entire corpus.")
    write(args.output, result)
    return dict(selected=len(chosen), requested=args.count)


def export_requests(args):
    ids = selected_ids(args.ids)
    with database(args.db) as db:
        check_ids(db, ids)
        c = contract(args.model); bind(db, "extraction_contract", c)
        write_jsonl(args.output, (dict(sample_id=sid, snapshot_sha256=sh, extraction_contract=c,
                                      messages=request(s)) for sid, sh, s in pending(db, ids, args.retry_failed)))
    return dict(output=args.output)


def call_model(s, endpoint, model, timeout, max_attempts, api_key=None):
    attempts = []; messages = request(s)
    for _ in range(max_attempts):
        attempt = {}
        try:
            body = dict(model=model, messages=messages, temperature=0, max_tokens=8192,
                        response_format={"type": "json_object"})
            headers = {"Content-Type": "application/json"}
            if api_key: headers["Authorization"] = "Bearer " + api_key
            req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                         dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.load(response)
            choice = data["choices"][0]; attempt.update(raw=data, finish_reason=choice["finish_reason"])
            if choice["finish_reason"] != "stop": raise ValueError("Incomplete model generation")
            payload = json.loads(choice["message"]["content"])
            validate(payload, s["caption"])
            attempts.append(attempt)
            return payload, attempts
        except Exception as exc:
            attempt["error"] = str(exc); attempts.append(attempt)
            # Fresh source prompt with validation feedback; no invented repair or reused draft.
            messages = request(s) + [dict(role="user", content="The last attempt failed validation: " + str(exc) +
                                        ". Return a fresh extraction using only the original caption.")]
    return None, attempts


def save_extraction(db, sid, payload, attempts, c):
    db.execute("UPDATE samples SET status=?,extraction=?,extraction_contract=?,attempts=? WHERE sid=?",
               ("ok" if payload is not None else "failed", dumps(payload) if payload is not None else None,
                dumps(c), dumps(attempts), sid))
    db.commit()


def extract(args):
    ids = selected_ids(args.ids); counts = Counter()
    if not 1 <= args.workers <= 32 or not 1 <= args.attempts <= 5 or args.timeout <= 0:
        raise ValueError("Invalid bounded concurrency/retry settings")
    with database(args.db) as db:
        check_ids(db, ids); c = contract(args.model); bind(db, "extraction_contract", c)
        stream = iter(pending(db, ids, args.retry_failed)); submitted = 0; exhausted = False
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {}
            while jobs or not exhausted:
                while not exhausted and len(jobs) < args.workers:
                    if args.limit and submitted >= args.limit: exhausted = True; break
                    item = next(stream, None)
                    if item is None: exhausted = True; break
                    sid, _, s = item; submitted += 1
                    future = pool.submit(call_model, s, args.endpoint, args.model, args.timeout, args.attempts,
                                         os.environ.get(args.api_key_env) if args.api_key_env else None)
                    jobs[future] = sid
                if not jobs: break
                done, _ = wait(jobs, return_when=FIRST_COMPLETED)
                for future in done:
                    sid = jobs.pop(future); payload, attempts = future.result()
                    save_extraction(db, sid, payload, attempts, c)
                    counts["ok" if payload is not None else "failed"] += 1
    return dict(counts)


def import_responses(args):
    counts = Counter()
    with database(args.db) as db:
        c = contract(args.model); bind(db, "extraction_contract", c)
        for r in rows(args.responses):
            sid = r["sample_id"]
            old = db.execute("SELECT snapshot_sha,snapshot,status,extraction FROM samples WHERE sid=?", (sid,)).fetchone()
            if old is None or old[0] != r["snapshot_sha256"] or r["extraction_contract"] != c:
                raise ValueError("Response/snapshot/model contract mismatch")
            if old[2] == "excluded": raise ValueError("Cannot import extraction for excluded sample")
            if r.get("finish_reason") != "stop": raise ValueError("Incomplete imported response")
            payload = validate(r["extraction"], json.loads(old[1])["caption"])
            if old[2] == "ok":
                if json.loads(old[3]) != payload: raise ValueError("Refusing to overwrite an accepted extraction")
                counts["reused"] += 1; continue
            save_extraction(db, sid, payload, [dict(import_file=str(Path(args.responses).resolve()), response=r)], c)
            counts["ok"] += 1
    return dict(counts)


def build_rows(db, aliases, ids=None):
    for sid, sh, raw, ext in db.execute("SELECT sid,snapshot_sha,snapshot,extraction FROM samples WHERE status='ok' ORDER BY sid"):
        if ids is not None and sid not in ids: continue
        s = json.loads(raw); e = json.loads(ext); summary = summarize(s, e, aliases)
        yield s, e, summary, dict(compact(s, summary), snapshot_sha256=sh)


def fresh_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()): raise ValueError("Output directory must be empty; do not mix partial/old exports")
    return path


def export(args):
    aliases_doc = read(args.aliases) if args.aliases else None; aliases = alias_map(aliases_doc)
    out = fresh_output(args.output); counts = Counter(); catalog = defaultdict(lambda: {"samples": 0, "pairs": set(), "examples": []})
    # Stream JSONL; optional Parquet writes bounded record batches.
    parquet = None; buffer = []
    if args.parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq
    with database(args.db) as db, (out / "sample_stats.jsonl").open("w") as stats, (out / "behavior_evidence.jsonl").open("w") as ev:
        statuses = dict(db.execute("SELECT status,COUNT(*) FROM samples GROUP BY status"))
        try:
            for s, e, summary, row in build_rows(db, aliases):
                stats.write(dumps(row) + "\n")
                ev.write(dumps(dict(sample_id=s["sample_id"], caption=s["caption"], caption_file=s["caption_file"],
                                    caption_file_sha256=s["caption_file_sha256"], extraction=e,
                                    summary=summary, time_evidence=dict(nod=s["nod"], audio=s["audio"]),
                                    face=s["face"], crop_revision=s["crop_revision"],
                                    snapshot_sha256=digest(s), training_admitted=False)) + "\n")
                counts["exported"] += 1
                if summary["consistency_flags"]: counts["consistency_review"] += 1
                for g in summary["groups"]:
                    c = catalog[g["key"]]; c["samples"] += 1; c["pairs"].add(row["pair"])
                    if len(c["examples"]) < 3:
                        c["examples"].append(dict(sample_id=s["sample_id"], quotes=[b["quote"] for b in e["behaviors"] if b["id"] in g["evidence_ids"]]))
                if args.parquet:
                    # Store nested fields as JSON strings to keep a stable schema across empty/list cases.
                    flat = {k: dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()}
                    # Explicit nullable float types avoid first-batch all-null inference.
                    for key in ("face_short_side_p10", "face_short_side_p50", "laplacian_reference_p80"):
                        if flat[key] is not None: flat[key] = float(flat[key])
                    buffer.append(flat)
                    if len(buffer) >= 1024:
                        table = pa.Table.from_pylist(buffer)
                        if parquet is None:
                            schema = pa.schema([pa.field(f.name, pa.float64()) if f.name in
                                                ("face_short_side_p10", "face_short_side_p50", "laplacian_reference_p80") else f for f in table.schema])
                            parquet = pq.ParquetWriter(out / "sample_stats.parquet", schema)
                        parquet.write_table(table.cast(parquet.schema)); buffer.clear()
            if args.parquet and buffer:
                table = pa.Table.from_pylist(buffer)
                if parquet is None: parquet = pq.ParquetWriter(out / "sample_stats.parquet", table.schema)
                parquet.write_table(table.cast(parquet.schema))
        finally:
            if parquet is not None: parquet.close()
        meta = dict(db.execute("SELECT key,value FROM meta"))
        errors = db.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
    write(out / "behavior_catalog.json", dict(schema="listen-richness-catalog-v1", aliases=aliases_doc,
          types={k: dict(samples=v["samples"], pairs=len(v["pairs"]), examples=v["examples"],
                         status="caption_extracted_candidate") for k,v in sorted(catalog.items())},
          note="Open vocabulary; reviewed aliases only. Unresolved synonyms may remain. Not a verified exhaustive taxonomy."))
    report = dict(counts=dict(counts), statuses=statuses, prepare_errors=errors, metadata=meta,
                  alias_sha256=digest(aliases_doc), all_prepared_resolved=not any(statuses.get(s, 0) for s in ("pending", "failed")),
                  training_admitted=False)
    write(out / "EXPORT_COMPLETE.json", report)
    return report


def selection(args):
    aliases_doc = read(args.aliases) if args.aliases else None; aliases = alias_map(aliases_doc)
    policy = read(args.policy) if args.policy else dict(DEFAULT_POLICY)
    out = fresh_output(args.output)
    with database(args.db) as db:
        statuses = dict(db.execute("SELECT status,COUNT(*) FROM samples GROUP BY status"))
        errors = db.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
        unresolved = errors + statuses.get("pending", 0) + statuses.get("failed", 0)
        if unresolved and not args.allow_partial:
            raise ValueError("Unresolved records; finish extraction or explicitly use --allow-partial for a pilot")
        matrix = [row for _, _, _, row in build_rows(db, aliases)]
        chosen, report = select(matrix, read(args.catalog), policy)
        report.update(prepare_errors=errors, statuses=statuses, partial=bool(unresolved),
                      catalog_sha256=file_hash(args.catalog), aliases_sha256=digest(aliases_doc),
                      extraction_metadata=dict(db.execute("SELECT key,value FROM meta")))
        def manifests():
            for row in chosen:
                raw, sh = db.execute("SELECT snapshot,snapshot_sha FROM samples WHERE sid=?", (row["sample_id"],)).fetchone()
                s = json.loads(raw)
                if file_hash(s["caption_file"]) != s["caption_file_sha256"]:
                    raise ValueError("Original caption file changed since statistics preparation")
                yield dict(row, caption=s["caption"], source=s["source"], caption_file=s["caption_file"],
                           caption_file_sha256=s["caption_file_sha256"], snapshot_sha256=sh,
                           crop_provenance=s["crop_provenance"], audio_recipe_sha256=s["audio"]["recipe_sha256"],
                           training_admitted=False, human_ground_truth=False)
        write_jsonl(out / "selected_manifest.jsonl", manifests())
    write(out / "selection_policy.json", policy); write(out / "selection_report.json", report)
    write(out / "EXPORT_COMPLETE.json", dict(selection_complete=report["selection_complete"],
          partial=report["partial"], manifest_sha256=file_hash(out / "selected_manifest.jsonl"),
          training_admitted=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="CPU metadata only; default limited to 200 available captions")
    origin = p.add_mutually_exclusive_group(required=True)
    origin.add_argument("--work"); origin.add_argument("--manifest")
    p.add_argument("--limit", type=int, default=200); p.set_defaults(func=prepare)
    p = sub.add_parser("pilot", help="Stratify from prepared pending records")
    p.add_argument("--count", type=int, default=200); p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--output", required=True); p.set_defaults(func=pilot)
    for name in ("requests", "extract"):
        p = sub.add_parser(name); p.add_argument("--model", default=MODEL); p.add_argument("--ids")
        p.add_argument("--retry-failed", action="store_true")
        if name == "requests":
            p.add_argument("--output", required=True); p.set_defaults(func=export_requests)
        else:
            p.add_argument("--endpoint", required=True, help="Existing OpenAI-compatible base URL ending in /v1")
            p.add_argument("--workers", type=int, default=2); p.add_argument("--limit", type=int, default=200)
            p.add_argument("--timeout", type=float, default=120); p.add_argument("--attempts", type=int, default=2)
            p.add_argument("--api-key-env"); p.set_defaults(func=extract)
    p = sub.add_parser("import-responses"); p.add_argument("--responses", required=True)
    p.add_argument("--model", default=MODEL); p.set_defaults(func=import_responses)
    for name in ("export", "select"):
        p = sub.add_parser(name); p.add_argument("--output", required=True); p.add_argument("--aliases")
        if name == "export":
            p.add_argument("--parquet", action="store_true"); p.set_defaults(func=export)
        else:
            p.add_argument("--policy"); p.add_argument("--catalog", required=True)
            p.add_argument("--allow-partial", action="store_true"); p.set_defaults(func=selection)
    for p in sub.choices.values(): p.add_argument("--db", required=True)
    args = parser.parse_args(argv)
    if getattr(args, "limit", 0) < 0 or getattr(args, "count", 1) <= 0: parser.error("Invalid limit/count")
    result = args.func(args)
    print(dumps(result))
    if result.get("counts", {}).get("errors", 0) or result.get("failed", 0): return 1
    if args.command == "select" and not result["selection_complete"]: return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
