
import streamlit as st
import asyncio
import aiohttp
import time
import json
import os
import re
import random
import difflib
import matplotlib.pyplot as plt
from urllib.parse import urlsplit, urlparse
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Race Condition Tester", layout="wide")

# -----------------------
# Files & constants
# -----------------------
HOME = Path.home()
HISTORY_FILE = HOME / ".race_tester_history.json"
LOG_FILE = HOME / ".race_tester_log.jsonl"
MAX_HISTORY = 100

# -----------------------
# Utility functions
# -----------------------
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(history):
    try:
        HISTORY_FILE.write_text(json.dumps(history[-MAX_HISTORY:], indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Failed to save history: {e}")

def append_log(entry: dict):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        st.error(f"Failed to write log: {e}")

def parse_raw_request(raw_text: str):
    """
    Parse a raw HTTP request (Burp/raw format).
    This is a more robust version that handles different HTTP versions.
    Returns dict with method, path, http_version, headers (dict), body (str).
    Raises ValueError on parse failure.
    """
    lines = raw_text.strip().splitlines()
    if not lines:
        raise ValueError("Empty request")

    # More robust regex for the request line, handles HTTP/1.0, 1.1, 2 etc.
    request_line_pattern = re.compile(
        r"^(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD|TRACE|CONNECT)\s+(\S+)\s+HTTP/\d(?:\.\d+)?$", re.I
    )
    request_line = lines[0]
    match = request_line_pattern.match(request_line)
    if not match:
        raise ValueError("Invalid request line. Expected format: 'METHOD /path HTTP/X.X'")

    method, path, http_version = match.group(1), match.group(2), request_line.split()[-1]

    headers = {}
    body_start_index = -1
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "":
            body_start_index = i + 1
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
        else: # Handle malformed or multi-line headers
            if headers:
                last_key = list(headers.keys())[-1]
                headers[last_key] += " " + line.strip()

    body = ""
    if body_start_index != -1:
        body = "\n".join(lines[body_start_index:])

    return {
        "method": method.upper(),
        "path": path,
        "http_version": http_version,
        "headers": headers,
        "body": body
    }


def substitute_variables(text: str, variables: dict):
    """
    Replace placeholders like _name_ or {{name}} with values in variables dict.
    """
    if not text:
        return text
    pattern = re.compile(r"_(\w+)_|\{\{(\w+)\}\}")
    def repl(m):
        key = m.group(1) or m.group(2)
        return variables.get(key, m.group(0))
    return pattern.sub(repl, text)

def short_diff(a: str, b: str, max_lines=20):
    s = difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True), lineterm="")
    lines = list(s)
    if not lines:
        return "(identical)"
    return "\n".join(lines[:max_lines])

# -----------------------
# Async request logic
# -----------------------
async def send_request(session, method, url, headers, data, timeout):
    start = time.perf_counter()
    try:
        
        async with session.request(method, url, headers=headers, data=data, timeout=timeout) as resp:
            text = await resp.text(errors='replace')
            elapsed = (time.perf_counter() - start) * 1000.0
            return {
                "ok": True,
                "status": resp.status,
                "headers": dict(resp.headers),
                "body": text,
                "elapsed_ms": elapsed
            }
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000.0
        return {"ok": False, "error": str(e), "elapsed_ms": elapsed}

def get_target_url(parsed_request):
    """
    Intelligently construct the base URL from the parsed request.
    It infers the scheme (http/https) from Origin or Referer headers,
    defaulting to https.
    """
    headers_lower = {k.lower(): v for k, v in parsed_request["headers"].items()}
    host = headers_lower.get("host")
    if not host:
        raise ValueError("Host header is missing, cannot determine target.")

    scheme = "https" # Default to https for safety
    origin = headers_lower.get("origin")
    referer = headers_lower.get("referer")

    if origin and "://" in origin:
        scheme = urlparse(origin).scheme
    elif referer and "://" in referer:
        scheme = urlparse(referer).scheme

    return f"{scheme}://{host}"


async def run_burst(parsed_request, concurrency, variable_template, jitter_ms_min, jitter_ms_max, timeout_sec, progress_hook=None):
    """
    Send 'concurrency' concurrent requests in one burst. Returns list of results.
    progress_hook: function(sent, total) -> None (called from event loop)
    """
    results = []
    base_url = get_target_url(parsed_request)


    conn = aiohttp.TCPConnector(ssl=False) if base_url.startswith("https://") else None

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        for i in range(concurrency):
            vars_for_req = {}
            for k, v in variable_template.items():
                if isinstance(v, str) and v.lower() == "unique":
                    vars_for_req[k] = f"{int(time.time()*1000)}-{random.randint(1000,9999)}"
                else:
                    vars_for_req[k] = str(v)

            path_sub = substitute_variables(parsed_request["path"], vars_for_req)
            url = f"{base_url}{path_sub}"

            headers_sub = {k: substitute_variables(v, vars_for_req) for k, v in parsed_request["headers"].items()}
            body_sub = substitute_variables(parsed_request["body"], vars_for_req) if parsed_request.get("body") else None

            delay_ms = random.randint(jitter_ms_min, jitter_ms_max) if jitter_ms_min is not None else 0

            async def single_request(url=url, method=parsed_request["method"], headers=headers_sub, body=body_sub, delay_ms=delay_ms):
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000.0)
                res = await send_request(session, method, url, headers, body, timeout_sec)
                return {"url": url, "method": method, "headers": headers, "body_sent": body, "result": res}

            tasks.append(single_request())

        total = len(tasks)
        gathered = await asyncio.gather(*tasks, return_exceptions=False)
        for completed, r in enumerate(gathered, 1):
            results.append(r)
            if progress_hook:
                try:
                    progress_hook(completed, total)
                except Exception:
                    pass

    return results

def run_full_test(parsed_request, concurrency, repetitions, variable_template, jitter_min, jitter_max, timeout_sec, progress_bar, status_text_area):
    """
    Synchronous wrapper to run the full test from Streamlit.
    """
    all_records = []
    total_steps = repetitions * concurrency
    sent_so_far = 0

    def make_progress_hook(burst_rep):
        def hook(burst_sent_inner, burst_total_inner):
            nonlocal sent_so_far
            sent_in_prev_bursts = (burst_rep - 1) * concurrency
            sent_so_far = sent_in_prev_bursts + burst_sent_inner
            frac = min(1.0, sent_so_far / max(1, total_steps))
            progress_bar.progress(frac)
            status_text_area.text(f"Sent {sent_so_far}/{total_steps} requests (burst {burst_rep}/{repetitions})")
        return hook

    async def async_runner():
        nonlocal all_records
        for rep in range(1, repetitions + 1):
            status_text_area.text(f"Starting burst {rep}/{repetitions}...")
            burst_progress_hook = make_progress_hook(rep)
            burst_results = await run_burst(parsed_request, concurrency, variable_template, jitter_min, jitter_max, timeout_sec, progress_hook=burst_progress_hook)
            timestamp = datetime.utcnow().isoformat() + "Z"
            for idx, r in enumerate(burst_results, start=1):
                rec = {
                    "timestamp": timestamp,
                    "rep": rep,
                    "idx_in_burst": idx,
                    "url": r["url"],
                    "method": r["method"],
                    "headers_sent": r["headers"],
                    "body_sent": r["body_sent"],
                    "result": r["result"]
                }
                all_records.append(rec)
                append_log(rec)
            status_text_area.text(f"Completed burst {rep}/{repetitions}. Collected {len(all_records)} records so far.")
        return all_records

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
             records = loop.run_until_complete(async_runner())
        else:
             records = asyncio.run(async_runner())
    except RuntimeError:
        records = asyncio.run(async_runner())
    except Exception as e:
        st.error(f"Error during test run: {e}")
        return [] # Return empty list on failure
    return records

# -----------------------
# Streamlit UI
# -----------------------
st.title("Race Condition Tester")

with st.sidebar:
    st.header("Configuration")
    concurrency = st.slider("Concurrent requests per burst", min_value=1, max_value=200, value=10, step=1)
    repetitions = st.number_input("Number of bursts (repetitions)", min_value=1, max_value=20, value=3, step=1)
    timeout_sec = st.number_input("Per-request timeout (seconds)", min_value=1, max_value=60, value=10, step=1)
    jitter_enabled = st.checkbox("Enable randomized jitter between requests in a burst", value=True)
    if jitter_enabled:
        jitter_min = st.number_input("Jitter min (ms)", min_value=0, max_value=5000, value=0, step=10)
        jitter_max = st.number_input("Jitter max (ms)", min_value=0, max_value=5000, value=200, step=10)
        if jitter_max < jitter_min:
            st.error("Jitter max must be >= jitter min")
    else:
        jitter_min, jitter_max = 0, 0

    st.markdown("---")
    st.subheader("Variable substitution")
    st.markdown("Placeholders: `_name_` or `{{name}}`. Set a variable to `unique` to generate a per-request unique token.")
    vars_raw = st.text_area("Variables (JSON)", value='{"token": "unique"}', height=100)
    try:
        variable_template = json.loads(vars_raw) if vars_raw.strip() else {}
    except Exception:
        st.error("Variables must be valid JSON (e.g. {\"token\": \"unique\"})")
        variable_template = {}

    st.markdown("---")
    st.subheader("History & logging")
    if st.button("Show file locations"):
        st.info(f"History saved to: {HISTORY_FILE}")
        st.info(f"Logs appended to: {LOG_FILE}")

    save_to_history = st.checkbox("Save request to history after run", value=True)
    if st.button("Clear saved request history"):
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        st.success("History cleared.")

    st.markdown("---")
    st.header("⚠️ Ethical Use Warning")
    st.warning(
        "You are responsible for your actions. Only use this tool on systems for which you have "
        "explicit, written permission to conduct testing. Unauthorized testing is illegal."
    )

st.subheader("Paste Raw HTTP Request (from Burp or other tools)")
default_req = (
    "POST /api/v1/transfer HTTP/1.1\n"
    "Host: example.com\n"
    "Origin: https://example.com\n"
    "User-Agent: RaceTester/1.0\n"
    "Content-Type: application/json\n\n"
    '{"amount": "100", "to_account": "_token_"}'
)

if "raw_request" not in st.session_state:
    st.session_state.raw_request = default_req

raw_request = st.text_area("Raw request", key="raw_request", height=220)

col_run, col_result = st.columns([1, 2])
with col_run:
    st.markdown("### Run controls")
    run_btn = st.button("Run Test", type="primary")
    st.caption("Ensure the Host header is correct. The scheme (http/https) will be inferred from Origin/Referer or default to https.")

with col_result:
    st.markdown("### Status")
    progress_bar = st.progress(0.0)
    status_text_area = st.empty()
    status_text_area.text("Idle.")

history = load_history()
st.subheader("Request History")
hist_cols = st.columns([3,1,1])
with hist_cols[0]:
    history_options = ["(none)"] + [h.get("title", h.get("ts","")) for h in history]
    selected_history_title = st.selectbox("Load a saved request", options=history_options, index=0)
with hist_cols[1]:
    if st.button("Load selected"):
        if selected_history_title != "(none)":
            for h in history:
                if h.get("title") == selected_history_title or h.get("ts") == selected_history_title:
                    st.session_state.raw_request = h.get("raw", raw_request)
                    st.rerun()
                    break
with hist_cols[2]:
    if st.button("Save current to history"):
        try:
            parsed_try = parse_raw_request(raw_request)
            host_header = parsed_try["headers"].get("Host", "unknown.host")
            entry = {"ts": datetime.utcnow().isoformat()+"Z", "title": f"{parsed_try['method']} {host_header}{parsed_try['path']}", "raw": raw_request}
            history.append(entry)
            save_history(history)
            st.success("Saved to history.")
            st.rerun()
        except Exception as e:
            st.error(f"Cannot save — parse failed: {e}")

parsed = None
parse_error = None
try:
    parsed = parse_raw_request(raw_request)
    st.markdown("**Parsed request preview:**")
    st.text(f"Method: {parsed['method']}")
    st.text(f"Path: {parsed['path']}")
    st.text(f"HTTP Version: {parsed['http_version']}")
    with st.expander("Parsed Headers"):
        st.json(parsed['headers'])
    if parsed.get("body"):
        st.text("Body:")
        st.code(parsed["body"], language="text")
except Exception as e:
    parse_error = str(e)
    st.error(f"Parse error: {parse_error}")

if run_btn:
    if parse_error:
        st.error("Fix parse errors before running.")
    else:
        progress_bar.progress(0.0)
        status_text_area.text("Preparing test...")
        jitter_min_val = int(jitter_min) if jitter_enabled else 0
        jitter_max_val = int(jitter_max) if jitter_enabled else 0
        
        records = run_full_test(parsed, concurrency, int(repetitions), variable_template, jitter_min_val, jitter_max_val, int(timeout_sec), progress_bar, status_text_area)
        
        if records:
            st.success(f"Test completed. Collected {len(records)} records.")
            if save_to_history:
                host_header = parsed["headers"].get("Host", "unknown.host")
                entry = {"ts": datetime.utcnow().isoformat()+"Z", "title": f"{parsed['method']} {host_header}{parsed['path']}", "raw": raw_request}
                history.append(entry)
                save_history(history)

            st.subheader("Summary")
            status_counts = {}
            times = []
            for r in records:
                res = r["result"]
                status = res.get("status", "ERR") if res.get("ok") else "ERR"
                status_counts[status] = status_counts.get(status, 0) + 1
                if res.get("ok"):
                    times.append(res.get("elapsed_ms", 0))
                
            st.write("Status code distribution:", status_counts)
            if times:
                st.write(f"Min/Max/Avg response time (ms): {min(times):.2f} / {max(times):.2f} / {(sum(times)/len(times)):.2f}")

            st.subheader("Response differences (first response vs others)")
            if records and records[0]["result"].get("ok"):
                first_body = records[0]["result"].get("body", "")
                with st.expander("Show diffs of response bodies"):
                    for i, rec in enumerate(records[1:], start=1):
                        body = rec["result"].get("body", "") if rec["result"].get("ok") else f"ERROR: {rec['result'].get('error')}"
                        diff = short_diff(first_body, body)
                        if diff != "(identical)":
                            st.markdown(f"**Record {i+1} vs Record 1 (status: {rec['result'].get('status') or 'ERR'})**")
                            st.code(diff, language='diff')

            st.subheader("Raw responses")
            # FIX: Removed nested expanders to prevent StreamlitAPIException
            with st.expander("Show all request/response pairs"):
                for i, rec in enumerate(records):
                    st.markdown("---")
                    st.markdown(f"**Record {i+1} — Rep {rec['rep']} Idx {rec['idx_in_burst']} — Status: {rec['result'].get('status') or 'ERR'}**")
                    st.write("URL:", rec["url"])
                    st.write(f"Elapsed (ms): {rec['result'].get('elapsed_ms', 0):.2f}")
                    if rec["result"].get("ok"):
                        st.write("**Response headers:**")
                        st.json(rec["result"].get("headers"))
                        st.write("**Response body (first 4000 chars):**")
                        st.code(rec["result"].get("body", "")[:4000])
                    else:
                        st.write("**Error:**")
                        st.error(rec["result"].get("error"))

            st.subheader("Response time plot")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot([r["result"].get("elapsed_ms", 0) for r in records], marker="o", linestyle='-')
            ax.set_title("Response times (ms) per request")
            ax.set_xlabel("Request #")
            ax.set_ylabel("Elapsed ms")
            st.pyplot(fig)
        else:
            st.info("No records collected or test failed to run.")
