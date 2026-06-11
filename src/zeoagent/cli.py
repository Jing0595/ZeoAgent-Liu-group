"""CLI driver for ZeoAgent to exercise the agent graph without FastAPI."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional

from zeoagent.agent.agent import AgentState, run_zeoagent_conversation


DEFAULT_OUTPUT_DIR = Path("out") / "cli_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ZeoAgent via CLI. If no prompt is provided, starts an interactive loop."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="User prompt/question, e.g., 'What is the pore diameter of CHA?'. If omitted, interactive mode starts.",
    )
    parser.add_argument(
        "--history",
        help="Optional JSON array of messages [{'role':'user','content':'...'}]",
        default=None,
    )
    parser.add_argument(
        "--history-file",
        help="Path to a JSON file with prior messages; will be merged into the session state.",
    )
    parser.add_argument(
        "--session-file",
        help="Path to persist conversation state across runs (memory + traces).",
    )
    parser.add_argument(
        "--trace-file",
        help="Write run report JSON (default: out/cli_outputs/YYYYMMDD-<prompt>.json).",
    )
    return parser.parse_args()


def _load_state(args: argparse.Namespace) -> AgentState:
    state = AgentState()
    preload = args.session_file or args.history_file
    if args.history:
        try:
            loaded = json.loads(args.history)
            if isinstance(loaded, list):
                state.messages.extend(loaded)
        except json.JSONDecodeError:
            print("Invalid history JSON; ignoring.", file=sys.stderr)

    if preload:
        path = Path(preload)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                state.messages.extend(data.get("messages", []))
                state.memory.update(data.get("memory", {}))
            except json.JSONDecodeError:
                print(f"Invalid JSON in {preload}; starting fresh.", file=sys.stderr)
    return state


def _save_state(state: AgentState, path_str: str | None) -> None:
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.snapshot(), indent=2, ensure_ascii=False))


def _default_report_path(prompt: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    name = _sanitize_prompt_filename(prompt, max_length=60)
    if not name:
        name = "prompt"
    return DEFAULT_OUTPUT_DIR / f"{stamp}-{name}.json"


def _sanitize_prompt_filename(prompt: str, max_length: int = 80) -> str:
    cleaned = []
    for ch in (prompt or "").strip():
        if ch.isalnum() or ch in {" ", "-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    name = "".join(cleaned)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "prompt"
    if len(name) > max_length:
        name = name[:max_length].rstrip("_")
    return name


def _persist_cli_report(prompt: str, report: Dict[str, object], report_path: Optional[str] = None) -> Optional[Path]:
    if not report:
        return None
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    target_dir = DEFAULT_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    base_path = _default_report_path(prompt)
    path = base_path
    if path.exists():
        index = 1
        while True:
            candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
            if not candidate.exists():
                path = candidate
                break
            index += 1
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _run_once(prompt: str, state: AgentState) -> Dict[str, object]:
    try:
        result = run_zeoagent_conversation(
            [{"role": "user", "content": prompt}],
            state=state,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return {}
    print(result.get("answer") or "")
    return result


def main() -> None:
    args = parse_args()
    state = _load_state(args)

    # Interactive when no prompt is given
    if args.prompt is None:
        print("ZeoAgent CLI (interactive). Type 'exit' or Ctrl-D to quit.")
        try:
            while True:
                user_input = input("> ").strip()
                if not user_input or user_input.lower() in {"exit", "quit"}:
                    break
                result = _run_once(user_input, state)
                _persist_cli_report(user_input, result, args.trace_file)
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        result = _run_once(args.prompt, state)
        _persist_cli_report(args.prompt, result, args.trace_file)

    _save_state(state, args.session_file or args.history_file)


if __name__ == "__main__":
    main()
