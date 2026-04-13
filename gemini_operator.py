#!/usr/bin/env python3
"""
gemini-operator: A command-line tool powered by Gemini that converts natural
language prompts into OS-specific shell commands, explains what they will do,
and asks for confirmation before executing them.

gemini-operator：以 Gemini 將自然語言轉成符合目前作業系統的 shell 指令、說明
其效果與副作用，並在執行前請使用者確認的指令列工具。
"""

import getpass
import json
import os
import platform
import subprocess
import sys

import google.generativeai as genai
from google.api_core import exceptions as google_api_exceptions

# ─── ANSI colour helpers / ANSI 色彩輔助 ────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
MAGENTA = "\033[35m"


def _colour(text: str, *codes: str) -> str:
    """
    Wrap *text* in the given ANSI codes when stdout is a TTY.
    僅在標準輸出為終端機（TTY）時，為 *text* 套用指定的 ANSI 轉義序列。
    """
    if sys.stdout.isatty():
        return "".join(codes) + text + RESET
    return text


# ─── OS detection / 作業系統偵測 ───────────────────────────────────────────

def detect_os() -> str:
    """
    Return a normalised OS identifier: 'windows', 'macos', or 'linux'.
    回傳正規化後的作業系統代碼：'windows'、'macos' 或 'linux'。
    """
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    if system == "darwin":
        return "macos"
    return "linux"


# Human-readable OS labels for the banner / 啟動橫幅上顯示的作業系統說明文字
OS_DISPLAY = {
    "windows": "Windows (cmd)",
    "macos": "macOS (bash/zsh)",
    "linux": "Linux (bash)",
}

# Wording injected into the model system prompt / 寫入模型系統提示的 shell 描述
OS_SHELL_NAME = {
    "windows": "Windows Command Prompt (cmd.exe)",
    "macos": "macOS bash/zsh shell",
    "linux": "Linux bash shell",
}

# ─── Gemini integration / Gemini 整合 ───────────────────────────────────────

# Default when GEMINI_MODEL is unset; Enter in the picker selects this id.
# 未設定 GEMINI_MODEL 時的預設值；互動選單中按 Enter 會選此模型 ID。
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"

# System instruction for JSON {command, explanation}; keep English for the model.
# 要求模型輸出 JSON（command、explanation）；內文維持英文以利模型遵循。
SYSTEM_PROMPT_TEMPLATE = """\
You are a command-line expert.  The user is running {os_display}.

When the user describes a task in natural language, you must respond with a
JSON object (no markdown fences, no extra commentary) with exactly two keys:

  "command"     – the single shell command (or short pipeline) to accomplish
                  the task on {os_display}, suitable for copy-paste execution.
  "explanation" – a concise, plain-English explanation of what the command
                  does, including any side-effects the user should be aware of.

Rules:
- Use only commands that are available by default on {os_display}.
- For Windows use cmd.exe syntax; for macOS/Linux use bash/sh syntax.
- Never produce multiple alternative commands – pick the best single command.
- If the task cannot be safely expressed as a single command, chain with &&
  (Linux/macOS) or & (Windows).
- Output ONLY the raw JSON object, nothing else.
"""


def build_model(
    api_key: str, current_os: str, model_name: str
) -> genai.GenerativeModel:
    """
    Configure the SDK and return a GenerativeModel with the OS-specific system prompt.
    設定 SDK，並回傳已套用依作業系統客製化系統提示詞的 GenerativeModel。
    """
    genai.configure(api_key=api_key)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        os_display=OS_SHELL_NAME[current_os]
    )
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )


def _model_id_from_list_name(name: str) -> str:
    """Strip the ``models/`` prefix from API list names. / 移除 API 列舉名稱的 ``models/`` 前綴。"""
    return name.removeprefix("models/")


def fetch_models_via_sdk(api_key: str) -> list[str]:
    """
    All model ids from ``google.generativeai.list_models()`` for this key that
    advertise ``generateContent`` (SDK-dynamic list).
    以 ``list_models()`` 動態取得此 API 金鑰可用、且支援 ``generateContent`` 的模型 ID。
    """
    genai.configure(api_key=api_key)
    found: list[str] = []
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", None) or []
        if "generateContent" not in methods:
            continue
        found.append(_model_id_from_list_name(m.name))

    if not found:
        raise RuntimeError(
            "list_models() returned no models with generateContent for this key."
        )
    return sorted(set(found))


def resolve_api_key() -> str:
    """
    Read GEMINI_API_KEY from the environment, else prompt (hidden input).
    優先讀取環境變數 GEMINI_API_KEY；若無則以隱藏輸入提示貼上金鑰。
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    print()
    print(
        _colour(
            "GEMINI_API_KEY is not set. Paste your Gemini API key "
            "(https://aistudio.google.com/api-keys ).",
            YELLOW,
            BOLD,
        )
    )
    while True:
        try:
            key = getpass.getpass("API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if key:
            return key
        print(_colour("Key cannot be empty.", RED))


def select_model_interactive(eligible: list[str], default_id: str) -> str:
    """
    Numbered menu over *eligible*; empty input selects *default_id*.
    以編號選單從 *eligible* 選模型；空白輸入則選 *default_id*。
    """
    # Default first, then remaining ids alphabetically (no duplicate of default).
    # 預設模型置頂，其餘依字母排序（避免預設重複出現）。
    rest = sorted(m for m in eligible if m != default_id)
    ordered = [default_id] + rest

    print()
    print(
        _colour(
            "GEMINI_MODEL is not set. Choose a model (from genai.list_models() "
            "for this key):",
            BOLD,
        )
    )
    if default_id not in eligible:
        print(
            _colour(
                f"Note: default {default_id!r} was not returned by the API for "
                "this key; Enter still selects it.",
                YELLOW,
            )
        )

    for i, model_id in enumerate(ordered, start=1):
        mark = (
            " " + _colour("[default on Enter]", GREEN, BOLD)
            if model_id == default_id
            else ""
        )
        print(f"  {i}. {_colour(model_id, CYAN)}{mark}")

    n = len(ordered)
    while True:
        try:
            raw = input(
                _colour(
                    f"Choice [1-{n}, Enter = {default_id}]: ",
                    BOLD,
                )
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if raw == "":
            return default_id

        if not raw.isdigit():
            print(_colour("Enter a number, or press Enter for the default.", RED))
            continue

        choice = int(raw)
        if 1 <= choice <= n:
            return ordered[choice - 1]

        print(_colour(f"Pick a number from 1 to {n}.", RED))


def resolve_model_name(api_key: str) -> str:
    """
    Use GEMINI_MODEL if set; otherwise list_models() via the SDK and prompt.
    若已設定 GEMINI_MODEL 則直接使用；否則透過 SDK 列舉並互動選擇。
    """
    env_model = os.environ.get("GEMINI_MODEL", "").strip()
    if env_model:
        return env_model

    eligible = fetch_models_via_sdk(api_key)
    return select_model_interactive(eligible, DEFAULT_GEMINI_MODEL)


def ask_gemini(model: genai.GenerativeModel, prompt: str) -> tuple[str, str]:
    """
    Return (command, explanation) parsed from the Gemini response.
    Raises ValueError if the response cannot be parsed.
    Raises google.api_core.exceptions.GoogleAPIError for API-level failures.
    從 Gemini 回應解析並回傳 (指令, 說明)。無法解析時拋出 ValueError；
    API 層級錯誤則為 GoogleAPIError。
    """
    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Strip markdown fences if the model adds them despite instructions.
    # 若模型仍輸出 markdown 程式碼區塊，於此剝除包圍的 ```。
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned non-JSON output:\n{raw}"
        ) from exc

    command = data.get("command", "").strip()
    explanation = data.get("explanation", "").strip()

    if not command:
        raise ValueError("Gemini returned an empty command.")

    return command, explanation


# ─── Command execution / 指令執行 ─────────────────────────────────────────

def run_command(command: str, current_os: str) -> int:
    """
    Execute *command* in the appropriate shell; return the exit code.
    在對應 shell 執行 *command*，並回傳結束代碼。
    """
    if current_os == "windows":
        result = subprocess.run(["cmd.exe", "/C", command])
    else:
        result = subprocess.run(["/bin/bash", "-c", command])
    return result.returncode


# ─── Interactive loop / 互動主迴圈 ─────────────────────────────────────────

BANNER = """\
╔══════════════════════════════════════════════════════════════════╗
║              gemini-operator  –  AI Shell Assistant             ║
╚══════════════════════════════════════════════════════════════════╝
Type a task in natural language and press Enter.
Commands: 'exit' or 'quit' to leave, Ctrl-C to cancel at any time.
"""


def print_banner(current_os: str, model_name: str) -> None:
    """Print startup banner, detected OS, and model name. / 顯示啟動橫幅、偵測到的 OS 與模型名稱。"""
    print(_colour(BANNER, CYAN, BOLD))
    print(
        _colour("Detected OS: ", BOLD)
        + _colour(OS_DISPLAY[current_os], GREEN, BOLD)
    )
    print(
        _colour("Model: ", BOLD)
        + _colour(model_name, GREEN, BOLD)
        + "\n"
    )


def prompt_user_action(command: str) -> str:
    """
    Show the command to the user and ask what to do.
    Returns the (possibly edited) command to run, or '' to cancel.
    顯示指令並詢問動作；回傳要執行的指令（可經編輯），取消則回傳空字串。
    """
    print()
    print(_colour("┌─ Command to execute " + "─" * 45, CYAN))
    print(_colour("│  ", CYAN) + _colour(command, YELLOW, BOLD))
    print(_colour("└" + "─" * 66, CYAN))
    print(_colour("⚠  Review the command carefully before executing it.", YELLOW))
    print()
    print(
        _colour("[y]", GREEN) + " Execute   "
        + _colour("[n]", RED) + " Cancel   "
        + _colour("[e]", YELLOW) + " Edit command"
    )

    while True:
        try:
            choice = input(_colour("Your choice [y/n/e]: ", BOLD)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return ""

        if choice in ("y", "yes", ""):
            return command
        if choice in ("n", "no"):
            return ""
        if choice in ("e", "edit"):
            print(_colour(f"Current command: {command}", CYAN))
            try:
                edited = input(_colour("New command (leave blank to keep original): ", YELLOW)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return ""
            if not edited:
                print(_colour("No changes made; using original command.", YELLOW))
                return command
            return edited
        print(_colour("Please enter y, n, or e.", RED))


def main() -> None:
    # ── API key / API 金鑰 ───────────────────────────────────────────────────
    api_key = resolve_api_key()
    current_os = detect_os()

    try:
        selected_model = resolve_model_name(api_key)
    except google_api_exceptions.PermissionDenied:
        print(
            _colour(
                "Error: invalid API key while listing models (PermissionDenied). "
                "Check your GEMINI_API_KEY or pasted key.",
                RED,
                BOLD,
            )
        )
        sys.exit(1)
    except google_api_exceptions.GoogleAPIError as exc:
        print(_colour(f"API error while listing models: {exc}", RED))
        sys.exit(1)
    except (OSError, RuntimeError) as exc:
        print(_colour(f"Could not list models: {exc}", RED))
        sys.exit(1)

    try:
        model = build_model(api_key, current_os, selected_model)
    except google_api_exceptions.PermissionDenied:
        print(_colour("Error: invalid API key (PermissionDenied). Check GEMINI_API_KEY.", RED, BOLD))
        sys.exit(1)
    except google_api_exceptions.GoogleAPIError as exc:
        print(_colour(f"API error during initialisation: {exc}", RED))
        sys.exit(1)
    except (OSError, RuntimeError, ValueError) as exc:
        print(_colour(f"Failed to initialise Gemini model: {exc}", RED))
        sys.exit(1)

    print_banner(current_os, selected_model)

    # ── Main REPL / 主讀取–求值迴圈 ─────────────────────────────────────────
    while True:
        try:
            user_input = input(_colour("gemini-operator> ", MAGENTA, BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + _colour("Goodbye!", CYAN))
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print(_colour("Goodbye!", CYAN))
            break

        # ── Ask Gemini / 呼叫 Gemini ───────────────────────────────────────────
        print(_colour("⏳ Thinking…", CYAN))
        try:
            command, explanation = ask_gemini(model, user_input)
        except google_api_exceptions.PermissionDenied:
            print(_colour("API error: invalid or unauthorised API key. Check GEMINI_API_KEY.", RED))
            continue
        except google_api_exceptions.ResourceExhausted:
            print(_colour("API error: rate limit reached. Please wait and try again.", RED))
            continue
        except google_api_exceptions.GoogleAPIError as exc:
            print(_colour(f"API error: {exc}", RED))
            continue
        except ValueError as exc:
            print(_colour(f"Parse error: {exc}", RED))
            continue
        except (OSError, RuntimeError) as exc:
            print(_colour(f"Unexpected error: {exc}", RED))
            continue

        # ── Display explanation / 顯示說明 ─────────────────────────────────────
        print()
        print(_colour("ℹ  Explanation:", BOLD))
        print(f"   {explanation}")

        # ── Confirm & execute / 確認並執行 ───────────────────────────────────
        final_command = prompt_user_action(command)

        if not final_command:
            print(_colour("Cancelled.", YELLOW))
            continue

        print()
        print(_colour(f"▶ Running: {final_command}", GREEN, BOLD))
        print(_colour("─" * 68, CYAN))

        exit_code = run_command(final_command, current_os)

        print(_colour("─" * 68, CYAN))
        if exit_code == 0:
            print(_colour("✔ Command completed successfully.", GREEN))
        else:
            print(_colour(f"✘ Command exited with code {exit_code}.", RED))
        print()


if __name__ == "__main__":
    main()
