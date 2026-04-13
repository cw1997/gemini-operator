# gemini-operator

A command-line AI shell assistant powered by **Google Gemini**.  
Describe a task in plain language and gemini-operator will:

1. Convert it into the right shell command for your OS (Windows `cmd`, Linux `bash`, or macOS `zsh/bash`).
2. Show you the command and **explain** what it will do.
3. Ask for confirmation before running it.
4. Let you **edit** the command before execution.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| google-generativeai | 0.8+ |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain a Gemini API key

Visit [Google AI Studio](https://aistudio.google.com/app/apikey) and create a free API key.

### 3. Export the key as an environment variable

**Linux / macOS**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Windows (cmd)**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

**Windows (PowerShell)**
```powershell
$env:GEMINI_API_KEY = "your-api-key-here"
```

---

## Usage

```bash
python gemini_operator.py
```

You will see a prompt:

```
gemini-operator>
```

Type any task in your language and press **Enter**. For example:

```
gemini-operator> Open Chrome and navigate to YouTube
gemini-operator> Create a file called test.txt on the Desktop with the content "Today's Weather"
gemini-operator> List all running processes and sort by memory usage
gemini-operator> Show disk usage for the current directory
```

### Confirmation prompt

Before each command is executed you will see:

```
┌─ Command to execute ─────────────────────────────────────────────
│  google-chrome https://www.youtube.com
└──────────────────────────────────────────────────────────────────

[y] Execute   [n] Cancel   [e] Edit command
Your choice [y/n/e]:
```

| Key | Action |
|-----|--------|
| `y` (or Enter) | Execute the command as shown |
| `n` | Cancel — do not run anything |
| `e` | Edit the command in the terminal, then execute the edited version |

Type `exit` or `quit` (or press **Ctrl-C**) to leave the program.

---

## Example session

```
gemini-operator> create a file called hello.txt on the Desktop containing the text "Hello World"

ℹ  Explanation:
   Creates the file ~/Desktop/hello.txt and writes the text "Hello World" into it.

┌─ Command to execute ─────────────────────────────────────────────
│  echo "Hello World" > ~/Desktop/hello.txt
└──────────────────────────────────────────────────────────────────

[y] Execute   [n] Cancel   [e] Edit command
Your choice [y/n/e]: y

▶ Running: echo "Hello World" > ~/Desktop/hello.txt
────────────────────────────────────────────────────────────────────
✔ Command completed successfully.
```

---

## Security note

`gemini-operator` executes shell commands directly on your machine.  
Always review the explanation and the generated command **before** confirming execution.  
Never run the tool as `root` / Administrator unless absolutely necessary.