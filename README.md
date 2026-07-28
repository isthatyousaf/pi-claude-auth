# pi-claude-auth

`pi-claude-auth` makes Pi's `anthropic` provider bill against your Claude Pro or Max subscription.

When Pi calls Anthropic with an OAuth token, Anthropic treats it as third-party harness traffic. It routes those requests through a separate "extra usage" bucket and bills them per token, outside your plan window. This extension sends the Claude Code billing header and identity, so Anthropic sees the request as Claude Code traffic and draws it from your plan quota.

Pi's built-in `anthropic` provider owns the OAuth lifecycle (browser login, token refresh, and credential storage in `~/.pi/agent/auth.json`). This extension adds only the pieces pi's provider does not send, so requests bill against your plan: the Claude Code user-agent header and the billing header, plus a retry on Anthropic classifier refusals. It deliberately does **not** register a custom OAuth lifecycle — doing so would overwrite pi's built-in provider and break `/login`. (Pi already prefers the stored OAuth token over any `ANTHROPIC_API_KEY` env var, so the extension never needs to touch credentials.)

## 🌐 **Join the Community**

> [!NOTE]
> **Building with AI doesn’t have to be a solo grind.**  
> Join our Discord community to meet other people exploring the latest models, tools, workflows, and ideas: **https://discord.gg/whhrDtCrSS**
>
> We talk about what’s new, what’s useful, and what’s actually worth paying attention to in AI.  
> *And if you want more than conversation,* members also get access to **heavily discounted AI products and services** — including deals on tools like **ChatGPT Plus** and more for just a few dollars.

## Install

```bash
pi install git:github.com/edxeth/pi-claude-auth
```

## How it works

Three things have to line up before Anthropic bills a request against your plan: the right OAuth token, the Claude Code identity in the system prompt, and the billing header that carries the version. Pi's built-in Anthropic provider handles the token plumbing (login, refresh, storage) and the identity. This extension adds the billing header and keeps the version current.

### Credentials

Run `/login anthropic`. Pi's built-in provider performs the browser OAuth flow and writes the `anthropic` entry to `~/.pi/agent/auth.json`:

```json
{ "anthropic": { "type": "oauth", "access": "…", "refresh": "…", "expires": 1750… } }
```

That file is the single source of truth; the extension never reads or writes it. Pi loads the OAuth entry at startup, and its `getApiKey` already prefers that token over any `ANTHROPIC_API_KEY` in your environment, so no credential handling is needed here. Token refresh is handled by pi's built-in provider (which mints and refreshes at Anthropic's `platform.claude.com/v1/oauth/token` endpoint); the extension does not refresh tokens itself.

### The billing header

Every request gets an `x-anthropic-billing-header` system block that carries the Claude Code version and entrypoint. That header is what routes billing to the subscription plan.

Pi's own system prompt gets relocated into the first user message. Anthropic rejects OAuth requests that carry third-party system prompts alongside the Claude Code identity, so the prompt has to move out of `system[]` to avoid a 400 "out of extra usage" rejection.

The `cch` token uses a simplified scheme. It works because Anthropic does not currently enforce `cch` validation. The day Anthropic starts enforcing it, requests will fail until the extension ships an update.

### Version sync

The billing version has to match current Claude Code, or Anthropic rejects the request. The extension resolves the latest `@anthropic-ai/claude-code` version from the npm registry at startup, caches it under `~/.pi/agent/claude-code-version.json`, and falls back to that cache when the registry is unreachable.

The version suffix is computed per request from the current Claude Code algorithm. It is not pinned to a fixed build hash.

If startup cannot reach npm and has no cache, Pi falls back to a built-in version and shows a red alert. If it falls back to a cached version, it shows a yellow alert. Both dismiss with Enter or Escape. Offline runs stay silent.

### Fable 5 and Opus 5 refusal handling

Claude Fable 5 and Opus 5 route some requests through safety classifiers. When a classifier blocks a turn, Anthropic returns the refusal as a finished message with `stop_reason: "refusal"` and an explanation. Pi maps that to `stopReason: "error"` with the explanation in `errorMessage`.

This extension pauses the TUI when a finalized Anthropic Fable 5 or Opus 5 assistant message looks like a classifier refusal. The user can switch to Claude Opus 4.8 and continue there, or branch to the exact point immediately before the refusal and type new steering instructions with the original model still selected.

This is not Anthropic server-side fallback. Server-side fallback sends one request with a `fallbacks` chain and lets Anthropic pick the model internally. The extension runs a separate Pi turn using the normalized refusal message Pi exposes.

#### What counts as a refusal

All five conditions must hold on the finalized assistant message:

- role is `assistant`
- provider is `anthropic`
- model id contains `claude-fable-5` or `claude-opus-5`
- `stopReason` is `error`
- `errorMessage` matches refusal wording (`refus`, `classifier`, `safety`, `safeguard`, `usage policy`, `violative`, or `refusals-and-fallback`)

Network timeouts and generic proxy errors fail the wording gate. Other model families fail the model gate. The Claude Opus 4.8 continuation also fails the model gate, so it cannot retrigger the workflow.

#### Interactive choices

**Continue with Claude Opus 4.8** waits until Pi has persisted the refusal, switches the active model, and sends a hidden custom message containing exactly `continue`. The fallback model sees the existing transcript and completed tool work, but the continuation message is not rendered to the user. Claude Opus 4.8 remains selected for later turns.

**Edit and retry** branches the session tree directly at `agent_end` to the safe point immediately before the event that triggered the refusal — skipping both that trigger event and the refusal itself. No command staging or extra keypress is needed. The user's in-progress editor draft is restored. A hidden extension entry makes the selected branch durable across session reopen, and every later `context` event rebuilds provider input from that active branch until Pi performs supported tree navigation or compaction. Reload restores that repair state from the marker. Completed work before the trigger remains on the active branch; the refused path stays available in `/tree`.

**Known limitation:** The visible TUI transcript does not refresh after direct branching because `sessionManager.branch()` (cast from the read-only type) cannot rebuild Pi's private `agent.state.messages` or chat component tree. `navigateTree()` — which synchronizes both — is only available on `ExtensionCommandContext`, not the `ExtensionContext` that event handlers receive. The model still receives the correct active-branch context. Manual `/tree` navigation, successful compaction, reload, or session replacement rebuilds the visible transcript.

Escape stops and leaves the refusal as the active leaf.

Set `PI_CLAUDE_AUTH_REFUSAL_MODE=auto` to skip the menu and always continue with Claude Opus 4.8. The default is `ask`. Non-TUI modes stop instead of silently choosing when the mode is `ask`.

#### Limits

The extension can recover only after Pi finishes the refusal and emits `message_end`. If the stream hangs first, there is no finalized refusal entry to handle.

Pi does not expose queue clearing to event handlers. Steering or follow-up messages already queued when the refusal finishes may still run automatically after the branch.

Edit and retry rewinds Pi's conversation/session context only. It does not undo filesystem changes, shell commands, network calls, or other external side effects already produced by the abandoned tool turn.

#### Deterministic mid-work refusal demo

From the repository root, use the fake Anthropic provider to exercise the critical multi-turn path without making an external model request:

```bash
PI_OFFLINE=1 \
PI_CLAUDE_AUTH_REFUSAL_MODE=ask \
pi --no-extensions \
  -e ./src/index.ts \
  -e ./test/fixtures/refusal-simulator.ts \
  --model anthropic/claude-fable-5 \
  --api-key refusal-simulator-local-only \
  --thinking high \
  --session-dir "/tmp/pi-refusal-demo-$(date +%s)"
```

Enter any prompt. The simulator produces this sequence:

1. Fable thinking + assistant text + baseline tool A
2. The completed result from tool A
3. More Fable thinking + assistant text + tool B
4. The completed result from tool B
5. More Fable thinking + assistant text + a separate tool C
6. The completed result from tool C
7. A classifier refusal

Choosing **Edit and retry** keeps everything through tool B's completed result, then abandons the assistant turn that called tool C, tool C's result, and the refusal. Enter a revised prompt to see Fable continue from immediately before tool C. Choosing **Continue** switches to the simulated Opus 4.8 and sends the hidden `continue` message. Run `/reload` after Edit to repaint the transcript from the active branch.

If B and C are emitted together inside one assistant message, Pi has no branch boundary between them. That assistant message and all of its tool results form one atomic batch, so Edit must discard the whole batch rather than leave an orphaned tool call or result.

### `/login anthropic`

`/login anthropic` works exactly the usual Pi way — the extension does not register a custom OAuth lifecycle, so pi's built-in `anthropic` provider stays in charge of the browser login, refresh, and storage. Log in once and pi persists the credential to `~/.pi/agent/auth.json`, which this extension then reads on every start.

## Environment variables

| Variable | Use |
| --- | --- |
| `ANTHROPIC_CLI_VERSION` | Override the Claude Code version. Must be valid semver, or the extension ignores it and uses the resolved version. |
| `CLAUDE_CODE_ENTRYPOINT` | Override the billing entrypoint mirrored in the user-agent suffix. |
| `ANTHROPIC_USER_AGENT` | Override the whole user-agent string. |
| `PI_CLAUDE_AUTH_DEBUG` | Set `1` for opt-in diagnostic logging to `~/.pi/agent/pi-claude-auth-debug.log`. Secrets are redacted before anything is written. |
| `PI_CLAUDE_AUTH_REFUSAL_MODE` | Refusal policy for Fable 5 and Opus 5: `ask` (default) or `auto`. |

## Credits

- upstream foundation: [pankajudhas81/pi-claude-auth](https://github.com/pankajudhas81/pi-claude-auth)
- this fork: [edxeth/pi-claude-auth](https://github.com/edxeth/pi-claude-auth)

## License

MIT
