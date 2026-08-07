import {
	buildBillingHeaderValue,
	getCliVersion,
	getEntrypoint,
} from "./signing.ts";

const BILLING_PREFIX = "x-anthropic-billing-header";
// Identity line pi's built-in provider emits for OAuth tokens (our trigger).
const LEGACY_IDENTITY =
	"You are Claude Code, Anthropic's official CLI for Claude.";
// Identity line current Claude Code (2.1.224) sends; we substitute this for
// pi's legacy line so the request matches real CC traffic.
const AGENT_SDK_IDENTITY =
	"You are a Claude agent, built on Anthropic's Claude Agent SDK.";

type SystemEntry = { type?: string; text?: string } & Record<string, unknown>;

interface AnthropicPayload {
	model?: unknown;
	system?: unknown;
	messages?: unknown;
}

function isClaudeModel(model: unknown): model is string {
	return typeof model === "string" && model.toLowerCase().includes("claude");
}

function entryText(entry: unknown): string {
	if (typeof entry === "string") return entry;
	if (entry && typeof entry === "object") {
		const text = (entry as { text?: unknown }).text;
		if (typeof text === "string") return text;
	}
	return "";
}

/**
 * Transform an Anthropic OAuth request so Anthropic bills it to the Claude
 * Pro/Max subscription instead of pay-as-you-go / "extra usage".
 *
 * pi opens OAuth requests with its legacy Claude Code identity at system[0],
 * followed by whatever system prompt the agent supplied. This function:
 *
 * - Prepends the `x-anthropic-billing-header` block as system[0].
 * - Swaps pi's legacy identity for the Agent SDK identity Claude Code 2.1.224
 *   sends (system[1]).
 * - Relocates every other system block (pi's prompt, other extensions' prompts)
 *   into the first user message. This move is mandatory: a live test confirmed
 *   that leaving any third-party system block in system[] makes Anthropic bill
 *   the request as extra usage.
 *
 * Returns the mutated payload when transformed, or undefined to leave it
 * unchanged (non-Claude requests, non-OAuth requests, already-transformed).
 */
export function injectBillingHeader(
	payload: unknown,
): AnthropicPayload | undefined {
	if (!payload || typeof payload !== "object") return undefined;

	const p = payload as AnthropicPayload;
	if (!isClaudeModel(p.model)) return undefined;
	if (!Array.isArray(p.messages)) return undefined;

	const system: SystemEntry[] = Array.isArray(p.system)
		? (p.system as SystemEntry[])
		: [];

	// Already transformed (billing header present): no-op.
	if (system.some((e) => entryText(e).startsWith(BILLING_PREFIX))) {
		return undefined;
	}
	// Only act in OAuth mode, signalled by pi's legacy identity. A plain
	// API-key request has no such block and passes through unchanged.
	if (!system.some((e) => entryText(e) === LEGACY_IDENTITY)) return undefined;

	const messages = p.messages as Array<{
		role?: string;
		content?: string | Array<{ type?: string; text?: string }>;
	}>;
	const billingHeader = buildBillingHeaderValue(
		messages,
		getCliVersion(),
		getEntrypoint(),
	);

	// Drop pi's legacy identity and relocate every other system block (pi's real
	// prompt, etc.) into the first user message.
	const movedTexts: string[] = [];
	for (const entry of system) {
		const txt = entryText(entry);
		if (txt.length > 0 && txt !== LEGACY_IDENTITY) movedTexts.push(txt);
	}

	p.system = [
		{ type: "text", text: billingHeader },
		{ type: "text", text: AGENT_SDK_IDENTITY },
	];

	if (movedTexts.length > 0) {
		const firstUser = messages.find((m) => m.role === "user");
		if (firstUser) {
			const prefix = movedTexts.join("\n\n");
			const content = firstUser.content;
			if (typeof content === "string") {
				firstUser.content = `${prefix}\n\n${content}`;
			} else if (Array.isArray(content)) {
				content.unshift({ type: "text", text: prefix });
			}
		}
	}

	return p;
}
