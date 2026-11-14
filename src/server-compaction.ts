import {
	type Api,
	type Message,
	type Model,
	type Usage,
	calculateCost,
} from "@earendil-works/pi-ai";
import {
	buildSessionProjection,
	convertToLlm,
	type ExtensionAPI,
	type ExtensionContext,
	type SessionBeforeCompactEvent,
	type SessionEntry,
	type SessionProjection,
} from "@earendil-works/pi-coding-agent";
import { isAnthropicOAuthToken } from "./cch.ts";
import { log } from "./logger.ts";

/**
 * Anthropic server-side on-demand compaction for subscription (OAuth) sessions.
 *
 * Pi's own compaction writes a text summary, and Anthropic rejects the thinking
 * in the turns kept after a summary it did not write (`thinking_dropped`,
 * `prefix_binding_mismatch`). Here Anthropic writes the summary instead: the
 * compaction request returns a signed `compaction` block, Pi stores it in the
 * compaction entry's `details`, and every later request sends the block in
 * place of Pi's summary message, so the kept thinking stays valid.
 *
 * Pi's system prompt has to leave `system[]` for plan billing (see
 * transforms.ts), and the API rejects kept thinking when that text lands inside
 * a kept turn. After a server compaction the prompt therefore goes to an
 * anchor: a fixed point right after the kept turns of the compaction that
 * placed it, the same on every later request and after a resume.
 */

export const COMPACT_BETA = "compact-2026-09-04";

/** Models the compaction docs list for on-demand compaction on the Claude API. */
export const SERVER_COMPACTION_MODELS: ReadonlySet<string> = new Set([
	"claude-fable-5-2",
	"claude-mythos-5-2",
	"claude-fable-5-1",
	"claude-mythos-5-1",
	"claude-fable-5",
	"claude-mythos-5",
	"claude-mythos-preview",
	"claude-opus-5-5",
	"claude-opus-5",
	"claude-sonnet-5-5",
	"claude-sonnet-5",
	"claude-haiku-5-5",
	"claude-opus-4-8",
	"claude-opus-4-7",
	"claude-opus-4-6",
	"claude-sonnet-4-6",
]);

/**
 * Placeholder text of the anchor message. It exists only between the
 * `context_with_system` hook and the payload transform, which replaces it with
 * the system prompt or removes it; it never reaches the API.
 */
export const ANCHOR_TEXT = "<pi-claude-auth:system-prompt-anchor/>";
const ANCHOR_CUSTOM_TYPE = "pi-claude-auth-system-anchor";

const MAX_INSTRUCTIONS_CHARS = 16_384;
const DISABLE_VALUES = new Set(["0", "false", "no", "off"]);

export interface CompactionBlock {
	type: "compaction";
	content: string;
	[key: string]: unknown;
}

/** `details` of a compaction entry written by this extension. */
export interface ServerCompactionDetails {
	anthropicCompaction: CompactionBlock;
	readFiles: string[];
	modifiedFiles: string[];
}

/** Kill switch: `PI_CLAUDE_AUTH_SERVER_COMPACTION=0` restores Pi's own compaction. */
export function serverCompactionEnabled(): boolean {
	const value = process.env.PI_CLAUDE_AUTH_SERVER_COMPACTION;
	return value === undefined || !DISABLE_VALUES.has(value.trim().toLowerCase());
}

export function supportsServerCompaction(modelId: unknown): boolean {
	return typeof modelId === "string" && SERVER_COMPACTION_MODELS.has(modelId);
}

function isCompactionBlock(value: unknown): value is CompactionBlock {
	if (!value || typeof value !== "object") return false;
	const block = value as { type?: unknown; content?: unknown };
	return (
		block.type === "compaction" &&
		typeof block.content === "string" &&
		block.content.length > 0
	);
}

/** The signed block of a compaction entry this extension wrote, if any. */
export function blockFromEntry(entry: SessionEntry | undefined): CompactionBlock | undefined {
	if (entry?.type !== "compaction") return undefined;
	const block = (entry.details as Partial<ServerCompactionDetails> | undefined)
		?.anthropicCompaction;
	return isCompactionBlock(block) ? block : undefined;
}

/**
 * The text of Pi's summary message, as it reaches the provider. Derived from
 * Pi's own converter so a change in Pi's wrapper text cannot desynchronize it.
 */
export function summaryMessageText(summary: string): string {
	const [message] = convertToLlm([
		{ role: "compactionSummary", summary, tokensBefore: 0, timestamp: 0 },
	] as never) as Message[];
	const content = message?.content;
	if (!Array.isArray(content)) return "";
	return content
		.map((block) => (block.type === "text" ? block.text : ""))
		.join("");
}

// Keyed by the wire text of the summary message the block replaces. Blocks are
// small and the text identifies its block exactly, so entries from any session
// in this process can share the map.
const blocksBySummaryText = new Map<string, CompactionBlock>();

export function rememberCompactionBlock(summary: string, block: CompactionBlock): void {
	blocksBySummaryText.set(summaryMessageText(summary), block);
}

export function rememberCompactionBlocks(entries: readonly SessionEntry[]): void {
	for (const entry of entries) {
		const block = blockFromEntry(entry);
		if (block && entry.type === "compaction")
			rememberCompactionBlock(entry.summary, block);
	}
}

export function forgetCompactionBlocks(): void {
	blocksBySummaryText.clear();
}

// ---------------------------------------------------------------------------
// Provider payload
// ---------------------------------------------------------------------------

type WireBlock = {
	type?: string;
	text?: string;
	cache_control?: unknown;
} & Record<string, unknown>;

interface WireMessage {
	role?: string;
	content?: string | WireBlock[];
	output_config?: { effort?: unknown } & Record<string, unknown>;
}

interface WirePayload {
	model?: unknown;
	messages?: unknown;
	betas?: unknown;
	[key: string]: unknown;
}

/** Where `injectBillingHeader` puts Pi's relocated system text. */
export type SystemTextPlacement = "first-user" | "anchor" | "drop";

function messageText(message: { content?: unknown } | undefined): string | undefined {
	const content = message?.content;
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return undefined;
	if (!content.every((block) => block?.type === "text")) return undefined;
	return content.map((block) => block.text ?? "").join("");
}

/**
 * Whether a provider context carries a summary message whose signed block is
 * known. Such requests must not use Anthropic message threads: the API answers
 * `thread_unsupported_request` for every request that carries a block.
 */
export function contextCarriesCompactionBlock(messages: readonly unknown[]): boolean {
	if (!serverCompactionEnabled() || blocksBySummaryText.size === 0) return false;
	if (!Array.isArray(messages)) return false;
	return messages.some((message) => {
		const m = message as { role?: string; content?: unknown };
		if (m.role !== "user") return false;
		const text = messageText(m);
		return text !== undefined && blocksBySummaryText.has(text);
	});
}

export function isAnchorBlock(block: unknown): boolean {
	return (
		!!block &&
		typeof block === "object" &&
		(block as WireBlock).type === "text" &&
		(block as WireBlock).text === ANCHOR_TEXT
	);
}

function isEffortMarker(message: WireMessage | undefined): boolean {
	return (
		message?.role === "system" &&
		Array.isArray(message.content) &&
		message.content.length === 0 &&
		typeof message.output_config?.effort === "string"
	);
}

/** Remove every anchor block, and messages left empty by that. */
export function stripAnchors(payload: unknown): void {
	const messages = (payload as WirePayload | undefined)?.messages;
	if (!Array.isArray(messages)) return;
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i] as WireMessage;
		if (!Array.isArray(message?.content)) continue;
		if (!message.content.some(isAnchorBlock)) continue;
		message.content = message.content.filter((block) => !isAnchorBlock(block));
		if (message.content.length === 0) messages.splice(i, 1);
	}
}

function addBeta(payload: WirePayload, beta: string): void {
	const betas = Array.isArray(payload.betas) ? (payload.betas as unknown[]) : [];
	if (!betas.includes(beta)) payload.betas = [...betas, beta];
}

// Compaction requests are marked by identity: the payload object leaves this
// extension's onPayload and reaches the overlay's transform unchanged, and a
// marker field would have to be removed before serialization.
const compactionRequestPayloads = new WeakSet<object>();

export function markCompactionRequestPayload(payload: object): void {
	compactionRequestPayloads.add(payload);
}

/**
 * Rewrite one OAuth payload for server compaction, before the billing
 * transform runs. Returns where the billing transform must put the system text.
 *
 * - Pi's summary message at the start becomes the signed block. When the kept
 *   turns start with an assistant message (Pi cut inside a turn), the effort
 *   marker pi-ai put before it is dropped: the API forbids a `system` message
 *   as the first kept message.
 * - With a block, the system text goes to the anchor. A compaction request
 *   without an anchor drops it: adding the text to a summarized message would
 *   change the prefix the kept thinking was produced with.
 * - Without a block, anchors are removed and the system text goes to the first
 *   user message, as before.
 */
export function applyServerCompactionPayload(payload: unknown): SystemTextPlacement {
	if (!payload || typeof payload !== "object") return "first-user";
	const p = payload as WirePayload;
	const messages = p.messages as WireMessage[] | undefined;
	if (!Array.isArray(messages)) return "first-user";
	const compactionRequest = compactionRequestPayloads.has(p);

	const first = messages[0];
	const summaryText = first?.role === "user" ? messageText(first) : undefined;
	const block =
		serverCompactionEnabled() && supportsServerCompaction(p.model) && summaryText !== undefined
			? blocksBySummaryText.get(summaryText)
			: undefined;
	if (!block) {
		stripAnchors(p);
		if (compactionRequest) addBeta(p, COMPACT_BETA);
		return "first-user";
	}

	const summaryBlock = Array.isArray(first.content) ? first.content.at(-1) : undefined;
	const cacheControl = summaryBlock?.cache_control;
	messages[0] = {
		role: "assistant",
		content: [
			cacheControl === undefined ? { ...block } : { ...block, cache_control: cacheControl },
		],
	};
	if (isEffortMarker(messages[1]) && messages[2]?.role === "assistant")
		messages.splice(1, 1);
	addBeta(p, COMPACT_BETA);

	const anchors: Array<{ message: number; block: number }> = [];
	messages.forEach((message, i) => {
		if (!Array.isArray(message.content)) return;
		message.content.forEach((b, j) => {
			if (isAnchorBlock(b)) anchors.push({ message: i, block: j });
		});
	});
	if (anchors.length === 0) return compactionRequest ? "drop" : "first-user";
	if (anchors.length > 1) {
		// More than one anchor means the context hook misplaced one; none of
		// them can be trusted to match earlier requests.
		stripAnchors(p);
		log("server_compaction_anchor_ambiguous", { count: anchors.length });
		return compactionRequest ? "drop" : "first-user";
	}
	moveAnchorPastToolResults(messages, anchors[0].message);
	return "anchor";
}

/**
 * A standalone anchor message right before tool results would separate them
 * from their tool call, which the API rejects. Put the anchor block after the
 * results instead.
 */
function moveAnchorPastToolResults(messages: WireMessage[], index: number): void {
	const anchor = messages[index];
	const next = messages[index + 1];
	if (anchor?.role !== "user" || next?.role !== "user") return;
	if (!Array.isArray(anchor.content) || anchor.content.length !== 1) return;
	if (!Array.isArray(next.content) || next.content[0]?.type !== "tool_result") return;
	const insertAt = next.content.findIndex((block) => block.type !== "tool_result");
	next.content.splice(insertAt < 0 ? next.content.length : insertAt, 0, anchor.content[0]);
	messages.splice(index, 1);
}

// ---------------------------------------------------------------------------
// Anchor placement in the session
// ---------------------------------------------------------------------------

/**
 * Index of the projected entry the anchor goes before (the projection's length
 * when it goes last), or undefined when the newest compaction carries no block.
 *
 * Each server compaction places its anchor right after itself in the session
 * path, which in the projection is right after its kept turns. A later server
 * compaction that keeps the active anchor's compaction keeps that anchor:
 * requests since then carried the system text there. A Pi summary resets the
 * chain, since the requests after it carried the system text in that summary.
 */
export function findAnchorIndex(
	branch: readonly SessionEntry[],
	projection: SessionProjection,
): number | undefined {
	const pathIndex = new Map(branch.map((entry, index) => [entry.id, index]));
	let active: SessionEntry | undefined;
	for (const entry of branch) {
		if (entry.type !== "compaction") continue;
		if (!blockFromEntry(entry)) {
			active = undefined;
			continue;
		}
		const firstKept = pathIndex.get(entry.firstKeptEntryId);
		const activeIndex = active ? pathIndex.get(active.id) : undefined;
		if (activeIndex !== undefined && firstKept !== undefined && activeIndex >= firstKept)
			continue;
		active = entry;
	}
	const newest = projection.entries[0]?.sourceEntry;
	if (!active || !blockFromEntry(newest)) return undefined;
	const activeIndex = pathIndex.get(active.id);
	if (activeIndex === undefined) return undefined;
	for (let i = 1; i < projection.entries.length; i++) {
		const index = pathIndex.get(projection.entries[i].sourceEntry.id);
		if (index !== undefined && index > activeIndex) return i;
	}
	return projection.entries.length;
}

function createAnchorMessage(timestamp: number) {
	return {
		role: "custom" as const,
		customType: ANCHOR_CUSTOM_TYPE,
		content: [{ type: "text" as const, text: ANCHOR_TEXT }],
		display: false,
		timestamp,
	};
}

interface TimedMessage {
	role?: string;
	timestamp?: number;
}

function sameMessage(a: TimedMessage, b: TimedMessage): boolean {
	return a.role === b.role && a.timestamp === b.timestamp;
}

/**
 * Insert the anchor into the messages of one agent request. Those are copies
 * of the projection after other extensions' context handlers, so the anchor is
 * located by the neighbouring messages' role and timestamp. Returns undefined
 * when the position cannot be found; the request then relocates the system
 * text the old way.
 */
export function insertAnchor<T extends TimedMessage>(
	messages: readonly T[],
	projection: SessionProjection,
	anchorIndex: number,
): T[] | undefined {
	const summaryIndex = messages.findIndex((m) => m.role === "compactionSummary");
	if (summaryIndex < 0) return undefined;
	const kept = projection.entries
		.slice(1, anchorIndex)
		.flatMap((entry) => entry.messages as TimedMessage[])
		.filter((m) => m.role !== "system");
	const next = projection.entries
		.slice(anchorIndex)
		.flatMap((entry) => entry.messages as TimedMessage[])[0];
	let at = -1;
	if (next) at = messages.findIndex((m, i) => i > summaryIndex && sameMessage(m, next));
	if (at < 0) {
		const previous = kept.at(-1);
		if (previous) {
			const found = messages.findLastIndex(
				(m, i) => i > summaryIndex && sameMessage(m, previous),
			);
			if (found >= 0) at = found + 1;
		} else {
			at = summaryIndex + 1;
		}
	}
	if (at < 0) return undefined;
	const anchor = createAnchorMessage(messages[at - 1]?.timestamp ?? Date.now());
	return [...messages.slice(0, at), anchor as unknown as T, ...messages.slice(at)];
}

// ---------------------------------------------------------------------------
// Compaction request
// ---------------------------------------------------------------------------

const SUMMARY_FORMAT = `## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]`;

const INSTRUCTIONS_HEAD = `Summarize the conversation above into a structured context checkpoint. The assistant will continue the work from this summary and the messages that follow it, so keep everything needed to continue.

If the conversation starts with an earlier summary, merge it into the new one: keep all of its information, add new progress, decisions, and context, move finished items from In Progress to Done, and update Next Steps.

If the conversation ends partway through a task, record the original request and the progress made so far. Later messages are kept separately; do not reconstruct them.

Use this EXACT format:

${SUMMARY_FORMAT}

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

const INSTRUCTIONS_TAIL = "Do not call tools. Respond with the summary text only.";

/**
 * The `compaction.instructions` text: Pi's summary format, the files Pi would
 * list, and the `/compact` focus, capped at the API's 16,384 characters. File
 * lists are shortened first, then the focus text.
 *
 * The files are named in prose. Instructions that ask for Pi's
 * `<read-files>` blocks, or for a file section copied exactly, were refused
 * outright in live tests (`stop_reason: "refusal"` with zero usage), while
 * the same paths in a sentence were accepted.
 */
export function buildCompactionInstructions(options: {
	customInstructions?: string;
	readFiles?: readonly string[];
	modifiedFiles?: readonly string[];
}): string {
	const focus = options.customInstructions?.trim();
	const focusText = focus ? `Additional focus: ${focus}` : "";
	const assemble = (lists: string, focusPart: string) =>
		[INSTRUCTIONS_HEAD, lists, INSTRUCTIONS_TAIL, focusPart]
			.filter((part) => part.length > 0)
			.join("\n\n");

	let readFiles = [...(options.readFiles ?? [])];
	let modifiedFiles = [...(options.modifiedFiles ?? [])];
	const lists = () => {
		const lines = [
			readFiles.length > 0 ? `a line listing the files read (${readFiles.join(", ")})` : "",
			modifiedFiles.length > 0
				? `a line listing the files modified (${modifiedFiles.join(", ")})`
				: "",
		].filter((line) => line.length > 0);
		return lines.length > 0 ? `End the summary with ${lines.join(" and ")}.` : "";
	};
	const focusBudget = Math.min(focusText.length, MAX_INSTRUCTIONS_CHARS - assemble("", "").length - 2);
	const focusPart = focusText.slice(0, Math.max(0, focusBudget));
	while (assemble(lists(), focusPart).length > MAX_INSTRUCTIONS_CHARS) {
		if (readFiles.length >= modifiedFiles.length && readFiles.length > 0) readFiles = readFiles.slice(0, -1);
		else if (modifiedFiles.length > 0) modifiedFiles = modifiedFiles.slice(0, -1);
		else break;
	}
	return assemble(lists(), focusPart).slice(0, MAX_INSTRUCTIONS_CHARS);
}

function isAnthropicEffort(value: unknown): value is string {
	return (
		value === "low" || value === "medium" || value === "high" || value === "xhigh" || value === "max"
	);
}

/**
 * The messages of the compaction request: the request Pi would send, cut at
 * the first kept entry, with the active anchor when it lies before the cut.
 * `keptAssistantEffort` is set when the kept part starts with an assistant
 * message whose effort pi-ai marks: the request then ends with that marker, so
 * it equals the request that produced the kept message.
 */
export function buildCompactionRequestMessages(
	projection: SessionProjection,
	firstKeptEntryId: string,
	anchorIndex: number | undefined,
	provider: string,
): { messages: unknown[]; keptAssistantEffort?: string } | undefined {
	const cut = projection.entries.findIndex((entry) => entry.sourceEntry.id === firstKeptEntryId);
	if (cut <= 0) return undefined;
	const messages: unknown[] = projection.entries.slice(0, cut).flatMap((entry) => entry.messages);
	if (anchorIndex !== undefined && anchorIndex <= cut) {
		const offset = projection.entries
			.slice(0, anchorIndex)
			.reduce((count, entry) => count + entry.messages.length, 0);
		const before = messages[offset - 1] as TimedMessage | undefined;
		messages.splice(offset, 0, createAnchorMessage(before?.timestamp ?? Date.now()));
	}
	const kept = projection.entries[cut]?.messages[0] as
		| { role?: string; api?: string; provider?: string; providerThinkingLevel?: unknown }
		| undefined;
	const keptAssistantEffort =
		kept?.role === "assistant" &&
		kept.api === "anthropic-messages" &&
		kept.provider === provider &&
		isAnthropicEffort(kept.providerThinkingLevel)
			? kept.providerThinkingLevel
			: undefined;
	return { messages, ...(keptAssistantEffort ? { keptAssistantEffort } : {}) };
}

interface CompactionIteration {
	type?: string;
	input_tokens?: number;
	output_tokens?: number;
	cache_read_input_tokens?: number;
	cache_creation_input_tokens?: number;
}

function usageFromIteration(model: Model<Api>, iteration: CompactionIteration | undefined): Usage {
	const usage: Usage = {
		input: iteration?.input_tokens ?? 0,
		output: iteration?.output_tokens ?? 0,
		cacheRead: iteration?.cache_read_input_tokens ?? 0,
		cacheWrite: iteration?.cache_creation_input_tokens ?? 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
	usage.totalTokens = usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
	usage.cost = calculateCost(model, usage);
	return usage;
}

export type CompactionOutcome =
	| { ok: true; block: CompactionBlock; usage: Usage }
	| { ok: false; reason: string; stopReason?: string };

type StreamSimple = ExtensionContext["modelRegistry"]["streamSimple"];

/**
 * Send one compaction request through the provider path of a normal turn
 * (pi-claude-auth's overlay included) and return the signed block. pi-ai does
 * not know the `compaction` stop reason and ends the stream with an error; the
 * block, stop reason, and usage are read from the raw stream events instead.
 */
export async function requestServerCompaction(options: {
	streamSimple: StreamSimple;
	model: Model<Api>;
	apiKey?: string;
	headers?: Record<string, string>;
	sessionId?: string;
	reasoning?: string;
	messages: Message[];
	keptAssistantEffort?: string;
	instructions: string;
	signal?: AbortSignal;
}): Promise<CompactionOutcome> {
	let block: CompactionBlock | undefined;
	let stopReason: string | undefined;
	let stopDetails: unknown;
	let iteration: CompactionIteration | undefined;
	const reasoning =
		options.model.reasoning && options.reasoning && options.reasoning !== "off"
			? options.reasoning
			: undefined;
	const stream = options.streamSimple(
		options.model,
		{ messages: options.messages },
		{
			...(options.apiKey ? { apiKey: options.apiKey } : {}),
			...(options.headers ? { headers: options.headers } : {}),
			...(options.sessionId ? { sessionId: options.sessionId } : {}),
			...(reasoning ? { reasoning } : {}),
			signal: options.signal,
			disableContinuation: true,
			onPayload: (params: unknown) => {
				const p = params as WirePayload;
				const messages = p.messages as WireMessage[] | undefined;
				if (Array.isArray(messages) && isEffortMarker(messages.at(-1))) {
					const marker = messages.at(-1) as WireMessage;
					if (options.keptAssistantEffort)
						marker.output_config = { ...marker.output_config, effort: options.keptAssistantEffort };
					else messages.pop();
				}
				p.compaction = { type: "summarize", instructions: options.instructions };
				addBeta(p, COMPACT_BETA);
				markCompactionRequestPayload(p);
				return p;
			},
			onProviderStreamEvent: (event: unknown) => {
				const e = event as {
					type?: string;
					content_block?: unknown;
					delta?: { stop_reason?: string; stop_details?: unknown };
					usage?: { iterations?: CompactionIteration[] };
				};
				if (e.type === "content_block_start" && isCompactionBlock(e.content_block))
					block = structuredClone(e.content_block);
				if (e.type === "message_delta") {
					stopReason = e.delta?.stop_reason ?? stopReason;
					stopDetails = e.delta?.stop_details ?? stopDetails;
					iteration = e.usage?.iterations?.find((it) => it.type === "compaction") ?? iteration;
				}
			},
		} as never,
	);
	const result = await stream.result();
	if (options.signal?.aborted) return { ok: false, reason: "aborted" };
	if (stopReason !== "compaction" || !block) {
		const details = stopDetails === undefined ? "" : ` ${JSON.stringify(stopDetails)}`;
		return {
			ok: false,
			reason: stopReason ? `${stopReason}${details}` : (result.errorMessage ?? "no compaction block"),
			...(stopReason ? { stopReason } : {}),
		};
	}
	return { ok: true, block, usage: usageFromIteration(options.model, iteration) };
}

function eligibleModel(model: Model<Api> | undefined): model is Model<Api> {
	return (
		model?.provider === "anthropic" &&
		model.api === "anthropic-messages" &&
		supportsServerCompaction(model.id)
	);
}

function fileLists(fileOps: SessionBeforeCompactEvent["preparation"]["fileOps"]) {
	const modified = new Set([...fileOps.edited, ...fileOps.written]);
	return {
		readFiles: [...fileOps.read].filter((file) => !modified.has(file)).sort(),
		modifiedFiles: [...modified].sort(),
	};
}

async function compactOnServer(event: SessionBeforeCompactEvent, ctx: ExtensionContext) {
	if (!serverCompactionEnabled()) return undefined;
	const model = ctx.model;
	if (!eligibleModel(model) || event.signal.aborted) return undefined;
	const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
	if (!auth.ok || !isAnthropicOAuthToken(auth.apiKey)) return undefined;

	const { preparation } = event;
	const fallback = (reason: string) => {
		log("server_compaction_fallback", { reason });
		if (ctx.hasUI && reason !== "aborted")
			ctx.ui.notify(
				`Anthropic compaction did not return a summary (${reason.slice(0, 200)}). Using Pi's summary instead.`,
				"warning",
			);
		return undefined;
	};

	rememberCompactionBlocks(event.branchEntries);
	const projection = buildSessionProjection(event.branchEntries);
	const anchorIndex = findAnchorIndex(event.branchEntries, projection);
	const request = buildCompactionRequestMessages(
		projection,
		preparation.firstKeptEntryId,
		anchorIndex,
		model.provider,
	);
	if (!request) return fallback("first kept entry not found");

	const { readFiles, modifiedFiles } = fileLists(preparation.fileOps);
	const attempt = async (instructions: string): Promise<CompactionOutcome> => {
		try {
			return await requestServerCompaction({
				streamSimple: ctx.modelRegistry.streamSimple.bind(ctx.modelRegistry),
				model,
				apiKey: auth.apiKey,
				headers: auth.headers as Record<string, string> | undefined,
				sessionId: ctx.sessionManager.getSessionId(),
				reasoning: ctx.thinkingLevel,
				messages: convertToLlm(request.messages as never),
				keptAssistantEffort: request.keptAssistantEffort,
				instructions,
				signal: event.signal,
			});
		} catch (err) {
			return { ok: false, reason: err instanceof Error ? err.message : String(err) };
		}
	};
	let outcome = await attempt(
		buildCompactionInstructions({
			customInstructions: event.customInstructions,
			readFiles,
			modifiedFiles,
		}),
	);
	// A refused request generates nothing and costs nothing; the file lists are
	// the part of the instructions most likely to be refused, so retry once
	// without them.
	if (!outcome.ok && outcome.stopReason === "refusal" && readFiles.length + modifiedFiles.length > 0) {
		log("server_compaction_retry", { reason: outcome.reason });
		outcome = await attempt(
			buildCompactionInstructions({ customInstructions: event.customInstructions }),
		);
	}
	if (!outcome.ok) return fallback(outcome.reason);

	rememberCompactionBlock(outcome.block.content, outcome.block);
	log("server_compaction", {
		model: model.id,
		splitTurn: preparation.isSplitTurn,
		summaryChars: outcome.block.content.length,
	});
	const details: ServerCompactionDetails = {
		anthropicCompaction: outcome.block,
		readFiles,
		modifiedFiles,
	};
	return {
		compaction: {
			summary: outcome.block.content,
			firstKeptEntryId: preparation.firstKeptEntryId,
			tokensBefore: preparation.tokensBefore,
			usage: outcome.usage,
			details,
		},
	};
}

/**
 * Insert the anchor into an agent request after a server compaction. Only
 * OAuth requests to models that accept the block get it: the overlay replaces
 * it with the system text or strips it, and API-key requests bypass the
 * overlay's transforms.
 */
async function anchorAgentRequest<T extends TimedMessage>(
	messages: T[],
	ctx: ExtensionContext,
): Promise<T[] | undefined> {
	const model = ctx.model;
	if (!serverCompactionEnabled() || !eligibleModel(model)) return undefined;
	const branch = ctx.sessionManager.getBranch();
	rememberCompactionBlocks(branch);
	const projection = ctx.sessionManager.buildSessionProjection();
	const anchorIndex = findAnchorIndex(branch, projection);
	if (anchorIndex === undefined) return undefined;
	const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
	if (!auth.ok || !isAnthropicOAuthToken(auth.apiKey)) return undefined;
	const anchored = insertAnchor(messages, projection, anchorIndex);
	if (!anchored) log("server_compaction_anchor_missing", {});
	return anchored;
}

export function registerServerCompaction(pi: ExtensionAPI): void {
	pi.on("session_start", (_event, ctx) => {
		rememberCompactionBlocks(ctx.sessionManager.getEntries());
	});
	pi.on("session_tree", (_event, ctx) => {
		rememberCompactionBlocks(ctx.sessionManager.getBranch());
	});
	pi.on("session_compact", (event) => {
		rememberCompactionBlocks([event.compactionEntry]);
	});
	pi.on("session_before_compact", async (event, ctx) => {
		try {
			return await compactOnServer(event, ctx);
		} catch (err) {
			log("server_compaction_error", {
				error: err instanceof Error ? err.message : String(err),
			});
			return undefined;
		}
	});
	pi.on("context_with_system", async (event, ctx) => {
		try {
			const messages = await anchorAgentRequest(event.messages, ctx);
			return messages ? { messages } : undefined;
		} catch (err) {
			log("server_compaction_anchor_error", {
				error: err instanceof Error ? err.message : String(err),
			});
			return undefined;
		}
	});
}
