import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import type { Api, Message, Model } from "@earendil-works/pi-ai";
import {
	buildSessionProjection,
	convertToLlm,
	type ExtensionAPI,
	type SessionEntry,
} from "@earendil-works/pi-coding-agent";
import {
	ANCHOR_TEXT,
	applyServerCompactionPayload,
	buildCompactionInstructions,
	buildCompactionRequestMessages,
	canonicalMessages,
	COMPACT_BETA,
	type CompactionBlock,
	contextCarriesCompactionBlock,
	findAnchorIndex,
	forgetAgentRequests,
	forgetCompactionBlocks,
	insertAnchor,
	markCompactionRequestPayload,
	payloadDeclarationsKey,
	recordAgentRequest,
	registerServerCompaction,
	rememberCompactionBlock,
	requestServerCompaction,
	stripAnchors,
	summaryMessageText,
	wireMessageRecords,
	wirePrefixMismatch,
} from "../src/server-compaction.ts";
import { injectBillingHeader } from "../src/transforms.ts";

const LEGACY_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude.";
const BLOCK: CompactionBlock = {
	type: "compaction",
	content: "## Goal\nPlan the release.",
	signature: "sig-abc",
	tool_changes: [{ kind: "kept-as-is" }],
};

const anchorMessage = () => ({
	role: "user",
	content: [{ type: "text", text: ANCHOR_TEXT }],
});
const marker = (effort: string) => ({
	role: "system",
	content: [],
	output_config: { effort },
});
const summaryMessage = (summary = BLOCK.content) => ({
	role: "user",
	content: [{ type: "text", text: summaryMessageText(summary) }],
});

// biome-ignore lint/suspicious/noExplicitAny: wire payloads are untyped JSON
function payload(messages: any[], model = "claude-haiku-5-5"): any {
	return {
		model,
		system: [
			{ type: "text", text: LEGACY_IDENTITY, cache_control: { type: "ephemeral" } },
			{ type: "text", text: "Pi system prompt.", cache_control: { type: "ephemeral" } },
		],
		messages,
		betas: ["oauth-2025-04-20"],
	};
}

beforeEach(() => {
	process.env.ANTHROPIC_CLI_VERSION = "1.2.3";
	forgetCompactionBlocks();
	rememberCompactionBlock(BLOCK.content, BLOCK);
});

afterEach(() => {
	delete process.env.ANTHROPIC_CLI_VERSION;
	delete process.env.PI_CLAUDE_AUTH_SERVER_COMPACTION;
	forgetCompactionBlocks();
	forgetAgentRequests();
});

describe("summary message text", () => {
	it("matches Pi's own conversion of a compaction summary", () => {
		const [converted] = convertToLlm([
			{ role: "compactionSummary", summary: "S", tokensBefore: 1, timestamp: 1 },
		] as never) as Array<{ content: Array<{ text: string }> }>;
		expect(summaryMessageText("S")).toBe(converted.content[0].text);
		expect(summaryMessageText("S")).toContain("<summary>\nS\n</summary>");
	});

	it("recognizes provider contexts that carry a known block", () => {
		expect(contextCarriesCompactionBlock([summaryMessage()])).toBe(true);
		expect(contextCarriesCompactionBlock([summaryMessage("other")])).toBe(false);
		expect(contextCarriesCompactionBlock(undefined as never)).toBe(false);
		process.env.PI_CLAUDE_AUTH_SERVER_COMPACTION = "0";
		expect(contextCarriesCompactionBlock([summaryMessage()])).toBe(false);
	});
});

describe("applyServerCompactionPayload", () => {
	it("swaps Pi's summary for the signed block exactly as returned and adds the beta", () => {
		const p = payload([
			summaryMessage(),
			{ role: "user", content: [{ type: "text", text: "kept" }] },
			anchorMessage(),
			{ role: "user", content: [{ type: "text", text: "new" }] },
		]);
		expect(applyServerCompactionPayload(p)).toBe("anchor");
		expect(p.messages[0]).toEqual({ role: "assistant", content: [BLOCK] });
		expect(p.betas).toEqual(["oauth-2025-04-20", COMPACT_BETA]);
	});

	it("keeps the summary's cache breakpoint on the block", () => {
		const summary = summaryMessage();
		(summary.content[0] as Record<string, unknown>).cache_control = { type: "ephemeral" };
		const p = payload([summary]);
		applyServerCompactionPayload(p);
		expect(p.messages[0]).toEqual({
			role: "assistant",
			content: [{ ...BLOCK, cache_control: { type: "ephemeral" } }],
		});
	});

	it("drops the effort marker when the kept turns start with an assistant message", () => {
		const assistant = { role: "assistant", content: [{ type: "text", text: "kept reply" }] };
		const p = payload([summaryMessage(), marker("high"), assistant, anchorMessage()]);
		applyServerCompactionPayload(p);
		expect(p.messages.slice(0, 2)).toEqual([{ role: "assistant", content: [BLOCK] }, assistant]);
	});

	it("keeps the effort marker that belongs to a later assistant message", () => {
		const user = { role: "user", content: [{ type: "text", text: "kept" }] };
		const p = payload([summaryMessage(), user, marker("high"), { role: "assistant", content: [] }]);
		applyServerCompactionPayload(p);
		expect(p.messages[2]).toEqual(marker("high"));
	});

	it("moves an anchor that sits before tool results behind them", () => {
		const results = {
			role: "user",
			content: [
				{ type: "tool_result", tool_use_id: "a" },
				{ type: "tool_result", tool_use_id: "b" },
				{ type: "text", text: "note" },
			],
		};
		const p = payload([summaryMessage(), { role: "assistant", content: [] }, anchorMessage(), results]);
		expect(applyServerCompactionPayload(p)).toBe("anchor");
		expect(p.messages).toHaveLength(3);
		expect(p.messages[2].content.map((b: { type: string; text?: string }) => b.text ?? b.type)).toEqual([
			"tool_result",
			"tool_result",
			ANCHOR_TEXT,
			"note",
		]);
	});

	it("leaves Pi's summary alone and strips anchors when the block is unknown", () => {
		const p = payload([summaryMessage("unknown"), anchorMessage(), { role: "user", content: "x" }]);
		expect(applyServerCompactionPayload(p)).toBe("first-user");
		expect(p.messages).toEqual([summaryMessage("unknown"), { role: "user", content: "x" }]);
		expect(p.betas).toEqual(["oauth-2025-04-20"]);
	});

	it("does not swap on models without on-demand compaction or with the kill switch", () => {
		const unsupported = payload([summaryMessage(), anchorMessage()], "claude-sonnet-4-5");
		expect(applyServerCompactionPayload(unsupported)).toBe("first-user");
		expect(unsupported.messages).toEqual([summaryMessage()]);

		process.env.PI_CLAUDE_AUTH_SERVER_COMPACTION = "0";
		const disabled = payload([summaryMessage(), anchorMessage()]);
		expect(applyServerCompactionPayload(disabled)).toBe("first-user");
		expect(disabled.messages).toEqual([summaryMessage()]);
	});

	it("drops the system text from a compaction request that starts with a block but has no anchor", () => {
		const request = payload([summaryMessage(), { role: "user", content: "kept" }]);
		markCompactionRequestPayload(request);
		expect(applyServerCompactionPayload(request)).toBe("drop");

		const turn = payload([summaryMessage(), { role: "user", content: "kept" }]);
		expect(applyServerCompactionPayload(turn)).toBe("first-user");
	});

	it("distrusts more than one anchor", () => {
		const p = payload([summaryMessage(), anchorMessage(), { role: "user", content: "x" }, anchorMessage()]);
		expect(applyServerCompactionPayload(p)).toBe("first-user");
		expect(JSON.stringify(p.messages)).not.toContain(ANCHOR_TEXT);
	});

	it("stripAnchors removes anchor blocks and the messages they leave empty", () => {
		const p = {
			messages: [
				anchorMessage(),
				{ role: "user", content: [{ type: "tool_result" }, { type: "text", text: ANCHOR_TEXT }] },
			],
		};
		stripAnchors(p);
		expect(p.messages).toEqual([{ role: "user", content: [{ type: "tool_result" }] }]);
	});
});

describe("system text placement in the billing transform", () => {
	it("puts the system text at the anchor, keeping the kept turns untouched", () => {
		const kept = { role: "user", content: [{ type: "text", text: "kept turn" }] };
		const p = payload([summaryMessage(), kept, anchorMessage(), { role: "user", content: "new" }]);
		const placement = applyServerCompactionPayload(p);
		const out = injectBillingHeader(p, { device: null, systemTextPlacement: placement }) as {
			system: Array<{ text: string }>;
			messages: Array<{ content: unknown }>;
		};
		expect(out.system).toHaveLength(2);
		expect(out.messages[1]).toEqual(kept);
		expect(out.messages[2].content).toEqual([
			{ type: "text", text: "Pi system prompt.", cache_control: { type: "ephemeral" } },
		]);
	});

	it("keeps the anchor's conversation breakpoint when the system text has none", () => {
		const p = payload([
			summaryMessage(),
			{ role: "user", content: [{ type: "text", text: ANCHOR_TEXT, cache_control: { type: "ephemeral", ttl: "1h" } }] },
		]);
		p.system[1] = { type: "text", text: "Pi system prompt." };
		applyServerCompactionPayload(p);
		const out = injectBillingHeader(p, { device: null, systemTextPlacement: "anchor" }) as {
			messages: Array<{ content: unknown }>;
		};
		expect(out.messages[1].content).toEqual([
			{ type: "text", text: "Pi system prompt.", cache_control: { type: "ephemeral", ttl: "1h" } },
		]);
	});

	it("drops the system text for a compaction request without an anchor", () => {
		const kept = { role: "user", content: [{ type: "text", text: "summarized turn" }] };
		const p = payload([summaryMessage(), kept]);
		const out = injectBillingHeader(p, { device: null, systemTextPlacement: "drop" }) as {
			messages: unknown[];
		};
		expect(JSON.stringify(out)).not.toContain("Pi system prompt.");
		expect(out.messages[1]).toEqual(kept);
	});

	it("falls back to the first user message when the anchor is missing", () => {
		const p = payload([{ role: "user", content: [{ type: "text", text: "hi" }] }]);
		const out = injectBillingHeader(p, { device: null, systemTextPlacement: "anchor" }) as {
			messages: Array<{ content: Array<{ text: string }> }>;
		};
		expect(out.messages[0].content.map((b) => b.text)).toEqual(["Pi system prompt.", "hi"]);
	});
});

// ---------------------------------------------------------------------------
// Session structure
// ---------------------------------------------------------------------------

let clock = 1_000;
function entries(...specs: Array<Record<string, unknown>>): SessionEntry[] {
	let parentId: string | null = null;
	return specs.map((spec) => {
		const entry = { parentId, timestamp: new Date(clock).toISOString(), ...spec };
		parentId = entry.id as string;
		return entry as unknown as SessionEntry;
	});
}
const sys = (id: string) => ({
	type: "message",
	id,
	message: { role: "system", content: "", sections: { a: "Pi system prompt." }, timestamp: clock++ },
});
const user = (id: string) => ({
	type: "message",
	id,
	message: { role: "user", content: [{ type: "text", text: id }], timestamp: clock++ },
});
const assistant = (id: string, effort = "high") => ({
	type: "message",
	id,
	message: {
		role: "assistant",
		content: [{ type: "text", text: id }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-haiku-5-5",
		providerThinkingLevel: effort,
		stopReason: "stop",
		usage: {},
		timestamp: clock++,
	},
});
const toolCallAssistant = (id: string) => {
	const entry = assistant(id);
	entry.message.content = [{ type: "toolCall", id: "call_" + id, name: "read", arguments: {} }] as never;
	(entry.message as { stopReason: string }).stopReason = "toolUse";
	return entry;
};
const toolResult = (id: string, callId: string) => ({
	type: "message",
	id,
	message: {
		role: "toolResult",
		toolCallId: "call_" + callId,
		toolName: "read",
		content: [{ type: "text", text: "result " + id }],
		isError: false,
		timestamp: clock++,
	},
});
const serverCompaction = (id: string, firstKeptEntryId: string, summary = BLOCK.content) => ({
	type: "compaction",
	id,
	summary,
	firstKeptEntryId,
	tokensBefore: 10,
	fromHook: true,
	details: { anthropicCompaction: { ...BLOCK, content: summary }, readFiles: [], modifiedFiles: [] },
});
const piCompaction = (id: string, firstKeptEntryId: string) => ({
	type: "compaction",
	id,
	summary: "pi summary",
	firstKeptEntryId,
	tokensBefore: 10,
});

function ids(projection: ReturnType<typeof buildSessionProjection>, from: number, to?: number) {
	return projection.entries.slice(from, to).map((entry) => entry.sourceEntry.id);
}

describe("findAnchorIndex", () => {
	it("has no anchor without a server compaction", () => {
		const branch = entries(sys("s"), user("u1"), assistant("a1"));
		expect(findAnchorIndex(branch, buildSessionProjection(branch))).toBeUndefined();
		const pi = entries(sys("s"), user("u1"), assistant("a1"), user("u2"), piCompaction("c", "u2"), user("u3"));
		expect(findAnchorIndex(pi, buildSessionProjection(pi))).toBeUndefined();
	});

	it("places the anchor after the kept turns of a server compaction", () => {
		const branch = entries(
			sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
			serverCompaction("c1", "u2"), user("u3"), assistant("a3"),
		);
		const projection = buildSessionProjection(branch);
		const index = findAnchorIndex(branch, projection) as number;
		expect(ids(projection, 1, index)).toEqual(["u2", "a2"]);
		expect(ids(projection, index)).toEqual(["u3", "a3"]);
	});

	it("places the anchor last when nothing follows the compaction yet", () => {
		const branch = entries(sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"), serverCompaction("c1", "u2"));
		const projection = buildSessionProjection(branch);
		expect(findAnchorIndex(branch, projection)).toBe(projection.entries.length);
	});

	it("keeps an earlier anchor that a later compaction keeps", () => {
		const branch = entries(
			sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
			serverCompaction("c1", "u2", "first"), user("u3"), assistant("a3"),
			serverCompaction("c2", "a2", "second"), user("u4"),
		);
		const projection = buildSessionProjection(branch);
		const index = findAnchorIndex(branch, projection) as number;
		expect(ids(projection, index)).toEqual(["u3", "a3", "u4"]);
	});

	it("moves the anchor once a later compaction summarizes it", () => {
		const branch = entries(
			sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
			serverCompaction("c1", "u2", "first"), user("u3"), assistant("a3"),
			serverCompaction("c2", "a3", "second"), user("u4"),
		);
		const projection = buildSessionProjection(branch);
		const index = findAnchorIndex(branch, projection) as number;
		expect(ids(projection, index)).toEqual(["u4"]);
	});

	it("keeps the active anchor when the cut lands right after it", () => {
		// c2 keeps u3, the first message after c1, so c1's anchor sits right
		// before the kept part: earlier requests sent it and u3 as one user turn.
		const branch = entries(
			sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
			serverCompaction("c1", "u2", "first"), { type: "thinking_level_change", id: "t", thinkingLevel: "low" },
			user("u3"), assistant("a3"),
			serverCompaction("c2", "u3", "second"), user("u4"),
		);
		const projection = buildSessionProjection(branch);
		const index = findAnchorIndex(branch, projection) as number;
		expect(ids(projection, index).filter((id) => id !== "t")).toEqual(["u3", "a3", "u4"]);
	});

	it("lets a later compaction summarize an anchor that follows tool results", () => {
		// c1 cut inside a turn after the tool results, so its anchor joins them
		// in one user turn. c2 keeps the next message, an assistant reply: the
		// anchor goes to c2's summarized side, and c2 places a new one.
		const branch = entries(
			sys("s"), user("u1"), toolCallAssistant("a1"), toolResult("r1", "a1"),
			serverCompaction("c1", "a1", "first"), assistant("a2"),
			serverCompaction("c2", "a2", "second"), user("u3"),
		);
		const projection = buildSessionProjection(branch);
		const index = findAnchorIndex(branch, projection) as number;
		expect(ids(projection, index)).toEqual(["u3"]);
	});

	it("judges anchor history by what was visible when each compaction ran", () => {
		const omit = (id: string, targetId: string) => ({ type: "context_edit", id, targetId, replacement: null });
		// x was omitted before c2: the requests before c2 sent a2, anchor, u3.
		const before = entries(
			sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
			serverCompaction("c1", "u2", "first"), user("x"), user("u3"), assistant("a3"), omit("e", "x"),
			serverCompaction("c2", "u3", "second"), user("u4"),
		);
		let projection = buildSessionProjection(before);
		let index = findAnchorIndex(before, projection) as number;
		expect(projection.entries.slice(index).flatMap((entry) => entry.messages).map((m) => JSON.stringify((m as { content?: unknown }).content))[0]).toContain("u3");

		// x was omitted after c2: c2 summarized c1's anchor, so the anchor stays after c2's kept turns.
		const after = entries(
			sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
			serverCompaction("c1", "u2", "first"), user("x"), user("u3"), assistant("a3"),
			serverCompaction("c2", "u3", "second"), omit("e", "x"), user("u4"),
		);
		projection = buildSessionProjection(after);
		index = findAnchorIndex(after, projection) as number;
		expect(ids(projection, index)).toEqual(["e", "u4"]);
	});

	it("does not revive an anchor that was never used", () => {
		// c2 kept c1's anchor, so c2 placed none. c3 summarizes c1's anchor and
		// must place its own after its kept turns, not one right after c2.
		const branch = entries(
			sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
			serverCompaction("c1", "u2", "first"), user("u3"), assistant("a3"),
			serverCompaction("c2", "a2", "second"), user("u4"), assistant("a4"),
			serverCompaction("c3", "a3", "third"), user("u5"),
		);
		const projection = buildSessionProjection(branch);
		const index = findAnchorIndex(branch, projection) as number;
		expect(ids(projection, index)).toEqual(["u5"]);
	});

	it("starts over after a Pi summary", () => {
		const branch = entries(
			sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
			serverCompaction("c1", "u2", "first"), user("u3"), assistant("a3"),
			piCompaction("c2", "a2"), user("u4"), assistant("a4"),
			serverCompaction("c3", "u3", "third"), user("u5"),
		);
		const projection = buildSessionProjection(branch);
		const index = findAnchorIndex(branch, projection) as number;
		expect(ids(projection, index)).toEqual(["u5"]);
	});
});

describe("insertAnchor", () => {
	const branch = entries(
		sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
		serverCompaction("c1", "u2"), user("u3"),
	);
	const projection = buildSessionProjection(branch);
	const anchorIndex = findAnchorIndex(branch, projection) as number;
	type Msg = { role: string; timestamp: number; content?: unknown };
	const isAnchor = (m: Msg) => JSON.stringify(m.content ?? "").includes(ANCHOR_TEXT);
	const copy = () => structuredClone(projection.messages) as unknown as Msg[];

	it("inserts the anchor before the first message after the kept turns", () => {
		const out = insertAnchor(copy(), projection, anchorIndex) as Msg[];
		const at = out.findIndex(isAnchor);
		expect(JSON.stringify(out[at - 1].content)).toContain("a2");
		expect(JSON.stringify(out[at + 1].content)).toContain("u3");
	});

	it("falls back to the last kept message when the next one is gone", () => {
		const out = insertAnchor(copy().slice(0, -1), projection, anchorIndex) as Msg[];
		expect(isAnchor(out.at(-1) as Msg)).toBe(true);
	});

	it("gives up when neither neighbour can be found", () => {
		const messages = copy().filter((m) => m.role === "system" || m.role === "compactionSummary");
		expect(insertAnchor(messages, projection, anchorIndex)).toBeUndefined();
	});
});

/** The provider context of an agent request on `branch`, anchored as the context hook does it. */
function agentRequest(branch: SessionEntry[]) {
	const projection = buildSessionProjection(branch);
	const anchorIndex = findAnchorIndex(branch, projection);
	const messages =
		anchorIndex === undefined
			? projection.messages
			: (insertAnchor(projection.messages as never[], projection, anchorIndex) as never[]);
	return convertToLlm(messages as never) as Message[];
}

function build(branch: SessionEntry[], firstKeptEntryId: string, request = agentRequest(branch), provider = "anthropic") {
	return buildCompactionRequestMessages({
		request,
		canonical: canonicalMessages(buildSessionProjection(branch).entries),
		projection: buildSessionProjection(branch),
		firstKeptEntryId,
		model: "claude-haiku-5-5",
		provider,
	});
}

const texts = (messages: readonly unknown[]) =>
	messages.map((m) => JSON.stringify((m as { content?: unknown }).content ?? ""));

describe("buildCompactionRequestMessages", () => {
	it("cuts the last request right after the last summarized message", () => {
		const branch = entries(sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"), user("u3"));
		const request = build(branch, "u2");
		if (!request.ok) throw new Error(request.reason);
		expect(request.messages.map((m) => m.role)).toEqual(["system", "user", "assistant"]);
		expect(request.keptAssistantEffort).toBeUndefined();
	});

	it("summarizes the whole last request when the kept part starts with its response", () => {
		const branch = entries(sys("s"), user("u1"), assistant("a1", "medium"));
		const lastRequest = agentRequest(branch).slice(0, -1);
		const request = build(branch, "a1", lastRequest);
		if (!request.ok) throw new Error(request.reason);
		expect(request.messages).toEqual(lastRequest);
		expect(request.keptAssistantEffort).toBe("medium");
		const other = build(branch, "a1", lastRequest, "other");
		expect(other.ok && other.keptAssistantEffort).toBeUndefined();
	});

	it("finds the kept assistant's effort behind invisible entries", () => {
		const branch = entries(
			sys("s"), user("u1"), { type: "thinking_level_change", id: "t", thinkingLevel: "medium" },
			assistant("a1", "medium"), user("u2"),
		);
		expect(build(branch, "t")).toMatchObject({ ok: true, keptAssistantEffort: "medium" });
	});

	it("keeps an active anchor that lies before the cut, as earlier requests sent it", () => {
		const branch = entries(
			sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
			serverCompaction("c1", "u2"), user("u3"), assistant("a3"), user("u4"),
		);
		const request = build(branch, "u4");
		if (!request.ok) throw new Error(request.reason);
		const t = texts(request.messages);
		const at = t.findIndex((text) => text.includes(ANCHOR_TEXT));
		expect(t[at - 1]).toContain("a2");
		expect(t[at + 1]).toContain("u3");
	});

	it("leaves an anchor right before the first kept message on the kept side", () => {
		const branch = entries(
			sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
			serverCompaction("c1", "u2"), user("u3"), assistant("a3"),
		);
		const request = build(branch, "u3");
		if (!request.ok) throw new Error(request.reason);
		expect(request.messages.at(-1)?.role).toBe("assistant");
		expect(JSON.stringify(request.messages)).not.toContain(ANCHOR_TEXT);
	});

	it("puts an anchor that follows tool results on the summarized side", () => {
		const branch = entries(
			sys("s"), user("u1"), toolCallAssistant("a1"), toolResult("r1", "a1"),
			serverCompaction("c1", "a1"), assistant("a2", "medium"),
		);
		const lastRequest = agentRequest(branch).slice(0, -1);
		expect(lastRequest.at(-1)?.content).toEqual([{ type: "text", text: ANCHOR_TEXT }]);
		const request = build(branch, "a2", lastRequest);
		if (!request.ok) throw new Error(request.reason);
		expect(request.messages).toEqual(lastRequest);
		expect(request.keptAssistantEffort).toBe("medium");
	});

	it("tells apart a summary and the prompt that share its timestamp", () => {
		const branch = entries(
			sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"),
			serverCompaction("c1", "u2"), user("u3"), assistant("a3"), user("u4"),
		);
		const c1 = branch.find((entry) => entry.id === "c1") as SessionEntry;
		const u3 = branch.find((entry) => entry.id === "u3") as { message: { timestamp: number } };
		u3.message.timestamp = Date.parse(c1.timestamp);
		const request = build(branch, "a3");
		if (!request.ok) throw new Error(request.reason);
		expect(texts(request.messages).at(-1)).toContain("u3");
	});

	it("ignores replies pi-ai never sends at the boundary", () => {
		const aborted = assistant("ab");
		(aborted.message as { stopReason: string }).stopReason = "aborted";
		const branch = entries(sys("s"), user("u1"), assistant("a1"), user("u2"), aborted, user("u3"), assistant("a3"));
		expect(build(branch, "u3")).toMatchObject({ ok: false });
		expect(build(branch, "ab")).toMatchObject({ ok: false });
	});

	it("refuses a cut between two messages of the same wire role", () => {
		const custom = {
			type: "custom_message", id: "c", customType: "note", content: "injected", display: false,
		};
		const branch = entries(sys("s"), user("u1"), assistant("a1"), user("u2"), custom, assistant("a2"));
		expect(build(branch, "c")).toEqual({ ok: false, reason: "the cut splits a turn" });
	});

	it("refuses when tools changed during the session", () => {
		const change = {
			type: "message", id: "tc",
			message: { role: "system", content: "", toolsAdded: [{ name: "two", description: "", parameters: {} }], timestamp: clock++ },
		};
		const branch = entries(sys("s"), user("u1"), assistant("a1"), change, user("u2"), assistant("a2"), user("u3"));
		expect(build(branch, "u3")).toEqual({ ok: false, reason: "tools changed during the session" });
	});

	it("refuses when a system prompt update reaches the wire next to the kept turns", () => {
		const update = {
			type: "message", id: "su",
			message: { role: "system", content: "", sections: { b: "More." }, timestamp: clock++ },
		};
		const late = entries(sys("s"), user("u1"), assistant("a1"), user("u2"), update, assistant("a2"));
		expect(build(late, "u2")).toEqual({ ok: false, reason: "the system prompt changed in the kept turns" });
		const early = entries(sys("s"), user("u1"), update, assistant("a1"), user("u2"), assistant("a2"));
		expect(build(early, "u2")).toMatchObject({ ok: true });
	});

	it("refuses when the last request does not hold the summarized messages", () => {
		const branch = entries(sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"), user("u3"));
		const stale = agentRequest(branch).slice(0, 2);
		expect(build(branch, "u3", stale)).toEqual({ ok: false, reason: "the last request does not hold the summarized messages" });
		expect(build(branch, "missing")).toEqual({ ok: false, reason: "first kept entry not found" });
	});

	it("refuses messages the session does not have at the cut", () => {
		const branch = entries(sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"));
		const request = agentRequest(branch);
		request.splice(3, 0, { role: "assistant", content: [], timestamp: 1 } as never);
		expect(build(branch, "u2", request)).toEqual({ ok: false, reason: "unexpected messages at the cut" });
	});
});

describe("wirePrefixMismatch", () => {
	const u = (text: string) => ({ role: "user", content: [{ type: "text", text }] });
	const a = (text: string) => ({ role: "assistant", content: [{ type: "text", text }] });
	const recorded = wireMessageRecords({
		messages: [u("u1"), marker("high"), a("a1"), u("u2"), u("u3"), marker("medium"), a("a3"), u("u4"), marker("high")],
	});

	it("accepts an exact prefix that ends before a new turn", () => {
		expect(wirePrefixMismatch([u("u1"), marker("high"), a("a1")], recorded)).toBeUndefined();
		expect(wirePrefixMismatch([u("u1"), marker("high"), a("a1"), u("u2"), u("u3"), marker("medium")], recorded)).toBeUndefined();
		// The whole request, ending with the effort of its response.
		expect(wirePrefixMismatch(recorded.map((r) => JSON.parse(r.key)), recorded)).toBeUndefined();
	});

	it("ignores cache breakpoints", () => {
		const cached = { role: "user", content: [{ type: "text", text: "u1", cache_control: { type: "ephemeral" } }] };
		expect(wirePrefixMismatch([cached, marker("high"), a("a1")], recorded)).toBeUndefined();
	});

	it("rejects a cut inside a wire turn", () => {
		expect(wirePrefixMismatch([u("u1"), marker("high"), a("a1"), u("u2")], recorded)).toBe("the cut splits a turn on the wire");
	});

	it("rejects an effort marker the last request did not send there", () => {
		expect(wirePrefixMismatch([u("u1"), marker("high"), a("a1"), u("u2"), marker("high")], recorded)).toBe(
			"the compaction request differs from the last request",
		);
		expect(wirePrefixMismatch([u("u1"), marker("high")], wireMessageRecords({ messages: [u("u1"), marker("high"), u("u2")] }))).toBe(
			"the effort marker does not precede a kept reply",
		);
	});

	it("rejects messages the last request did not send", () => {
		expect(wirePrefixMismatch([u("rewritten"), marker("high"), a("a1")], recorded)).toBe("the compaction request differs from the last request");
		expect(wirePrefixMismatch([...recorded.map((r) => JSON.parse(r.key)), a("extra")], recorded)).toBe(
			"the compaction request outgrows the last request",
		);
	});
});

describe("buildCompactionInstructions", () => {
	it("asks for Pi's format, names files in prose, and forbids tools", () => {
		const text = buildCompactionInstructions({ readFiles: ["a.ts"], modifiedFiles: ["b.ts"], customInstructions: "Keep dates." });
		expect(text).toContain("## Goal");
		expect(text).toContain("## Critical Context");
		expect(text).toContain("files read (a.ts)");
		expect(text).toContain("files modified (b.ts)");
		expect(text).toContain("Do not call tools.");
		expect(text.endsWith("Additional focus: Keep dates.")).toBe(true);
		// Live tests: <read-files> blocks in the instructions were refused.
		expect(text).not.toContain("<read-files>");
	});

	it("stays within the API's 16,384 characters", () => {
		const many = Array.from({ length: 2_000 }, (_, i) => "src/file-" + i + ".ts");
		const text = buildCompactionInstructions({ readFiles: many, modifiedFiles: many, customInstructions: "x".repeat(20_000) });
		expect(text.length).toBeLessThanOrEqual(16_384);
		expect(text).toContain("Do not call tools.");
	});
});

// ---------------------------------------------------------------------------
// Compaction request
// ---------------------------------------------------------------------------

const MODEL = {
	id: "claude-haiku-5-5",
	provider: "anthropic",
	api: "anthropic-messages",
	reasoning: true,
	cost: { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
} as unknown as Model<Api>;

interface FakeRun {
	events: unknown[];
	payloads: Array<Record<string, unknown>>;
	options: Array<Record<string, unknown>>;
}

function fakeStreamSimple(run: FakeRun, basePayload: () => Record<string, unknown>) {
	return ((_model: unknown, _context: unknown, options: Record<string, unknown>) => {
		run.options.push(options);
		const result = (async () => {
			const p = basePayload();
			const next = await (options.onPayload as (p: unknown) => unknown)(p);
			run.payloads.push((next ?? p) as Record<string, unknown>);
			for (const event of run.events)
				await (options.onProviderStreamEvent as (e: unknown) => unknown)(event);
			return { role: "assistant", content: [], stopReason: "error", errorMessage: "Unhandled stop reason: compaction" };
		})();
		return { result: () => result } as never;
	}) as never;
}

const compactionEvents = (block: unknown = BLOCK, stop = "compaction") => [
	{ type: "message_start", message: { id: "m" } },
	{ type: "content_block_start", index: 0, content_block: block },
	{ type: "content_block_stop", index: 0 },
	{
		type: "message_delta",
		delta: { stop_reason: stop },
		usage: {
			input_tokens: 0,
			output_tokens: 0,
			iterations: [{ type: "compaction", input_tokens: 400, output_tokens: 300, cache_read_input_tokens: 1000, cache_creation_input_tokens: 0 }],
		},
	},
];
const withoutBlock = (events: unknown[]) =>
	events.filter((e) => (e as { type: string }).type !== "content_block_start");

describe("requestServerCompaction", () => {
	const base = () => ({ model: "claude-haiku-5-5", messages: [{ role: "user", content: "u" }, marker("high")] });
	const run = (events: unknown[]): FakeRun => ({ events, payloads: [], options: [] });

	it("returns the block exactly as streamed and the compaction iteration's usage", async () => {
		const r = run(compactionEvents());
		const outcome = await requestServerCompaction({ streamSimple: fakeStreamSimple(r, base), model: MODEL, messages: [], instructions: "I", reasoning: "high" });
		expect(outcome).toMatchObject({ ok: true, block: BLOCK });
		if (!outcome.ok) throw new Error("expected success");
		expect(outcome.usage).toMatchObject({ input: 400, output: 300, cacheRead: 1000, cacheWrite: 0, totalTokens: 1700 });
		expect(outcome.usage.cost.total).toBeGreaterThan(0);
		expect(r.options[0]).toMatchObject({ disableContinuation: true, reasoning: "high" });
	});

	it("drops the trailing effort marker and asks for a summary with the beta", async () => {
		const r = run(compactionEvents());
		await requestServerCompaction({ streamSimple: fakeStreamSimple(r, base), model: MODEL, messages: [], instructions: "I" });
		expect(r.payloads[0].messages).toEqual([{ role: "user", content: "u" }]);
		expect(r.payloads[0].compaction).toEqual({ type: "summarize", instructions: "I" });
		expect(r.payloads[0].betas).toEqual([COMPACT_BETA]);
	});

	it("ends with the kept assistant's effort when Pi cut inside a turn", async () => {
		const r = run(compactionEvents());
		await requestServerCompaction({ streamSimple: fakeStreamSimple(r, base), model: MODEL, messages: [], instructions: "I", keptAssistantEffort: "medium" });
		expect(r.payloads[0].messages).toEqual([{ role: "user", content: "u" }, marker("medium")]);
	});

	for (const stop of ["max_tokens", "refusal", "end_turn", "tool_use"]) {
		it("fails on stop reason " + stop, async () => {
			const r = run(withoutBlock(compactionEvents(undefined, stop)));
			const outcome = await requestServerCompaction({ streamSimple: fakeStreamSimple(r, base), model: MODEL, messages: [], instructions: "I" });
			expect(outcome).toMatchObject({ ok: false, stopReason: stop });
		});
	}

	it("fails when the block is missing or the request errored", async () => {
		const missing = run(compactionEvents({ type: "text", text: "hi" }));
		expect(await requestServerCompaction({ streamSimple: fakeStreamSimple(missing, base), model: MODEL, messages: [], instructions: "I" })).toMatchObject({ ok: false });
		const errored = run([]);
		expect(await requestServerCompaction({ streamSimple: fakeStreamSimple(errored, base), model: MODEL, messages: [], instructions: "I" })).toMatchObject({
			ok: false,
			reason: "Unhandled stop reason: compaction",
		});
	});

	it("reports an abort", async () => {
		const controller = new AbortController();
		controller.abort();
		const r = run(compactionEvents());
		expect(await requestServerCompaction({ streamSimple: fakeStreamSimple(r, base), model: MODEL, messages: [], instructions: "I", signal: controller.signal })).toEqual({
			ok: false,
			reason: "aborted",
		});
	});
});

describe("session_before_compact handler", () => {
	type Handler = (event: unknown, ctx: unknown) => unknown;
	function handlers() {
		const registered: Record<string, Handler[]> = {};
		registerServerCompaction({
			on: (name: string, handler: Handler) => {
				registered[name] ??= [];
				registered[name].push(handler);
			},
		} as unknown as ExtensionAPI);
		return registered;
	}

	const branch = entries(sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"));
	function event() {
		return {
			type: "session_before_compact",
			preparation: {
				firstKeptEntryId: "u2",
				tokensBefore: 1234,
				isSplitTurn: false,
				fileOps: { read: new Set(["r.ts"]), written: new Set<string>(), edited: new Set(["e.ts"]) },
			},
			branchEntries: branch,
			customInstructions: undefined,
			reason: "manual",
			signal: new AbortController().signal,
		};
	}
	function ctx(r: FakeRun, apiKey = "sk-ant-oat01-x", model: unknown = MODEL) {
		const notices: string[] = [];
		recordAgentRequest("sess", {
			modelId: "claude-haiku-5-5",
			messages: agentRequest(branch),
			canonical: canonicalMessages(buildSessionProjection(branch).entries),
			declarationsKey: payloadDeclarationsKey({}),
		});
		return {
			notices,
			model,
			hasUI: true,
			thinkingLevel: "high",
			ui: { notify: (message: string) => notices.push(message) },
			sessionManager: { getSessionId: () => "sess" },
			modelRegistry: {
				getApiKeyAndHeaders: async () => ({ ok: true, apiKey }),
				streamSimple: fakeStreamSimple(r, () => ({ model: "claude-haiku-5-5", messages: [] })),
			},
		};
	}

	it("returns Anthropic's summary with the whole block in details", async () => {
		const r: FakeRun = { events: compactionEvents(), payloads: [], options: [] };
		const result = (await handlers().session_before_compact[0](event(), ctx(r))) as { compaction: Record<string, unknown> };
		expect(result.compaction).toMatchObject({
			summary: BLOCK.content,
			firstKeptEntryId: "u2",
			tokensBefore: 1234,
			details: { anthropicCompaction: BLOCK, readFiles: ["r.ts"], modifiedFiles: ["e.ts"] },
		});
		expect(r.options[0]).toMatchObject({ apiKey: "sk-ant-oat01-x", sessionId: "sess" });
	});

	it("retries a refused request once without the file lists, then falls back to Pi", async () => {
		const r: FakeRun = { events: withoutBlock(compactionEvents(undefined, "refusal")), payloads: [], options: [] };
		const c = ctx(r);
		expect(await handlers().session_before_compact[0](event(), c)).toBeUndefined();
		expect(r.payloads).toHaveLength(2);
		expect(JSON.stringify(r.payloads[0].compaction)).toContain("r.ts");
		expect(JSON.stringify(r.payloads[1].compaction)).not.toContain("r.ts");
		expect(c.notices[0]).toContain("Using Pi's summary instead");
	});

	it("falls back without a request when there is no earlier agent request to continue", async () => {
		const r: FakeRun = { events: compactionEvents(), payloads: [], options: [] };
		const c = ctx(r);
		forgetAgentRequests();
		expect(await handlers().session_before_compact[0](event(), c)).toBeUndefined();
		expect(r.payloads).toHaveLength(0);
		expect(c.notices[0]).toContain("no earlier request to claude-haiku-5-5");
	});

	it("leaves compaction to Pi for API keys, other models, and the kill switch", async () => {
		const r: FakeRun = { events: compactionEvents(), payloads: [], options: [] };
		const h = handlers().session_before_compact[0];
		expect(await h(event(), ctx(r, "sk-ant-api03-x"))).toBeUndefined();
		expect(await h(event(), ctx(r, "sk-ant-oat01-x", { ...MODEL, id: "claude-sonnet-4-5" }))).toBeUndefined();
		expect(await h(event(), ctx(r, "sk-ant-oat01-x", { ...MODEL, provider: "openrouter" }))).toBeUndefined();
		process.env.PI_CLAUDE_AUTH_SERVER_COMPACTION = "0";
		expect(await h(event(), ctx(r))).toBeUndefined();
		expect(r.payloads).toHaveLength(0);
	});
});
