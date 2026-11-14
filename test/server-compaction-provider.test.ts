import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Api, Message, Model, Provider } from "@earendil-works/pi-ai";
import {
	buildSessionProjection,
	convertToLlm,
	type ExtensionAPI,
	ModelRuntime,
	type ProviderConfig,
	type SessionEntry,
} from "@earendil-works/pi-coding-agent";
import type { DeviceIdentity } from "../src/device-id.ts";
import { createClaudeAuthExtension } from "../src/index.ts";
import {
	ANCHOR_TEXT,
	forgetAgentRequests,
	forgetCompactionBlocks,
	rememberCompactionBlock,
	summaryMessageText,
} from "../src/server-compaction.ts";

/**
 * Server compaction through the real provider path: the registered overlay over
 * pi-ai's Anthropic provider, with a stubbed fetch that records each request
 * body and answers compaction requests with a signed block. Agent requests run
 * Pi's before_provider_request handlers as Pi does; their 500 answer does not
 * matter, only the recorded body.
 */

const OAUTH_TOKEN = "sk-ant-oat01-test-token";
const SESSION = "provider-compaction-session";
const SYSTEM_TEXT = "Pi system prompt for provider tests.";
const FIXTURE_IDENTITY: DeviceIdentity = {
	deviceId: "d".repeat(64),
	accountUuid: "00000000-0000-4000-8000-000000000001",
};

type Handler = (event: unknown, ctx: unknown) => unknown;
// biome-ignore lint/suspicious/noExplicitAny: wire bodies are untyped JSON
type Body = any;

const originalFetch = globalThis.fetch;

function sse(events: unknown[]): Response {
	const text = events
		.map((event) => "event: " + (event as { type: string }).type + "\ndata: " + JSON.stringify(event) + "\n\n")
		.join("");
	return new Response(text, { status: 200, headers: { "content-type": "text/event-stream" } });
}

function compactionResponse(content: string): Response {
	return sse([
		{
			type: "message_start",
			message: { id: "msg_c", type: "message", role: "assistant", model: "claude-haiku-5-5", content: [], stop_reason: null, usage: { input_tokens: 0, output_tokens: 0 } },
		},
		{ type: "content_block_start", index: 0, content_block: { type: "compaction", content, signature: "sig-" + content } },
		{ type: "content_block_stop", index: 0 },
		{
			type: "message_delta",
			delta: { stop_reason: "compaction" },
			usage: { input_tokens: 0, output_tokens: 0, iterations: [{ type: "compaction", input_tokens: 10, output_tokens: 5 }] },
		},
		{ type: "message_stop" },
	]);
}

const tool = (name: string) => ({
	name,
	description: name + " tool",
	parameters: { type: "object", properties: {}, required: [] },
});

let clock = 1_000_000;
function chain(...specs: Array<Record<string, unknown>>): SessionEntry[] {
	let parentId: string | null = null;
	return specs.map((spec) => {
		const entry = { parentId, timestamp: new Date(clock++).toISOString(), ...spec };
		parentId = entry.id as string;
		return entry as unknown as SessionEntry;
	});
}
const sys = (id: string, tools = [tool("read"), tool("codemode")]) => ({
	type: "message",
	id,
	message: { role: "system", content: SYSTEM_TEXT, toolsAdded: tools, timestamp: clock++ },
});
const user = (id: string) => ({
	type: "message",
	id,
	message: { role: "user", content: [{ type: "text", text: "prompt " + id }], timestamp: clock++ },
});
const assistant = (id: string, effort = "high", content: unknown[] = [{ type: "text", text: "reply " + id }]) => ({
	type: "message",
	id,
	message: {
		role: "assistant",
		content,
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-haiku-5-5",
		providerThinkingLevel: effort,
		stopReason: "stop",
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
		timestamp: clock++,
	},
});

// Pi stores the current system message in every compaction entry.
const snapshot = () => ({ role: "system", content: SYSTEM_TEXT, toolsAdded: [tool("read"), tool("codemode")], timestamp: clock++ });

describe("server compaction through the provider", () => {
	let dir: string;
	let runtime: ModelRuntime;
	let model: Model<Api>;
	let bodies: Body[];
	let handlers: Record<string, Handler[]>;
	let compactions: number;

	beforeEach(async () => {
		dir = mkdtempSync(join(tmpdir(), "pi-claude-auth-compaction-"));
		process.env.PI_CODING_AGENT_DIR = dir;
		process.env.ANTHROPIC_CLI_VERSION = "1.2.3";
		mkdirSync(dir, { recursive: true });
		writeFileSync(join(dir, "models.json"), "{}");
		bodies = [];
		compactions = 0;
		globalThis.fetch = (async (_input: unknown, init?: RequestInit) => {
			const body = JSON.parse(String(init?.body));
			bodies.push(body);
			if (body.compaction) return compactionResponse("Summary " + ++compactions);
			return new Response(JSON.stringify({ type: "error", error: { type: "api_error", message: "stub" } }), {
				status: 500,
				headers: { "content-type": "application/json" },
			});
		}) as typeof fetch;
		runtime = await ModelRuntime.create({
			authPath: join(dir, "auth.json"),
			modelsPath: join(dir, "models.json"),
			modelsStorePath: join(dir, "models-store.json"),
			allowModelNetwork: false,
			refreshOnCreate: false,
		});
		handlers = {};
		await createClaudeAuthExtension(async () => FIXTURE_IDENTITY)({
			registerProvider(nameOrProvider: string | Provider, config?: ProviderConfig) {
				if (typeof nameOrProvider === "string") runtime.registerProvider(nameOrProvider, config as never);
				else runtime.registerNativeProvider(nameOrProvider as never);
			},
			on(name: string, handler: Handler) {
				handlers[name] ??= [];
				handlers[name].push(handler);
			},
			registerCommand() {},
			getCommands: () => [],
		} as unknown as ExtensionAPI);
		const found = runtime.getModels("anthropic").find((m) => m.id === "claude-haiku-5-5");
		if (!found) throw new Error("claude-haiku-5-5 missing from the catalog");
		model = found;
	});

	afterEach(() => {
		globalThis.fetch = originalFetch;
		rmSync(dir, { recursive: true, force: true });
		delete process.env.PI_CODING_AGENT_DIR;
		delete process.env.ANTHROPIC_CLI_VERSION;
		forgetCompactionBlocks();
		forgetAgentRequests();
	});

	function ctx(branch: SessionEntry[]) {
		return {
			model,
			hasUI: false,
			thinkingLevel: "high",
			ui: { notify() {} },
			sessionManager: {
				getSessionId: () => SESSION,
				getBranch: () => branch,
				getEntries: () => branch,
				buildSessionProjection: () => buildSessionProjection(branch),
			},
			modelRegistry: {
				getApiKeyAndHeaders: async () => ({ ok: true, apiKey: OAUTH_TOKEN }),
				streamSimple: runtime.streamSimple.bind(runtime),
			},
		};
	}

	/** Send an agent request the way Pi does, including its before_provider_request handlers. */
	async function agentRequest(messages: Message[]): Promise<Body> {
		const before = bodies.length;
		const stream = runtime.streamSimple(model, { messages }, {
			apiKey: OAUTH_TOKEN,
			sessionId: SESSION,
			maxRetries: 0,
			reasoning: "high",
			onPayload: async (payload: unknown) => {
				let current = payload;
				for (const handler of handlers.before_provider_request ?? []) {
					const next = await handler({ type: "before_provider_request", payload: current }, {});
					if (next !== undefined) current = next;
				}
				return current === payload ? undefined : current;
			},
		} as never);
		await stream.result();
		expect(bodies.length).toBe(before + 1);
		return bodies[before];
	}

	/** The agent context Pi would send on \`branch\`, after this extension's context hook. */
	async function sessionRequest(branch: SessionEntry[]): Promise<Message[]> {
		const projection = buildSessionProjection(branch);
		let messages = projection.messages as unknown[];
		for (const handler of handlers.context_with_system ?? []) {
			const result = (await handler({ type: "context_with_system", messages }, ctx(branch))) as { messages?: unknown[] } | undefined;
			messages = result?.messages ?? messages;
		}
		return convertToLlm(messages as never) as Message[];
	}

	async function compact(branch: SessionEntry[], firstKeptEntryId: string) {
		const handler = handlers.session_before_compact[0];
		return (await handler(
			{
				type: "session_before_compact",
				preparation: {
					firstKeptEntryId,
					tokensBefore: 100,
					isSplitTurn: false,
					fileOps: { read: new Set<string>(), written: new Set<string>(), edited: new Set<string>() },
				},
				branchEntries: branch,
				customInstructions: undefined,
				reason: "manual",
				signal: new AbortController().signal,
			},
			ctx(branch),
		)) as { compaction: { summary: string; firstKeptEntryId: string; details: { anthropicCompaction: unknown } } } | undefined;
	}

	const compactionBodies = () => bodies.filter((body) => body.compaction);
	const toolNames = (body: Body) => (body.tools ?? []).map((t: { name: string }) => t.name);
	const stripCache = (value: unknown) =>
		JSON.parse(JSON.stringify(value, (key, v) => (key === "cache_control" ? undefined : v)));

	function appendCompaction(branch: SessionEntry[], result: NonNullable<Awaited<ReturnType<typeof compact>>>, id: string) {
		const [entry] = chain({
			type: "compaction",
			id,
			summary: result.compaction.summary,
			firstKeptEntryId: result.compaction.firstKeptEntryId,
			tokensBefore: 100,
			fromHook: true,
			details: result.compaction.details,
			systemMessage: snapshot(),
		});
		(entry as { parentId: string }).parentId = branch.at(-1)?.id as string;
		return [...branch, entry];
	}

	it("declares the tools the last agent request declared, not the raw session's [finding 1]", async () => {
		const branch = chain(sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"), user("u3"));
		// Pi's hidden-declaration projection removed \`read\` from every request.
		const sent = (await sessionRequest(branch)).map((m) =>
			m.role === "system" ? { ...m, toolsAdded: [tool("codemode")] } : m,
		) as Message[];
		const agentBody = await agentRequest(sent);
		expect(toolNames(agentBody)).not.toContain("Read");

		const result = await compact(branch, "u2");
		expect(result?.compaction.summary).toBe("Summary 1");
		const [compactionBody] = compactionBodies();
		expect(stripCache(compactionBody.tools)).toEqual(stripCache(agentBody.tools));
		expect(compactionBody.messages.map((m: { role: string }) => m.role)).toEqual(["user", "system", "assistant"]);
	});

	it("sends no compaction request when agent payloads declare tools it cannot reproduce [finding 1]", async () => {
		const branch = chain(sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"), user("u3"));
		handlers.before_provider_request.push((event) => {
			const payload = (event as { payload: Body }).payload;
			return { ...payload, tools: [...(payload.tools ?? []), { name: "extra", input_schema: { type: "object" } }] };
		});
		await agentRequest(await sessionRequest(branch));
		expect(await compact(branch, "u2")).toBeUndefined();
		expect(compactionBodies()).toHaveLength(0);
	});

	it("falls back when tools changed during the session [finding 2]", async () => {
		const change = { type: "message", id: "tc", message: { role: "system", content: "", toolsAdded: [tool("two")], timestamp: clock++ } };
		const branch = chain(sys("s", [tool("one")]), user("u1"), assistant("a1"), change, user("u2"), assistant("a2"), user("u3"));
		const agentBody = await agentRequest(await sessionRequest(branch));
		expect(JSON.stringify(agentBody.messages)).toContain("tool_addition");
		expect(await compact(branch, "u3")).toBeUndefined();
		expect(compactionBodies()).toHaveLength(0);
	});

	it("falls back when the cut separates two user messages [finding 3]", async () => {
		const custom = { type: "custom_message", id: "c", customType: "note", content: "injected note", display: false };
		const branch = chain(sys("s"), user("u1"), assistant("a1"), user("u2"), custom, assistant("a2"), user("u3"));
		const agentBody = await agentRequest(await sessionRequest(branch));
		const roles = agentBody.messages.filter((m: { role: string }) => m.role !== "system").map((m: { role: string }) => m.role);
		expect(roles.slice(2, 4)).toEqual(["user", "user"]);
		expect(await compact(branch, "c")).toBeUndefined();
		expect(compactionBodies()).toHaveLength(0);
	});

	it("ends with the kept assistant's effort when invisible entries precede it [finding 4]", async () => {
		const branch = chain(
			sys("s"), user("u1"), assistant("a1"), user("u2"),
			{ type: "thinking_level_change", id: "t", thinkingLevel: "medium" },
			assistant("a2", "medium"),
		);
		// The last agent request is the one that produced a2.
		await agentRequest((await sessionRequest(branch)).slice(0, -1));
		const result = await compact(branch, "t");
		expect(result?.compaction.firstKeptEntryId).toBe("t");
		const [body] = compactionBodies();
		expect(body.messages.at(-1)).toMatchObject({ role: "system", output_config: { effort: "medium" } });
		expect(body.messages.at(-2).role).toBe("user");
	});

	it("puts the system text after tool results when a block has no anchor [finding 5]", async () => {
		rememberCompactionBlock("Earlier summary", { type: "compaction", content: "Earlier summary", signature: "sig" });
		const messages = [
			{ role: "system", content: SYSTEM_TEXT, toolsAdded: [tool("read")], timestamp: 1 },
			{ role: "user", content: [{ type: "text", text: summaryMessageText("Earlier summary") }], timestamp: 2 },
			{ ...(assistant("a", "high", [{ type: "toolCall", id: "call_1", name: "read", arguments: {} }]).message), stopReason: "toolUse" },
			{ role: "toolResult", toolCallId: "call_1", toolName: "read", content: [{ type: "text", text: "file" }], isError: false, timestamp: clock++ },
		] as Message[];
		const body = await agentRequest(messages);
		expect(body.messages[0]).toMatchObject({ role: "assistant", content: [{ type: "compaction", content: "Earlier summary" }] });
		const results = body.messages.find((m: Body) => Array.isArray(m.content) && m.content.some((b: Body) => b.type === "tool_result"));
		expect(results.content.map((b: Body) => b.type)).toEqual(["tool_result", "text"]);
		expect(results.content[1].text).toBe(SYSTEM_TEXT);
	});

	it("summarizes an anchor that joined tool results when a repeated compaction keeps the next reply", async () => {
		rememberCompactionBlock("First", { type: "compaction", content: "First", signature: "sig-first" });
		const call = assistant("a1", "high", [{ type: "toolCall", id: "call_1", name: "read", arguments: {} }]);
		(call.message as { stopReason: string }).stopReason = "toolUse";
		const result1 = { type: "message", id: "r1", message: { role: "toolResult", toolCallId: "call_1", toolName: "read", content: [{ type: "text", text: "file" }], isError: false, timestamp: clock++ } };
		let branch = chain(
			sys("s"), user("u1"), call, result1,
			{ type: "compaction", id: "c1", summary: "First", firstKeptEntryId: "a1", tokensBefore: 1, fromHook: true, systemMessage: snapshot(), details: { anthropicCompaction: { type: "compaction", content: "First", signature: "sig-first" } } },
			assistant("a2", "medium"),
		);
		// The last agent request is the one that produced a2.
		await agentRequest((await sessionRequest(branch)).slice(0, -1));
		const result = await compact(branch, "a2");
		if (!result) throw new Error("expected a server compaction");
		const [body] = compactionBodies();
		expect(body.messages.at(-1)).toMatchObject({ role: "system", output_config: { effort: "medium" } });
		expect(JSON.stringify(body.messages.at(-2))).toContain(SYSTEM_TEXT);

		branch = appendCompaction(branch, result, "c2");
		const [u3] = chain(user("u3"));
		(u3 as { parentId: string }).parentId = "c2";
		branch = [...branch, u3];
		const later = await agentRequest(await sessionRequest(branch));
		const texts = later.messages.map((m: Body) => JSON.stringify(m.content));
		expect(later.messages[0]).toMatchObject({ role: "assistant", content: [{ type: "compaction", content: "Summary 1" }] });
		expect(texts.findIndex((t: string) => t.includes(SYSTEM_TEXT))).toBe(texts.findIndex((t: string) => t.includes("prompt u3")) - 1);
		expect(texts.findIndex((t: string) => t.includes("reply a2"))).toBeLessThan(texts.findIndex((t: string) => t.includes(SYSTEM_TEXT)));
	});

	it("keeps the active anchor on the kept side when a repeated compaction cuts right after it [finding 6]", async () => {
		rememberCompactionBlock("First", { type: "compaction", content: "First", signature: "sig-first" });
		let branch = chain(sys("s"), user("u1"), assistant("a1"), user("u2"), assistant("a2"));
		const [c1] = chain({ type: "compaction", id: "c1", summary: "First", firstKeptEntryId: "u2", tokensBefore: 1, fromHook: true, systemMessage: snapshot(), details: { anthropicCompaction: { type: "compaction", content: "First", signature: "sig-first" } } });
		(c1 as { parentId: string }).parentId = "a2";
		branch = [...branch, c1];
		const after = chain(user("u3"), assistant("a3"), user("u4"));
		(after[0] as { parentId: string }).parentId = "c1";
		branch = [...branch, ...after];

		const before = await agentRequest(await sessionRequest(branch));
		const beforeTexts = before.messages.map((m: Body) => JSON.stringify(m.content));
		const anchorAt = beforeTexts.findIndex((t: string) => t.includes(SYSTEM_TEXT));
		expect(beforeTexts[anchorAt - 1]).toContain("reply a2");
		expect(beforeTexts[anchorAt + 1]).toContain("prompt u3");

		const result = await compact(branch, "u3");
		if (!result) throw new Error("expected a server compaction");
		const [body] = compactionBodies();
		expect(body.messages.at(-1).role).toBe("assistant");
		expect(JSON.stringify(body.messages.at(-1))).toContain("reply a2");
		expect(JSON.stringify(body)).not.toContain(SYSTEM_TEXT);
		expect(JSON.stringify(body)).not.toContain(ANCHOR_TEXT);

		branch = appendCompaction(branch, result, "c2");
		const later = await agentRequest(await sessionRequest(branch));
		expect(later.messages[0]).toMatchObject({ role: "assistant", content: [{ type: "compaction", content: "Summary 1" }] });
		const laterTexts = later.messages.map((m: Body) => JSON.stringify(m.content));
		expect(laterTexts[1]).toContain(SYSTEM_TEXT);
		expect(laterTexts[2]).toContain("prompt u3");
	});
});
