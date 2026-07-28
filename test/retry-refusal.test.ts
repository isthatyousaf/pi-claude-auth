import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import {
	type ExtensionAPI,
	SessionManager,
} from "@earendil-works/pi-coding-agent";
import {
	computeBranchTarget,
	createRetryAfterRefusalState,
	getRefusalMode,
	registerRetryAfterRefusal,
	shouldHandleRefusal,
} from "../src/retry-refusal.ts";

type Handler = (...args: unknown[]) => unknown;

let savedMode: string | undefined;
beforeEach(() => {
	savedMode = process.env.PI_CLAUDE_AUTH_REFUSAL_MODE;
	delete process.env.PI_CLAUDE_AUTH_REFUSAL_MODE;
});
afterEach(() => {
	if (savedMode === undefined) delete process.env.PI_CLAUDE_AUTH_REFUSAL_MODE;
	else process.env.PI_CLAUDE_AUTH_REFUSAL_MODE = savedMode;
});

const fableRefusal = {
	role: "assistant",
	provider: "anthropic",
	model: "claude-fable-5",
	stopReason: "error",
	errorMessage: "The request was blocked by a safety classifier",
};

interface HarnessOptions {
	action?: "continue" | "edit";
	mode?: "tui" | "rpc" | "json" | "print";
	refusedModel?: { provider: string; id: string; name: string };
	editorDraft?: string;
}

function createHarness(options: HarnessOptions = {}) {
	const handlers: Record<string, Handler[]> = {};
	const notifications: unknown[] = [];
	const sentMessages: unknown[] = [];
	const editorValues: string[] = [];
	const selectedModels: unknown[] = [];
	let customCalls = 0;
	let editorText = options.editorDraft ?? "";

	const refusedModel = options.refusedModel ?? {
		provider: "anthropic",
		id: "claude-fable-5",
		name: "Claude Fable 5",
	};
	const fallbackModel = {
		provider: "anthropic",
		id: "claude-opus-4-8",
		name: "Claude Opus 4.8",
	};

	const session = SessionManager.inMemory();
	session.appendMessage({
		role: "user",
		content: "Do the task",
		timestamp: Date.now(),
	});
	session.appendMessage({
		role: "assistant",
		content: [{ type: "toolCall", id: "t1", name: "read", arguments: {} }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-fable-5",
		usage: {
			input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "toolUse",
		timestamp: Date.now(),
	});
	session.appendMessage({
		role: "toolResult",
		toolCallId: "t1",
		toolName: "read",
		content: [{ type: "text", text: "binary contents" }],
		isError: false,
		timestamp: Date.now(),
	});

	const pi = {
		on(event: string, handler: Handler) {
			handlers[event] = [...(handlers[event] ?? []), handler];
		},
		sendMessage(message: unknown, sendOptions: unknown) {
			sentMessages.push({ message, options: sendOptions });
		},
		sendUserMessage() {},
		async setModel(model: unknown) {
			selectedModels.push(model);
			return true;
		},
	} as unknown as ExtensionAPI;

	const ctx = {
		mode: options.mode ?? "tui",
		hasUI: true,
		model: refusedModel,
		modelRegistry: {
			find(provider: string, id: string) {
				if (provider !== "anthropic") return undefined;
				if (id === fallbackModel.id) return fallbackModel;
				if (id === refusedModel.id) return refusedModel;
				return undefined;
			},
		},
		sessionManager: session,
		ui: {
			notify(message: string, kind: string) {
				notifications.push({ message, kind });
			},
			setEditorText(value: string) {
				editorText = value;
				editorValues.push(value);
			},
			getEditorText() {
				return editorText;
			},
			async custom() {
				customCalls++;
				return options.action;
			},
		},
	};

	registerRetryAfterRefusal(pi);
	return {
		ctx,
		editorValues,
		fallbackModel,
		get customCalls() {
			return customCalls;
		},
		handlers,
		notifications,
		selectedModels,
		sentMessages,
	};
}

describe("refusal detection", () => {
	it("handles Anthropic Fable 5 and Opus 5 classifier refusals", () => {
		expect(shouldHandleRefusal(fableRefusal)).toBe(true);
		expect(
			shouldHandleRefusal({
				...fableRefusal,
				model: "claude-opus-5",
				errorMessage: "Safeguards flagged this response",
			}),
		).toBe(true);
	});

	it("does not handle other model families or unrelated errors", () => {
		expect(
			shouldHandleRefusal({ ...fableRefusal, model: "claude-sonnet-4-5" }),
		).toBe(false);
		expect(
			shouldHandleRefusal({ ...fableRefusal, errorMessage: "network timeout" }),
		).toBe(false);
	});

	it("prevents duplicate handling until the current refusal is complete", () => {
		const state = createRetryAfterRefusalState();
		expect(state.begin(fableRefusal)).toBe(true);
		expect(state.begin(fableRefusal)).toBe(false);
		state.complete();
		expect(state.begin(fableRefusal)).toBe(true);
	});
});

describe("refusal mode", () => {
	it("asks by default and supports automatic continuation", () => {
		expect(getRefusalMode()).toBe("ask");
		process.env.PI_CLAUDE_AUTH_REFUSAL_MODE = "auto";
		expect(getRefusalMode()).toBe("auto");
	});

	it("falls back to ask for unknown values", () => {
		process.env.PI_CLAUDE_AUTH_REFUSAL_MODE = "unknown";
		expect(getRefusalMode()).toBe("ask");
	});
});

describe("branch target computation", () => {
	function makeToolBatchSession() {
		const session = SessionManager.inMemory();
		session.appendMessage({
			role: "user",
			content: "Do the task",
			timestamp: Date.now(),
		});
		session.appendMessage({
			role: "assistant",
			content: [{ type: "toolCall", id: "t1", name: "read", arguments: {} }],
			api: "anthropic-messages",
			provider: "anthropic",
			model: "claude-fable-5",
			usage: {
				input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "toolUse",
			timestamp: Date.now(),
		});
		session.appendMessage({
			role: "toolResult",
			toolCallId: "t1",
			toolName: "read",
			content: [{ type: "text", text: "file contents" }],
			isError: false,
			timestamp: Date.now(),
		});
		return session;
	}

	it("targets the tool-call batch parent when the trigger is a tool result", () => {
		const session = makeToolBatchSession();
		const leafId = session.getLeafId()!;
		const target = computeBranchTarget(session, leafId);
		const targetEntry = session.getEntry(target);
		expect(targetEntry?.type).toBe("message");
		expect((targetEntry as { message: { role: string } }).message.role).toBe("user");
	});

	it("targets the user message itself when the trigger is a user message", () => {
		const session = SessionManager.inMemory();
		const userId = session.appendMessage({
			role: "user",
			content: "Risky request",
			timestamp: Date.now(),
		});
		const target = computeBranchTarget(session, userId);
		expect(target).toBe(userId);
	});

	it("walks back through multiple tool results to find the batch start", () => {
		const session = SessionManager.inMemory();
		session.appendMessage({
			role: "user",
			content: "Do multi-step task",
			timestamp: Date.now(),
		});
		session.appendMessage({
			role: "assistant",
			content: [
				{ type: "toolCall", id: "a", name: "read", arguments: {} },
				{ type: "toolCall", id: "b", name: "bash", arguments: {} },
			],
			api: "anthropic-messages",
			provider: "anthropic",
			model: "claude-fable-5",
			usage: {
				input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "toolUse",
			timestamp: Date.now(),
		});
		session.appendMessage({
			role: "toolResult",
			toolCallId: "a",
			toolName: "read",
			content: [{ type: "text", text: "output a" }],
			isError: false,
			timestamp: Date.now(),
		});
		session.appendMessage({
			role: "toolResult",
			toolCallId: "b",
			toolName: "bash",
			content: [{ type: "text", text: "output b" }],
			isError: false,
			timestamp: Date.now(),
		});
		const leafId = session.getLeafId()!;
		const target = computeBranchTarget(session, leafId);
		const targetEntry = session.getEntry(target);
		expect(targetEntry?.type).toBe("message");
		expect((targetEntry as { message: { role: string } }).message.role).toBe("user");
	});
});

describe("interactive refusal handling", () => {
	it("switches after refusal persistence, sends hidden lowercase continue, and keeps Opus selected", async () => {
		const harness = createHarness({ action: "continue" });

		await harness.handlers.message_end[0]({ message: fableRefusal }, harness.ctx);
		expect(harness.selectedModels).toEqual([]);
		expect(harness.sentMessages).toEqual([]);

		await harness.handlers.agent_end[0]({}, harness.ctx);
		expect(harness.selectedModels).toEqual([harness.fallbackModel]);
		expect(harness.sentMessages).toEqual([
			{
				message: {
					customType: "claude-refusal-continue",
					content: "continue",
					display: false,
				},
				options: { deliverAs: "followUp", triggerTurn: true },
			},
		]);
	});

	it("branches directly at agent_end and restores the draft", async () => {
		const harness = createHarness({ action: "edit", editorDraft: "steer away" });

		await harness.handlers.message_end[0]({ message: fableRefusal }, harness.ctx);

		await harness.handlers.agent_end[0]({}, harness.ctx);

		// The session should have branched — active context no longer includes
		// the tool call, tool result, or refusal
		const sessionContext = (
			harness.ctx.sessionManager as unknown as SessionManager
		).buildSessionContext();
		expect(
			sessionContext.messages.map((m: { role: string }) => m.role),
		).toEqual(["user"]);

		// Draft restored
		expect(harness.editorValues).toEqual(["steer away"]);
	});

	it("leaves the refused branch unchanged when the menu is cancelled", async () => {
		const harness = createHarness();
		await harness.handlers.message_end[0]({ message: fableRefusal }, harness.ctx);
		expect(harness.sentMessages).toEqual([]);
		expect(harness.selectedModels).toEqual([]);
	});
});

describe("automatic and non-interactive handling", () => {
	it("auto mode skips the menu and continues after the refusal is persisted", async () => {
		process.env.PI_CLAUDE_AUTH_REFUSAL_MODE = "auto";
		const harness = createHarness();
		await harness.handlers.message_end[0]({ message: fableRefusal }, harness.ctx);
		expect(harness.customCalls).toBe(0);
		expect(harness.sentMessages).toEqual([]);
		await harness.handlers.agent_end[0]({}, harness.ctx);
		expect(harness.sentMessages).toHaveLength(1);
	});

	it("ask mode stops instead of silently choosing in non-TUI modes", async () => {
		const harness = createHarness({ mode: "print" });
		await harness.handlers.message_end[0]({ message: fableRefusal }, harness.ctx);
		expect(harness.customCalls).toBe(0);
		expect(harness.sentMessages).toEqual([]);
		expect(harness.notifications).toEqual([
			{
				message:
					"Claude Fable 5 returned an Anthropic classifier refusal. Interactive refusal handling requires Pi's TUI.",
				kind: "error",
			},
		]);
	});
});
