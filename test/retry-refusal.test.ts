import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
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
} as const;

interface HarnessOptions {
	action?: "continue" | "edit";
	mode?: "tui" | "rpc" | "json" | "print";
	refusedModel?: { provider: string; id: string; name: string };
	editorDraft?: string;
	session?: SessionManager;
	trigger?: "toolResult" | "user";
	initializeSession?: boolean;
	setupSession?: (session: SessionManager) => void;
}

function createHarness(options: HarnessOptions = {}) {
	const handlers: Record<string, Handler[]> = {};
	const notifications: unknown[] = [];
	const sentMessages: unknown[] = [];
	const appendedEntries: unknown[] = [];
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

	const session = options.session ?? SessionManager.inMemory();
	if (options.initializeSession !== false) {
		if (options.setupSession) {
			options.setupSession(session);
		} else {
			session.appendMessage({
				role: "user",
				content: "Do the task",
				timestamp: Date.now(),
			});
			if (options.trigger !== "user") {
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
			}
		}
	}
	const triggerId = session.getLeafId()!;

	const pi = {
		on(event: string, handler: Handler) {
			handlers[event] = [...(handlers[event] ?? []), handler];
		},
		appendEntry(customType: string, data: unknown) {
			appendedEntries.push({ customType, data });
			session.appendCustomEntry(customType, data);
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
		appendedEntries,
		ctx,
		editorValues,
		fallbackModel,
		get customCalls() {
			return customCalls;
		},
		handlers,
		notifications,
		persistRefusal() {
			return session.appendMessage({
				...fableRefusal,
				content: [],
				api: "anthropic-messages",
				usage: {
					input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				timestamp: Date.now(),
			});
		},
		selectedModels,
		sentMessages,
		session,
		triggerId,
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
		expect(target).not.toBeNull();
		const targetEntry = session.getEntry(target!);
		expect(targetEntry?.type).toBe("message");
		expect((targetEntry as { message: { role: string } }).message.role).toBe("user");
	});

	it("targets the parent when the trigger is a user message", () => {
		const session = SessionManager.inMemory();
		const userId = session.appendMessage({
			role: "user",
			content: "Risky request",
			timestamp: Date.now(),
		});
		const target = computeBranchTarget(session, userId);
		expect(target).toBeNull();
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
		expect(target).not.toBeNull();
		const targetEntry = session.getEntry(target!);
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
		harness.persistRefusal();

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

	it("branches directly at agent_end, persists the branch, and restores the draft", async () => {
		const harness = createHarness({ action: "edit", editorDraft: "steer away" });

		await harness.handlers.message_end[0]({ message: fableRefusal }, harness.ctx);
		harness.persistRefusal();
		await harness.handlers.agent_end[0]({}, harness.ctx);

		expect(
			harness.session.buildSessionContext().messages.map((message) => message.role),
		).toEqual(["user"]);
		expect(harness.session.getLeafEntry()).toMatchObject({
			type: "custom",
			customType: "claude-refusal-branch",
			data: { triggerId: harness.triggerId },
		});
		expect(harness.appendedEntries).toHaveLength(1);
		expect(harness.editorValues).toEqual(["steer away"]);
	});

	it("preserves tool B and rolls back the separate tool C turn", async () => {
		let toolBResultId = "";
		let toolCAssistantId = "";
		let toolCResultId = "";
		const harness = createHarness({
			action: "edit",
			editorDraft: "change course before tool C",
			setupSession(session) {
				session.appendMessage({
					role: "user",
					content: "Investigate the failing deployment",
					timestamp: Date.now(),
				});
				session.appendMessage({
					role: "assistant",
					content: [
						{ type: "thinking", thinking: "Inspect the baseline configuration first." },
						{ type: "text", text: "I found an earlier lead worth preserving." },
						{
							type: "toolCall",
							id: "baseline-read",
							name: "read",
							arguments: { path: "baseline.txt" },
						},
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
					toolCallId: "baseline-read",
					toolName: "read",
					content: [{ type: "text", text: "baseline configuration" }],
					isError: false,
					timestamp: Date.now(),
				});
				session.appendMessage({
					role: "assistant",
					content: [
						{ type: "thinking", thinking: "Check the deployment logs next." },
						{ type: "text", text: "The baseline is useful; I will verify it against the logs." },
						{
							type: "toolCall",
							id: "log-scan",
							name: "bash",
							arguments: { command: "scan logs" },
						},
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
				toolBResultId = session.appendMessage({
					role: "toolResult",
					toolCallId: "log-scan",
					toolName: "bash",
					content: [{ type: "text", text: "log scan output" }],
					isError: false,
					timestamp: Date.now(),
				});
				toolCAssistantId = session.appendMessage({
					role: "assistant",
					content: [
						{ type: "thinking", thinking: "One final artifact check may explain the failure." },
						{ type: "text", text: "I will inspect the deployed artifact now." },
						{
							type: "toolCall",
							id: "artifact-read",
							name: "read",
							arguments: { path: "artifact.txt" },
						},
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
				toolCResultId = session.appendMessage({
					role: "toolResult",
					toolCallId: "artifact-read",
					toolName: "read",
					content: [{ type: "text", text: "artifact contents" }],
					isError: false,
					timestamp: Date.now(),
				});
			},
		});

		await harness.handlers.message_end[0]({ message: fableRefusal }, harness.ctx);
		const refusalId = harness.persistRefusal();
		await harness.handlers.agent_end[0]({}, harness.ctx);

		const activeBranchIds = harness.session.getBranch().map((entry) => entry.id);
		expect(activeBranchIds).toContain(toolBResultId);
		expect(activeBranchIds).not.toContain(toolCAssistantId);
		expect(activeBranchIds).not.toContain(toolCResultId);
		expect(activeBranchIds).not.toContain(refusalId);

		const activeContext = harness.session.buildSessionContext().messages;
		expect(activeContext.map((message) => message.role)).toEqual([
			"user",
			"assistant",
			"toolResult",
			"assistant",
			"toolResult",
		]);
		const activeToolCallIds = activeContext.flatMap((message) =>
			message.role === "assistant"
				? message.content
						.filter((content) => content.type === "toolCall")
						.map((content) => content.id)
				: [],
		);
		const activeToolResultIds = activeContext.flatMap((message) =>
			message.role === "toolResult" ? [message.toolCallId] : [],
		);
		expect(activeToolCallIds).toEqual(["baseline-read", "log-scan"]);
		expect(activeToolResultIds).toEqual(activeToolCallIds);
		expect(harness.session.getLeafEntry()).toMatchObject({
			type: "custom",
			parentId: toolBResultId,
			customType: "claude-refusal-branch",
		});
		expect(harness.editorValues).toEqual(["change course before tool C"]);

		harness.session.appendMessage({
			role: "user",
			content: "Use the completed log scan and avoid tool C",
			timestamp: Date.now(),
		});
		const nextProviderContext = (await harness.handlers.context[0]({}, harness.ctx)) as {
			messages: Array<{ role: string }>;
		};
		expect(nextProviderContext.messages.map((message) => message.role)).toEqual([
			"user",
			"assistant",
			"toolResult",
			"assistant",
			"toolResult",
			"user",
		]);
	});

	it("removes a root user trigger instead of keeping it on the active branch", async () => {
		const harness = createHarness({ action: "edit", trigger: "user" });

		await harness.handlers.message_end[0]({ message: fableRefusal }, harness.ctx);
		harness.persistRefusal();
		await harness.handlers.agent_end[0]({}, harness.ctx);

		expect(harness.session.buildSessionContext().messages).toEqual([]);
		expect(harness.session.getLeafEntry()).toMatchObject({
			type: "custom",
			parentId: null,
			customType: "claude-refusal-branch",
		});
	});

	it("rebuilds every later provider context until Pi performs supported tree navigation", async () => {
		const harness = createHarness({ action: "edit" });
		await harness.handlers.message_end[0]({ message: fableRefusal }, harness.ctx);
		harness.persistRefusal();
		await harness.handlers.agent_end[0]({}, harness.ctx);

		const first = (await harness.handlers.context[0]({}, harness.ctx)) as {
			messages: Array<{ role: string }>;
		};
		expect(first.messages.map((message) => message.role)).toEqual(["user"]);

		harness.session.appendMessage({
			role: "user",
			content: "Try a safer approach",
			timestamp: Date.now(),
		});
		const second = (await harness.handlers.context[0]({}, harness.ctx)) as {
			messages: Array<{ role: string }>;
		};
		expect(second.messages.map((message) => message.role)).toEqual([
			"user",
			"user",
		]);

		await harness.handlers.session_tree[0]({}, harness.ctx);
		expect(await harness.handlers.context[0]({}, harness.ctx)).toBeUndefined();
	});

	it("restores context repair after the extension reloads", async () => {
		const first = createHarness({ action: "edit" });
		await first.handlers.message_end[0]({ message: fableRefusal }, first.ctx);
		first.persistRefusal();
		await first.handlers.agent_end[0]({}, first.ctx);

		const reloaded = createHarness({
			session: first.session,
			initializeSession: false,
		});
		await reloaded.handlers.session_start[0]({ reason: "reload" }, reloaded.ctx);
		const context = (await reloaded.handlers.context[0]({}, reloaded.ctx)) as {
			messages: Array<{ role: string }>;
		};
		expect(context.messages.map((message) => message.role)).toEqual(["user"]);
	});

	it("reopens on the selected branch instead of the abandoned refusal", async () => {
		const sessionDir = mkdtempSync(join(tmpdir(), "pi-claude-auth-refusal-"));
		try {
			const session = SessionManager.create("/tmp/refusal-test", sessionDir);
			const harness = createHarness({ action: "edit", session });
			await harness.handlers.message_end[0]({ message: fableRefusal }, harness.ctx);
			harness.persistRefusal();
			await harness.handlers.agent_end[0]({}, harness.ctx);

			const reopened = SessionManager.open(session.getSessionFile()!, sessionDir);
			expect(
				reopened.buildSessionContext().messages.map((message) => message.role),
			).toEqual(["user"]);
			expect(reopened.getLeafEntry()).toMatchObject({
				type: "custom",
				customType: "claude-refusal-branch",
			});
		} finally {
			rmSync(sessionDir, { recursive: true, force: true });
		}
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
		harness.persistRefusal();
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
