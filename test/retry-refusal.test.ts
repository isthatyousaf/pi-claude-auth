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
	getRefusalMode,
	registerRetryAfterRefusal,
	shouldHandleRefusal,
} from "../src/retry-refusal.ts";

type Handler = (...args: unknown[]) => unknown;

const ZERO_USAGE = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

type ToolCall = { id: string; name: string };

function appendUser(session: SessionManager, content: string): string {
	return session.appendMessage({
		role: "user",
		content,
		timestamp: Date.now(),
	});
}

function appendToolCalls(
	session: SessionManager,
	calls: ToolCall[],
	text?: string,
): string {
	return session.appendMessage({
		role: "assistant",
		content: [
			...(text ? [{ type: "text" as const, text }] : []),
			...calls.map((call) => ({
				type: "toolCall" as const,
				id: call.id,
				name: call.name,
				arguments: {},
			})),
		],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-fable-5",
		usage: ZERO_USAGE,
		stopReason: "toolUse",
		timestamp: Date.now(),
	});
}

function appendToolResult(
	session: SessionManager,
	call: ToolCall,
	text: string,
): string {
	return session.appendMessage({
		role: "toolResult",
		toolCallId: call.id,
		toolName: call.name,
		content: [{ type: "text", text }],
		isError: false,
		timestamp: Date.now(),
	});
}

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
	const menuChoices: string[][] = [];
	let menuCalls = 0;
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
			appendUser(session, "Do the task");
			if (options.trigger !== "user") {
				const call = { id: "t1", name: "read" };
				appendToolCalls(session, [call]);
				appendToolResult(session, call, "binary contents");
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

	const mode = options.mode ?? "tui";
	const ctx = {
		mode,
		hasUI: mode === "tui" || mode === "rpc",
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
			async select(_title: string, choices: string[]) {
				menuCalls++;
				menuChoices.push(choices);
				if (options.action === "continue") return choices[0];
				if (options.action === "edit") return choices[1];
				return undefined;
			},
		},
	};

	registerRetryAfterRefusal(pi);
	return {
		appendedEntries,
		ctx,
		editorValues,
		fallbackModel,
		get menuCalls() {
			return menuCalls;
		},
		menuChoices,
		handlers,
		notifications,
		/**
		 * Persist the refusal like Pi does, then fire agent_end carrying the same
		 * message object Pi stored. `beforeHandlers` simulates another extension
		 * appending an entry — and moving the leaf — ahead of this one.
		 */
		async refuse(beforeHandlers?: () => void) {
			const refusal = {
				...fableRefusal,
				content: [],
				api: "anthropic-messages",
				usage: ZERO_USAGE,
				timestamp: Date.now(),
			};
			const refusalId = session.appendMessage(refusal);
			beforeHandlers?.();
			for (const handler of handlers.agent_end ?? []) {
				await handler({ messages: [refusal] }, ctx);
			}
			return refusalId;
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

	it("handles the wording Anthropic and Pi actually send", () => {
		// Anthropic's documented explanation text, which never says "refusal".
		for (const category of ["cyber", "biological", "frontier model"]) {
			expect(
				shouldHandleRefusal({
					...fableRefusal,
					errorMessage: `This request was declined because it could enable ${category} harm.`,
				}),
			).toBe(true);
		}
		// Pi's substitute when Anthropic sends no explanation.
		expect(
			shouldHandleRefusal({
				...fableRefusal,
				errorMessage: "The model refused to complete the request",
			}),
		).toBe(true);
	});

	it("does not handle other model families or unrelated errors", () => {
		expect(
			shouldHandleRefusal({ ...fableRefusal, model: "claude-sonnet-4-5" }),
		).toBe(false);
		for (const errorMessage of [
			"network timeout",
			"overloaded_error: Overloaded",
			"503 service unavailable",
			"fetch failed",
		]) {
			expect(shouldHandleRefusal({ ...fableRefusal, errorMessage })).toBe(false);
		}
	});

	it("ignores runs that did not end in a refusal", async () => {
		const harness = createHarness({ action: "continue" });
		await harness.handlers.agent_end[0](
			{
				messages: [
					{ role: "user", content: "hi" },
					{
						...fableRefusal,
						model: "claude-opus-4-8",
						stopReason: "stop",
						errorMessage: undefined,
					},
				],
			},
			harness.ctx,
		);
		expect(harness.menuCalls).toBe(0);
		expect(harness.sentMessages).toEqual([]);
	});

	it("does not reopen the menu on the continuation run", async () => {
		const harness = createHarness({ action: "continue" });
		await harness.refuse();
		expect(harness.menuCalls).toBe(1);

		// The Opus 4.8 continuation ends in a normal assistant message.
		await harness.handlers.agent_end[0](
			{
				messages: [
					{
						role: "assistant",
						provider: "anthropic",
						model: "claude-opus-4-8",
						stopReason: "stop",
					},
				],
			},
			harness.ctx,
		);
		expect(harness.menuCalls).toBe(1);
		expect(harness.sentMessages).toHaveLength(1);
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
		appendUser(session, "Do the task");
		const call = { id: "t1", name: "read" };
		appendToolCalls(session, [call]);
		appendToolResult(session, call, "file contents");
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
		const userId = appendUser(session, "Risky request");
		const target = computeBranchTarget(session, userId);
		expect(target).toBeNull();
	});

	it("walks back through multiple tool results to find the batch start", () => {
		const session = SessionManager.inMemory();
		appendUser(session, "Do multi-step task");
		const readCall = { id: "a", name: "read" };
		const bashCall = { id: "b", name: "bash" };
		appendToolCalls(session, [readCall, bashCall]);
		appendToolResult(session, readCall, "output a");
		appendToolResult(session, bashCall, "output b");
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

		await harness.refuse();
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

		await harness.refuse();

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
				const baselineCall = { id: "baseline-read", name: "read" };
				const logCall = { id: "log-scan", name: "bash" };
				const artifactCall = { id: "artifact-read", name: "read" };

				appendUser(session, "Investigate the failing deployment");
				appendToolCalls(
					session,
					[baselineCall],
					"I found an earlier lead worth preserving.",
				);
				appendToolResult(session, baselineCall, "baseline configuration");
				appendToolCalls(
					session,
					[logCall],
					"The baseline is useful; I will verify it against the logs.",
				);
				toolBResultId = appendToolResult(session, logCall, "log scan output");
				toolCAssistantId = appendToolCalls(
					session,
					[artifactCall],
					"I will inspect the deployed artifact now.",
				);
				toolCResultId = appendToolResult(
					session,
					artifactCall,
					"artifact contents",
				);
			},
		});

		const refusalId = await harness.refuse();

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

		appendUser(harness.session, "Use the completed log scan and avoid tool C");
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

	it("branches from the refusal entry even when another extension moved the leaf", async () => {
		const harness = createHarness({ action: "edit" });
		await harness.refuse(() => {
			harness.session.appendCustomEntry("other-extension", { note: "noise" });
		});

		expect(
			harness.session
				.buildSessionContext()
				.messages.map((message) => message.role),
		).toEqual(["user"]);
		expect(harness.session.getLeafEntry()).toMatchObject({
			type: "custom",
			customType: "claude-refusal-branch",
		});
	});

	it("removes a root user trigger instead of keeping it on the active branch", async () => {
		const harness = createHarness({ action: "edit", trigger: "user" });

		await harness.refuse();

		expect(harness.session.buildSessionContext().messages).toEqual([]);
		expect(harness.session.getLeafEntry()).toMatchObject({
			type: "custom",
			parentId: null,
			customType: "claude-refusal-branch",
		});
	});

	it("rebuilds every later provider context until Pi performs supported tree navigation", async () => {
		const harness = createHarness({ action: "edit" });
		await harness.refuse();

		const first = (await harness.handlers.context[0]({}, harness.ctx)) as {
			messages: Array<{ role: string }>;
		};
		expect(first.messages.map((message) => message.role)).toEqual(["user"]);

		appendUser(harness.session, "Try a safer approach");
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
		await first.refuse();

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
		await harness.refuse();

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
		const refusalId = await harness.refuse();
		expect(harness.sentMessages).toEqual([]);
		expect(harness.selectedModels).toEqual([]);
		expect(harness.session.getLeafId()).toBe(refusalId);
		expect(harness.appendedEntries).toEqual([]);
	});
});

describe("automatic and non-interactive handling", () => {
	it("auto mode skips the menu and continues after the refusal is persisted", async () => {
		process.env.PI_CLAUDE_AUTH_REFUSAL_MODE = "auto";
		const harness = createHarness();
		await harness.refuse();
		expect(harness.menuCalls).toBe(0);
		expect(harness.sentMessages).toHaveLength(1);
	});

	it("ask mode stops instead of silently choosing without an interactive UI", async () => {
		const harness = createHarness({ mode: "print" });
		await harness.refuse();
		expect(harness.menuCalls).toBe(0);
		expect(harness.sentMessages).toEqual([]);
		expect(harness.notifications).toEqual([
			{
				message:
					"Claude Fable 5 returned an Anthropic classifier refusal. Interactive refusal handling requires an interactive UI.",
				kind: "error",
			},
		]);
	});
});
