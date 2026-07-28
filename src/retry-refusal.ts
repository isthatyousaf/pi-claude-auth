import type {
	ExtensionAPI,
	ExtensionContext,
	SessionManager,
} from "@earendil-works/pi-coding-agent";
import { showRefusalMenu } from "./refusal-ui.ts";

const OPUS_FALLBACK_MODEL_ID = "claude-opus-4-8";
const CONTINUE_MESSAGE_TYPE = "claude-refusal-continue";
const BRANCH_ENTRY_TYPE = "claude-refusal-branch";

type RefusalMode = "ask" | "auto";

export interface AssistantRefusalCandidate {
	role?: unknown;
	provider?: unknown;
	model?: unknown;
	stopReason?: unknown;
	errorMessage?: unknown;
}

function isAnthropicFableOrOpus5(model: unknown): boolean {
	if (typeof model !== "string") return false;
	const id = model.toLowerCase();
	return id.includes("claude-fable-5") || id.includes("claude-opus-5");
}

function isRefusalError(message: unknown): boolean {
	if (typeof message !== "string") return false;
	return /refus|classifier|safety|safeguard|usage policy|violative|refusals-and-fallback/i.test(
		message,
	);
}

export function shouldHandleRefusal(
	message: AssistantRefusalCandidate,
): boolean {
	return (
		message.role === "assistant" &&
		message.provider === "anthropic" &&
		isAnthropicFableOrOpus5(message.model) &&
		message.stopReason === "error" &&
		isRefusalError(message.errorMessage)
	);
}

export function getRefusalMode(): RefusalMode {
	return process.env.PI_CLAUDE_AUTH_REFUSAL_MODE?.toLowerCase() === "auto"
		? "auto"
		: "ask";
}

export function createRetryAfterRefusalState() {
	let handlingRefusal = false;
	return {
		begin(message: AssistantRefusalCandidate): boolean {
			if (handlingRefusal || !shouldHandleRefusal(message)) return false;
			handlingRefusal = true;
			return true;
		},
		complete(): void {
			handlingRefusal = false;
		},
	};
}

function modelField(model: unknown, field: "id" | "name"): string | undefined {
	if (!model || typeof model !== "object") return undefined;
	const value = (model as Record<string, unknown>)[field];
	return typeof value === "string" && value.trim().length > 0
		? value
		: undefined;
}

function displayModelName(
	ctx: ExtensionContext,
	provider: string,
	modelId: unknown,
): string {
	if (typeof modelId !== "string" || modelId.trim().length === 0)
		return "unknown model";

	const selectedId = modelField(ctx.model, "id");
	const selectedName = modelField(ctx.model, "name");
	if (selectedId === modelId && selectedName) return selectedName;

	const registryModel = ctx.modelRegistry.find(provider, modelId);
	return modelField(registryModel, "name") ?? modelId;
}

interface SessionMessageEntry {
	type: "message";
	id: string;
	parentId: string | null;
	message: { role?: string };
}

/**
 * Compute the active leaf that skips both the trigger event and the refusal.
 *
 * - Tool-result trigger: walk back through consecutive tool results to find the
 *   assistant tool-call message that started the batch, then keep its parent.
 * - User-message trigger: keep its parent so the triggering prompt is removed.
 * - Anything else: keep the trigger's parent.
 *
 * A null result means the trigger was the root entry and the leaf must be reset.
 */
export function computeBranchTarget(
	sessionManager: ExtensionContext["sessionManager"],
	triggerId: string,
): string | null {
	const trigger = sessionManager.getEntry(triggerId) as SessionMessageEntry | undefined;
	if (!trigger || trigger.type !== "message") return triggerId;

	if (trigger.message.role === "toolResult") {
		let currentId: string | null = triggerId;
		while (currentId) {
			const entry = sessionManager.getEntry(currentId) as SessionMessageEntry | undefined;
			if (!entry || entry.type !== "message" || entry.message.role !== "toolResult")
				break;
			const parentId = entry.parentId;
			if (!parentId) return null;
			const parent = sessionManager.getEntry(parentId) as SessionMessageEntry | undefined;
			if (parent?.type === "message" && parent.message.role === "toolResult") {
				currentId = parentId;
				continue;
			}
			return parent?.type === "message" ? parent.parentId : parentId;
		}
	}

	return trigger.parentId;
}

function hasRefusalBranchMarker(
	sessionManager: ExtensionContext["sessionManager"],
): boolean {
	return sessionManager
		.getBranch()
		.some(
			(entry) =>
				entry.type === "custom" && entry.customType === BRANCH_ENTRY_TYPE,
		);
}

type PendingAction =
	| { type: "continue"; refusedModelName: string }
	| {
			type: "edit";
			triggerId: string;
			targetId: string | null;
			draft: string;
	  };

export function registerRetryAfterRefusal(pi: ExtensionAPI): void {
	const state = createRetryAfterRefusalState();
	let pending: PendingAction | undefined;
	let needsContextRebuild = false;

	pi.on("session_start", (_event, ctx) => {
		needsContextRebuild = hasRefusalBranchMarker(ctx.sessionManager);
	});

	async function continueWithFallback(
		ctx: ExtensionContext,
		refusedModelName: string,
	): Promise<void> {
		const fallbackModel = ctx.modelRegistry.find(
			"anthropic",
			OPUS_FALLBACK_MODEL_ID,
		);
		const fallbackModelName = displayModelName(
			ctx,
			"anthropic",
			OPUS_FALLBACK_MODEL_ID,
		);
		if (!fallbackModel) {
			state.complete();
			ctx.ui.notify(
				`${refusedModelName} refusal detected, but ${fallbackModelName} could not be selected.`,
				"error",
			);
			return;
		}

		const switched = await pi.setModel(fallbackModel);
		if (!switched) {
			state.complete();
			ctx.ui.notify(
				`${refusedModelName} refusal detected, but ${fallbackModelName} could not be selected.`,
				"error",
			);
			return;
		}

		state.complete();
		ctx.ui.notify(
			`Switched to ${fallbackModelName} and continuing from the current state.`,
			"warning",
		);
		pi.sendMessage(
			{
				customType: CONTINUE_MESSAGE_TYPE,
				content: "continue",
				display: false,
			},
			{ deliverAs: "followUp", triggerTurn: true },
		);
	}

	pi.on("message_end", async (event, ctx) => {
		const message = event.message as AssistantRefusalCandidate;
		if (message.role === "user") {
			pending = undefined;
			return;
		}
		if (!state.begin(message)) return;

		const refusedModelName = displayModelName(ctx, "anthropic", message.model);
		if (getRefusalMode() === "auto") {
			pending = { type: "continue", refusedModelName };
			return;
		}

		if (ctx.mode !== "tui") {
			state.complete();
			ctx.ui.notify(
				`${refusedModelName} returned an Anthropic classifier refusal. Interactive refusal handling requires Pi's TUI.`,
				"error",
			);
			return;
		}

		const fallbackModelName = displayModelName(
			ctx,
			"anthropic",
			OPUS_FALLBACK_MODEL_ID,
		);
		const triggerId = ctx.sessionManager.getLeafId();
		const action = await showRefusalMenu(
			ctx,
			refusedModelName,
			fallbackModelName,
			triggerId !== null,
		);

		if (action === "continue") {
			pending = { type: "continue", refusedModelName };
			return;
		}
		if (action === "edit" && triggerId) {
			state.complete();
			pending = {
				type: "edit",
				triggerId,
				targetId: computeBranchTarget(ctx.sessionManager, triggerId),
				draft: ctx.ui.getEditorText(),
			};
			return;
		}

		state.complete();
	});

	pi.on("agent_end", async (_event, ctx) => {
		const action = pending;
		pending = undefined;
		if (!action) return;

		if (action.type === "continue") {
			await continueWithFallback(ctx, action.refusedModelName);
			return;
		}

		// Edit: move the session leaf to the safe parent, then append a hidden
		// extension entry there. SessionManager persists entries, not leaf moves,
		// so the marker makes the selected branch survive reopening the session.
		// The context hook below keeps provider requests on that branch while
		// Pi's private agent state remains stale.
		//
		// KNOWN LIMITATION: direct SessionManager mutation cannot rebuild Pi's
		// visible transcript. It stays stale until a supported tree navigation,
		// compaction, reload, or session replacement reconstructs the TUI.
		const sm = ctx.sessionManager as unknown as SessionManager;
		try {
			if (action.targetId === null) sm.resetLeaf();
			else sm.branch(action.targetId);
		} catch {
			ctx.ui.notify("Could not branch to the point before the refusal.", "error");
			state.complete();
			return;
		}

		try {
			pi.appendEntry(BRANCH_ENTRY_TYPE, {
				triggerId: action.triggerId,
				targetId: action.targetId,
			});
		} catch {
			ctx.ui.notify(
				"Branched before the refusal, but could not persist that branch selection.",
				"warning",
			);
		}

		needsContextRebuild = true;
		ctx.ui.setEditorText(action.draft);
		state.complete();
	});

	// Safety net: direct branching does not mutate agent.state.messages. Rebuild
	// every provider context from the active session branch until Pi performs an
	// operation that synchronizes its own state.
	pi.on("context", async (_event, ctx) => {
		if (!needsContextRebuild) return undefined;
		const sessionContext = (
			ctx.sessionManager as unknown as SessionManager
		).buildSessionContext();
		return { messages: sessionContext.messages };
	});

	const markContextSynchronized = () => {
		needsContextRebuild = false;
	};
	pi.on("session_tree", markContextSynchronized);
	pi.on("session_compact", markContextSynchronized);
}
