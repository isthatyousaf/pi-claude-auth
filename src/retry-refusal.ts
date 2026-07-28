import type {
	ExtensionAPI,
	ExtensionContext,
	SessionManager,
} from "@earendil-works/pi-coding-agent";
import { showRefusalMenu } from "./refusal-ui.ts";

const OPUS_FALLBACK_MODEL_ID = "claude-opus-4-8";
const CONTINUE_MESSAGE_TYPE = "claude-refusal-continue";

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
 * Compute the branch target that skips both the trigger event and the refusal.
 *
 * - Tool-result trigger: walk back through consecutive tool results to find the
 *   assistant tool-call message that started the batch, then target its parent.
 * - User-message trigger: target the user message itself so Pi rewinds to its
 *   parent and restores its text for editing.
 * - Anything else: target the trigger's parent.
 */
export function computeBranchTarget(
	sessionManager: ExtensionContext["sessionManager"],
	triggerId: string,
): string {
	const trigger = sessionManager.getEntry(triggerId) as SessionMessageEntry | undefined;
	if (!trigger || trigger.type !== "message") return triggerId;

	if (trigger.message.role === "toolResult") {
		let currentId: string | null = triggerId;
		while (currentId) {
			const entry = sessionManager.getEntry(currentId) as SessionMessageEntry | undefined;
			if (!entry || entry.type !== "message" || entry.message.role !== "toolResult")
				break;
			const parentId = entry.parentId;
			if (!parentId) break;
			const parent = sessionManager.getEntry(parentId) as SessionMessageEntry | undefined;
			if (parent?.type === "message" && parent.message.role === "toolResult") {
				currentId = parentId;
				continue;
			}
			return parent?.parentId ?? parentId;
		}
	}

	if (trigger.message.role === "user") {
		return triggerId;
	}

	return trigger.parentId ?? triggerId;
}

type PendingAction =
	| { type: "continue"; refusedModelName: string }
	| {
			type: "edit";
			targetId: string;
			draft: string;
	  };

export function registerRetryAfterRefusal(pi: ExtensionAPI): void {
	const state = createRetryAfterRefusalState();
	let pending: PendingAction | undefined;
	let needsContextRebuild = false;

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

		// Edit: branch the session tree directly. The refusal has been
		// persisted as a child of the trigger entry. Branching moves the leaf
		// to the safe parent, abandoning the trigger event and the refusal to
		// an alternate branch. The context event below rebuilds the message
		// list so the model sees the correct state despite
		// agent.state.messages being stale.
		//
		// KNOWN LIMITATION: sessionManager.branch() moves the persisted leaf
		// but does NOT rebuild agent.state.messages or emit session_tree, so
		// the TUI transcript stays stale until the next full render cycle.
		// navigateTree() (which does both) is only available on
		// ExtensionCommandContext, not ExtensionContext. See handoff doc for
		// the full investigation and remaining options.
		const sm = ctx.sessionManager as unknown as SessionManager;
		try {
			sm.branch(action.targetId);
		} catch {
			ctx.ui.notify("Could not branch to the point before the refusal.", "error");
			state.complete();
			return;
		}

		needsContextRebuild = true;
		ctx.ui.setEditorText(action.draft);
		state.complete();
	});

	// Safety net: after a direct branch, agent.state.messages is stale. The
	// context event fires before every LLM call and can replace the message
	// list entirely. We rebuild from the session tree, which has the correct
	// active branch.
	pi.on("context", async (_event, ctx) => {
		if (!needsContextRebuild) return undefined;
		needsContextRebuild = false;
		const sessionContext = (
			ctx.sessionManager as unknown as SessionManager
		).buildSessionContext();
		return { messages: sessionContext.messages };
	});
}
