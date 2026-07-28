import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	createFauxCore,
	fauxAssistantMessage,
	fauxText,
	fauxThinking,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";

const REFUSAL_ERROR_MESSAGE =
	"The request was blocked by a safety classifier. Refusals-and-fallback mechanism triggered.";
const RECOVERY_RESPONSE_COUNT = 20;

type ProviderModels = NonNullable<
	Parameters<ExtensionAPI["registerProvider"]>[1]["models"]
>;

const SIMULATOR_MODELS: ProviderModels = [
	{
		id: "claude-fable-5",
		name: "Claude Fable 5 (refusal simulator)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200_000,
		maxTokens: 16_384,
	},
	{
		id: "claude-opus-4-8",
		name: "Claude Opus 4.8 (refusal simulator)",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200_000,
		maxTokens: 16_384,
	},
];

const wait = (ms: number) =>
	new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * Deterministic Anthropic refusal simulator for manually exercising the real
 * pi-claude-auth refusal UI without making an external model request.
 *
 * Sequence:
 * 1. assistant thinking + text + baseline tool A
 * 2. assistant thinking + text + tool B
 * 3. assistant thinking + text + separate tool C
 * 4. classifier refusal
 * 5+. normal response from whichever model the user chose for recovery
 */
export default function refusalSimulator(pi: ExtensionAPI): void {
	const faux = createFauxCore({
		provider: "anthropic",
		api: "anthropic-messages",
		tokensPerSecond: 120,
		models: SIMULATOR_MODELS,
	});

	const recoveryResponse = (
		_context: unknown,
		_options: unknown,
		state: { callCount: number },
		model: { name: string },
	) =>
		fauxAssistantMessage([
			fauxThinking("The refusal was handled and the safe branch is active."),
			fauxText(
				`Recovery response from ${model.name} on provider call ${state.callCount}.`,
			),
		]);

	faux.setResponses([
		fauxAssistantMessage(
			[
				fauxThinking(
					"First inspect the baseline configuration and preserve what is already known.",
				),
				fauxText("I found an earlier lead. I will read the baseline before continuing."),
				fauxToolCall(
					"refusal_sim_read",
					{ path: "baseline.txt" },
					{ id: "refusal-sim-baseline-read" },
				),
			],
			{ stopReason: "toolUse" },
		),
		fauxAssistantMessage(
			[
				fauxThinking("Now correlate the baseline with the deployment logs."),
				fauxText("The baseline is useful. I will verify it against tool B."),
				fauxToolCall(
					"refusal_sim_shell",
					{ command: "scan deployment logs" },
					{ id: "refusal-sim-log-scan" },
				),
			],
			{ stopReason: "toolUse" },
		),
		fauxAssistantMessage(
			[
				fauxThinking(
					"The log result is complete. One separate tool C check may explain the failure.",
				),
				fauxText("I will inspect the deployed artifact with tool C now."),
				fauxToolCall(
					"refusal_sim_read",
					{ path: "deployed-artifact.txt" },
					{ id: "refusal-sim-artifact-read" },
				),
			],
			{ stopReason: "toolUse" },
		),
		fauxAssistantMessage("REFUSAL_BODY", {
			stopReason: "error",
			errorMessage: REFUSAL_ERROR_MESSAGE,
		}),
		...Array.from({ length: RECOVERY_RESPONSE_COUNT }, () => recoveryResponse),
	]);

	pi.registerProvider("anthropic", {
		name: "Anthropic refusal simulator",
		baseUrl: "http://localhost:0",
		apiKey: "refusal-simulator-local-only",
		api: "anthropic-messages",
		models: SIMULATOR_MODELS,
		streamSimple: faux.streamSimple,
	});

	pi.registerTool({
		name: "refusal_sim_read",
		label: "Refusal simulator read",
		description: "Return deterministic file contents for the refusal simulator",
		parameters: Type.Object({ path: Type.String() }),
		async execute(_toolCallId, params) {
			return {
				content: [
					{
						type: "text",
						text:
							params.path === "baseline.txt"
								? "BASELINE_RESULT: completed earlier work that should survive the retry"
								: "ARTIFACT_RESULT: output from the final batch that should be abandoned",
					},
				],
				details: { path: params.path },
			};
		},
	});

	pi.registerTool({
		name: "refusal_sim_shell",
		label: "Refusal simulator shell",
		description: "Return deterministic log output for the refusal simulator",
		parameters: Type.Object({ command: Type.String() }),
		async execute(_toolCallId, params) {
			await wait(120);
			return {
				content: [
					{
						type: "text",
						text: "LOG_RESULT: completed tool B output that should survive the retry",
					},
				],
				details: { command: params.command },
			};
		},
	});
}
