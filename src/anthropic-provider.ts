import type {
	Api,
	AssistantMessageEventStream,
	Context,
	Model,
	Provider,
	ProviderHeaders,
	SimpleStreamOptions,
	StreamOptions,
} from "@earendil-works/pi-ai";
import { createClaudeCodeFetch, isAnthropicOAuthToken } from "./cch.ts";
import type { DeviceIdentity } from "./device-id.ts";
import { log } from "./logger.ts";
import { buildUserAgent } from "./signing.ts";
import {
	applyServerCompactionPayload,
	claimAgentRequestContext,
	contextCarriesCompactionBlock,
	isAgentPayload,
	payloadDeclarationsKey,
	recordAgentRequest,
	recordsAgentRequests,
	type RequestAuth,
	stripAnchors,
	wireMessageRecords,
} from "./server-compaction.ts";
import { injectBillingHeader } from "./transforms.ts";

/**
 * Claude Code identity headers added to every OAuth request. The wrapped fetch
 * additionally injects `x-client-request-id` and `x-claude-code-prompt-id` per
 * request. Pi's built-in provider already sends `user-agent: claude-cli/<version>`
 * and `x-app: cli` for OAuth tokens, but `options.headers` is merged last and
 * wins, so the full `claude-cli/<version> (external, <entrypoint>)` form
 * overrides it. `x-claude-code-request-class: main` marks session traffic the
 * way real first-party requests do.
 */
function claudeCodeHeaders(sessionId: string | undefined): ProviderHeaders {
	return {
		"user-agent": buildUserAgent(),
		"x-app": "cli",
		"x-claude-code-request-class": "main",
		...(sessionId ? { "x-claude-code-session-id": sessionId } : {}),
	};
}

/** Claude Code-compatible opt-in for Anthropic's one-hour prompt-cache TTL. */
function oneHourCacheEnabled(): boolean {
	const value = process.env.ENABLE_PROMPT_CACHING_1H?.trim().toLowerCase();
	return value === "1" || value === "true" || value === "yes" || value === "on";
}

/**
 * Merge Claude Code billing behavior into a single stream request's options.
 *
 * - Merges {@link claudeCodeHeaders} into `options.headers`.
 * - Wraps the transport with {@link createClaudeCodeFetch} so the `cch`
 *   placeholder is resolved against the final serialized body.
 * - Chains `onPayload` ({@link mergeServerCompactionOptions}): runs any
 *   pre-existing transform first, then the server-compaction rewrite, then
 *   {@link injectBillingHeader} (billing block + system relocation + device
 *   metadata), so other extensions' payload transforms compose correctly.
 *
 * Only applied to OAuth tokens (`sk-ant-oat`); plain API-key requests bill
 * normally on their own and get only {@link mergeServerCompactionOptions}.
 */
function mergeClaudeCodeOptions<T extends StreamOptions>(
	options: T,
	device: DeviceIdentity | null,
	context: Context,
): T {
	const transport = options.fetch ?? globalThis.fetch;
	return {
		...mergeServerCompactionOptions(options, context, "oauth", (current, systemTextPlacement) => {
			const updated = injectBillingHeader(current, {
				sessionId: options.sessionId,
				device,
				systemTextPlacement,
			});
			if (updated) log("billing_header_injected", {});
			return updated;
		}),
		// Claude Code uses ENABLE_PROMPT_CACHING_1H to force ttl="1h". Pi's
		// Anthropic provider maps cacheRetention="long" to the same wire shape.
		...(oneHourCacheEnabled() ? { cacheRetention: "long" } : {}),
		headers: { ...options.headers, ...claudeCodeHeaders(options.sessionId) },
		fetch: createClaudeCodeFetch(transport),
	} as T;
}

/**
 * The server-compaction part of a request's options, shared by OAuth and
 * API-key requests:
 *
 * - Records each agent request, which a server compaction request continues
 *   ({@link recordAgentRequest}), tied to the context hook state of that
 *   request ({@link claimAgentRequestContext}).
 * - Swaps in a server-compaction block ({@link applyServerCompactionPayload});
 *   OAuth requests then get the billing transform (`finish`).
 * - Turns off Anthropic message threads for requests that carry a block.
 *
 * Other extensions' payload transforms run first.
 */
function mergeServerCompactionOptions<T extends StreamOptions>(
	options: T,
	context: Context,
	auth: RequestAuth,
	finish?: (current: unknown, systemTextPlacement: ReturnType<typeof applyServerCompactionPayload>) => unknown,
): T {
	const originalOnPayload = options.onPayload;
	// Claimed before any await, so the hook state stays with this request.
	const agentRequest = claimAgentRequestContext(context.messages);
	return {
		...options,
		...(contextCarriesCompactionBlock(context.messages)
			? { disableContinuation: true }
			: {}),
		onPayload: async (payload, model) => {
			const prior = await originalOnPayload?.(payload, model);
			const current = prior ?? payload;
			if (
				recordsAgentRequests(model.id) &&
				(isAgentPayload(payload) || isAgentPayload(current))
			) {
				recordAgentRequest(
					agentRequest,
					options.sessionId,
					{
						modelId: model.id,
						messages: [...context.messages],
						wire: wireMessageRecords(current),
						declarationsKey: payloadDeclarationsKey(current),
						cacheRetention: options.cacheRetention,
						auth,
					},
					options.signal?.aborted,
				);
			}
			try {
				const compat = (model as { compat?: { supportsMidConvoSystemMessages?: boolean } }).compat;
				const systemTextPlacement = applyServerCompactionPayload(current, {
					auth,
					midConvoSystem: compat?.supportsMidConvoSystemMessages === true,
				});
				const updated = finish?.(current, systemTextPlacement);
				if (updated) return updated;
				if (!finish) return current;
			} catch (err) {
				log(finish ? "billing_header_error" : "server_compaction_payload_error", {
					error: err instanceof Error ? err.message : String(err),
				});
			} finally {
				stripAnchors(current);
			}
			return prior;
		},
	} as T;
}

/**
 * Options that only the api-shaped `Provider.stream` entry point carries.
 * `SimpleStreamOptions` requests derive thinking from provider-neutral fields
 * (`reasoning`, `thinkingBudgets`) instead, so their presence marks a request
 * that must keep flowing through `streamSimple`.
 */
const STREAM_ONLY_OPTION_KEYS = [
	"thinkingEnabled",
	"effort",
	"thinkingBudgetTokens",
] as const;

function isApiShapedOptions(options?: SimpleStreamOptions): boolean {
	if (!options) return false;
	// Probe opaquely: these keys exist on the api-shaped option types, and the
	// options bag crosses the composer collapsed into one untyped-in-practice
	// handler.
	const bag = options as Record<string, unknown>;
	return STREAM_ONLY_OPTION_KEYS.some((key) => bag[key] !== undefined);
}

/**
 * The catalog-preserving registration shape: a named stream overlay for pi's
 * built-in Anthropic provider. Registering this via
 * `pi.registerProvider("anthropic", overlay)` keeps pi's catalog-enabled
 * built-in provider as the base, so its `getModels()`, `refreshModels()` (the
 * pi.dev remote-catalog overlay that adds new models such as
 * `claude-sonnet-5-5`), model filtering, and OAuth lifecycle (`/login`, token
 * refresh, `~/.pi/agent/auth.json`) all stay intact. Replacing the whole
 * provider instead would drop the dynamic catalog and freeze the static
 * built-in model list.
 *
 * Pi's provider composer routes BOTH provider entry points (`stream` and
 * `streamSimple`) into the overlay's single `streamSimple` handler for every
 * `anthropic-messages` model. The handler routes back to the matching base
 * entry point: api-shaped options keep the api-shaped contract (their thinking
 * options are interpreted as-is rather than re-derived), everything else —
 * including pi's session traffic, which always arrives via `streamSimple` —
 * keeps the simple contract.
 *
 * OAuth requests (`sk-ant-oat` tokens) get the Claude Code billing header,
 * identity headers, and a real `cch` body checksum merged into their options.
 * Non-OAuth (API-key) requests bill normally on their own; they get only the
 * server-compaction options, for models that support it.
 */
export interface AnthropicStreamOverlay {
	api: Api;
	streamSimple(
		model: Model<Api>,
		context: Context,
		options?: SimpleStreamOptions,
	): AssistantMessageEventStream;
}

/**
 * Build the Claude Code stream overlay on top of pi's built-in Anthropic
 * provider, which remains the streaming delegate (its stream methods dispatch
 * straight to pi-ai's `anthropic-messages` implementation).
 *
 * `device` is the ~/.claude.json identity resolved once at startup (null when
 * absent); it rides into every OAuth request's metadata.user_id.
 */
export function createAnthropicStreamOverlay(
	base: Provider,
	device: DeviceIdentity | null = null,
): AnthropicStreamOverlay {
	if (base.id !== "anthropic")
		throw new Error(
			`pi-claude-auth cannot overlay provider "${base.id}"`,
		);
	// The options bag passes through opaquely: the composer collapsed the two
	// entry points into one handler, so neither pi-ai parameter type fits both.
	type StreamDelegate = (
		model: Model<Api>,
		context: Context,
		options?: SimpleStreamOptions,
	) => AssistantMessageEventStream;
	return {
		api: "anthropic-messages",
		streamSimple(model, context, options) {
			const delegate: StreamDelegate = isApiShapedOptions(options)
				? (m, c, o) => base.stream(m, c, o as never)
				: (m, c, o) => base.streamSimple(m, c, o);
			if (!options) return delegate(model, context, options);
			if (isAnthropicOAuthToken(options.apiKey))
				return delegate(model, context, mergeClaudeCodeOptions(options, device, context));
			// API-key requests bill on their own; they only get server compaction.
			if (options.apiKey && recordsAgentRequests(model.id))
				return delegate(model, context, mergeServerCompactionOptions(options, context, "api-key"));
			return delegate(model, context, options);
		},
	};
}
