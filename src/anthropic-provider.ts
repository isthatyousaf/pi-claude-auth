import type {
	Api,
	ApiStreamOptions,
	Context,
	Model,
	Provider,
	ProviderHeaders,
	SimpleStreamOptions,
	StreamOptions,
} from "@earendil-works/pi-ai";
import { createClaudeCodeFetch, isAnthropicOAuthToken } from "./cch.ts";
import { log } from "./logger.ts";
import { buildUserAgent } from "./signing.ts";
import { injectBillingHeader } from "./transforms.ts";

/**
 * Claude Code identity headers added to every OAuth request. The wrapped fetch
 * additionally injects `x-client-request-id` per request. Pi's built-in
 * provider already sends `user-agent: claude-cli/<version>` and `x-app: cli`
 * for OAuth tokens, but `options.headers` is merged last and wins, so the full
 * `claude-cli/<version> (external, <entrypoint>)` form overrides it.
 */
function claudeCodeHeaders(sessionId: string | undefined): ProviderHeaders {
	return {
		"user-agent": buildUserAgent(),
		"x-app": "cli",
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
 * - Chains `onPayload`: runs any pre-existing transform first, then applies
 *   {@link injectBillingHeader} (billing block + system relocation) to the
 *   result, so other extensions' payload transforms compose correctly.
 *
 * Only applied to OAuth tokens (`sk-ant-oat`); plain API-key requests pass
 * through the built-in provider untouched and bill normally on their own.
 */
function mergeClaudeCodeOptions<T extends StreamOptions>(options: T): T {
	const originalOnPayload = options.onPayload;
	const transport = options.fetch ?? globalThis.fetch;
	return {
		...options,
		// Claude Code uses ENABLE_PROMPT_CACHING_1H to force ttl="1h". Pi's
		// Anthropic provider maps cacheRetention="long" to the same wire shape.
		...(oneHourCacheEnabled() ? { cacheRetention: "long" } : {}),
		headers: { ...options.headers, ...claudeCodeHeaders(options.sessionId) },
		fetch: createClaudeCodeFetch(transport),
		onPayload: async (payload, model) => {
			const prior = await originalOnPayload?.(payload, model);
			try {
				const updated = injectBillingHeader(prior ?? payload);
				if (updated) {
					log("billing_header_injected", {});
					return updated;
				}
			} catch (err) {
				log("billing_header_error", {
					error: err instanceof Error ? err.message : String(err),
				});
			}
			return prior;
		},
	} as T;
}

/**
 * Wrap Pi's built-in Anthropic provider so OAuth requests carry the Claude Code
 * billing header, identity headers, and a real `cch` body checksum. Non-OAuth
 * (API-key) requests are delegated unchanged.
 *
 * The original provider's credential lifecycle (browser `/login`, token refresh,
 * `~/.pi/agent/auth.json` storage) is preserved: only `stream`/`streamSimple`
 * are overridden, and only the per-request options are transformed.
 */
export function wrapAnthropicProvider(provider: Provider): Provider {
	if (provider.id !== "anthropic")
		throw new Error(`pi-claude-auth cannot wrap provider "${provider.id}"`);
	return {
		...provider,
		stream<T extends Api>(
			model: Model<T>,
			context: Context,
			options?: ApiStreamOptions<T>,
		) {
			if (!options || !isAnthropicOAuthToken(options.apiKey))
				return provider.stream(model, context, options);
			return provider.stream(
				model,
				context,
				mergeClaudeCodeOptions(options),
			);
		},
		streamSimple(
			model: Model<Api>,
			context: Context,
			options?: SimpleStreamOptions,
		) {
			if (!options || !isAnthropicOAuthToken(options.apiKey))
				return provider.streamSimple(model, context, options);
			return provider.streamSimple(
				model,
				context,
				mergeClaudeCodeOptions(options),
			);
		},
	};
}
