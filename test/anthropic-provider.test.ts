import { describe, expect, it } from "bun:test";
import type { Provider } from "@earendil-works/pi-ai";
import { wrapAnthropicProvider } from "../src/anthropic-provider.ts";

interface RecordedCall {
	method: "stream" | "streamSimple";
	options: unknown;
}

/**
 * Minimal provider that records delegated calls. Only `id`/`stream`/`streamSimple`
 * matter for the wrap, so the rest of the Provider shape is stubbed out.
 */
function recordingAnthropicProvider(): Provider & {
	calls: RecordedCall[];
} {
	const calls: RecordedCall[] = [];
	return {
		id: "anthropic",
		name: "anthropic",
		stream(_model, _context, options) {
			calls.push({ method: "stream", options });
			return {} as never;
		},
		streamSimple(_model, _context, options) {
			calls.push({ method: "streamSimple", options });
			return {} as never;
		},
		calls,
	} as unknown as Provider & { calls: RecordedCall[] };
}

describe("wrapAnthropicProvider", () => {
	it("rejects a non-anthropic provider", () => {
		expect(() =>
			wrapAnthropicProvider({ id: "openai" } as Provider),
		).toThrow(/cannot wrap/);
	});

	it("passes API-key requests through to the built-in provider unchanged", () => {
		const fake = recordingAnthropicProvider();
		const wrapped = wrapAnthropicProvider(fake);
		const opts = { apiKey: "sk-ant-api03-realapikey" };

		wrapped.stream({} as never, {} as never, opts as never);

		expect(fake.calls).toHaveLength(1);
		// Same object reference: the wrap must not merge anything for API keys.
		expect(fake.calls[0].options).toBe(opts);
	});

	it("passes through when no options are given", () => {
		const fake = recordingAnthropicProvider();
		const wrapped = wrapAnthropicProvider(fake);

		wrapped.stream({} as never, {} as never, undefined);

		expect(fake.calls).toHaveLength(1);
		expect(fake.calls[0].options).toBeUndefined();
	});

	it("merges Claude Code headers, wraps fetch, and chains onPayload for OAuth tokens", () => {
		const fake = recordingAnthropicProvider();
		const wrapped = wrapAnthropicProvider(fake);

		wrapped.stream({} as never, {} as never, {
			apiKey: "sk-ant-oat-xyz",
		} as never);

		expect(fake.calls).toHaveLength(1);
		const merged = fake.calls[0].options as Record<string, unknown>;
		const headers = merged.headers as Record<string, string>;
		expect(headers["user-agent"]).toMatch(/^claude-cli\/.*\(external, /);
		expect(headers["x-app"]).toBe("cli");
		expect(typeof merged.fetch).toBe("function");
		expect(typeof merged.onPayload).toBe("function");
	});

	it("runs an existing onPayload before injecting the billing header", async () => {
		const fake = recordingAnthropicProvider();
		const wrapped = wrapAnthropicProvider(fake);

		const seen: unknown[] = [];
		const priorTransform = (payload: unknown) => {
			seen.push(payload);
			return { ...(payload as object), priorRan: true };
		};

		wrapped.stream({} as never, {} as never, {
			apiKey: "sk-ant-oat-xyz",
			onPayload: priorTransform,
		} as never);
		const merged = fake.calls[0].options as {
			onPayload: (p: unknown, m: unknown) => Promise<unknown>;
		};

		const base = {
			model: "claude-haiku-4-5",
			max_tokens: 64,
			stream: true,
			system: [
				{
					type: "text",
					text: "You are Claude Code, Anthropic's official CLI for Claude.",
				},
			],
			messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
		};
		const result = (await merged.onPayload(base, {})) as {
			priorRan: boolean;
			system: { text: string }[];
		};

		expect(seen).toHaveLength(1); // the prior transform ran on the original payload
		expect(result.priorRan).toBe(true); // its output was preserved
		expect(result.system[0].text).toContain("x-anthropic-billing-header"); // billing injected after
	});

	it("streamSimple also merges options for OAuth tokens", () => {
		const fake = recordingAnthropicProvider();
		const wrapped = wrapAnthropicProvider(fake);

		wrapped.streamSimple({} as never, {} as never, {
			apiKey: "sk-ant-oat-xyz",
		} as never);

		expect(fake.calls).toHaveLength(1);
		expect(fake.calls[0].method).toBe("streamSimple");
		const merged = fake.calls[0].options as Record<string, unknown>;
		expect((merged.headers as Record<string, string>)["x-app"]).toBe("cli");
		expect(typeof merged.onPayload).toBe("function");
	});
});
