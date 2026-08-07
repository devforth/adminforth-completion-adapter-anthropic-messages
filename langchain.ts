import { ChatAnthropic } from "@langchain/anthropic";
import type { AdapterOptions } from "./types.js";
import {
  getApiKey,
  mapReasoningToThinking,
  type ReasoningEffort,
} from "./anthropic.js";

export type AgentModelPurpose = "primary" | "summary";

type LangChainContentBlock = Record<string, unknown> & { type?: string };

type LangChainModelCallRequest = {
  systemMessage?: { content?: unknown };
  messages?: unknown[];
  modelSettings?: Record<string, unknown>;
};

function getAgentReasoningEffort(
  purpose: AgentModelPurpose,
): ReasoningEffort {
  return purpose === "summary" ? "minimal" : "low";
}

function markSystemPromptCacheBreakpoint<T extends LangChainModelCallRequest>(
  request: T,
): T {
  const content = request.systemMessage?.content;

  if (!Array.isArray(content) || content.length === 0) {
    return request;
  }

  const blocks = content as LangChainContentBlock[];

  if (blocks[0].type !== "text" || blocks[0].cache_control) {
    return request;
  }

  return {
    ...request,
    systemMessage: {
      ...request.systemMessage,
      content: [
        { ...blocks[0], cache_control: { type: "ephemeral" } },
        ...blocks.slice(1),
      ],
    },
  };
}

function markHistoryCacheBreakpoint<T extends LangChainModelCallRequest>(
  request: T,
): T {
  // Nothing to anchor a breakpoint to, and never override an explicit setting.
  if (!request.messages?.length || request.modelSettings?.cache_control) {
    return request;
  }

  return {
    ...request,
    modelSettings: {
      ...request.modelSettings,
      // No `ttl` — Anthropic defaults to 5 minutes, refreshed on every cache hit.
      cache_control: { type: "ephemeral" },
    },
  };
}

function createAnthropicPromptCachingMiddleware() {
  return {
    name: "AnthropicPromptCachingMiddleware",
    async wrapModelCall(
      request: LangChainModelCallRequest,
      handler: (request: LangChainModelCallRequest) => unknown,
    ) {
      return handler(
        markHistoryCacheBreakpoint(markSystemPromptCacheBreakpoint(request)),
      );
    },
  };
}

export function createLangChainAgentSpec(params: {
  options: AdapterOptions;
  maxTokens: number;
  purpose: AgentModelPurpose;
}) {
  const extraRequestBodyParameters = {
    ...(params.options.extraRequestBodyParameters || {}),
  } as Record<string, unknown> & {
    thinking?: { type: "enabled"; budget_tokens: number };
  };
  const thinking =
    extraRequestBodyParameters.thinking ||
    mapReasoningToThinking(
      getAgentReasoningEffort(params.purpose),
      params.maxTokens,
    );

  delete extraRequestBodyParameters.thinking;

  return {
    model: new ChatAnthropic({
      model: params.options.model || "claude-sonnet-4-5-20250929",
      apiKey: getApiKey(params.options),
      maxTokens: params.maxTokens,
      ...(thinking ? { thinking } : {}),
      invocationKwargs: extraRequestBodyParameters,
    } as any),
    middleware: [createAnthropicPromptCachingMiddleware()],
  };
}
