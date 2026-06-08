import { ChatAnthropic } from "@langchain/anthropic";
import type { AdapterOptions } from "./types.js";
import {
  getApiKey,
  mapReasoningToThinking,
  type ReasoningEffort,
} from "./anthropic.js";

export type AgentModelPurpose = "primary" | "summary";

type LangChainMessageLike = {
  content?: unknown;
  text?: string;
  type?: string;
  getType?: () => string;
  _getType?: () => string;
};

type LangChainModelCallRequest = {
  systemMessage: LangChainMessageLike & {
    concat: (content: string) => LangChainMessageLike;
  };
  messages: LangChainMessageLike[];
};

function getAgentReasoningEffort(
  purpose: AgentModelPurpose,
): ReasoningEffort {
  return purpose === "summary" ? "minimal" : "low";
}

function isSystemMessage(message: LangChainMessageLike): boolean {
  return (
    message._getType?.() === "system" ||
    message.getType?.() === "system" ||
    message.type === "system"
  );
}

function contentToText(content: unknown, text?: string): string {
  if (text) return text;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  return content
    .map((block) => {
      if (typeof block === "string") return block;
      if (block && typeof block === "object" && "text" in block) {
        return String(block.text ?? "");
      }
      return "";
    })
    .join("");
}

function normalizeAnthropicSystemMessages<T extends LangChainModelCallRequest>(
  request: T,
): T {
  const existingSystemText = contentToText(
    request.systemMessage.content,
    request.systemMessage.text,
  );
  const extraSystemText = request.messages
    .filter(isSystemMessage)
    .map((message) => contentToText(message.content, message.text))
    .filter((text) => text && text !== existingSystemText)
    .join("\n\n");

  if (!extraSystemText) {
    return {
      ...request,
      messages: request.messages.filter((message) => !isSystemMessage(message)),
    };
  }

  const systemMessage = request.systemMessage.concat(extraSystemText);

  return {
    ...request,
    systemMessage,
    messages: request.messages.filter((message) => !isSystemMessage(message)),
  };
}

function createAnthropicSystemMessageMiddleware() {
  return {
    name: "AnthropicSystemMessageMiddleware",
    async wrapModelCall(
      request: LangChainModelCallRequest,
      handler: (request: LangChainModelCallRequest) => unknown,
    ) {
      return handler(normalizeAnthropicSystemMessages(request));
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
    middleware: [createAnthropicSystemMessageMiddleware()],
  };
}
