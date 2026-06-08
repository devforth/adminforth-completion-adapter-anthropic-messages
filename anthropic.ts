import type {
  CompletionStreamEvent,
  CompletionTool,
} from "adminforth";
import Anthropic from "@anthropic-ai/sdk";
import { jsonSchemaOutputFormat } from "@anthropic-ai/sdk/helpers/json-schema";
import type { AdapterOptions } from "./types.js";

export type StreamChunkCallback = (
  chunk: string,
  event?: CompletionStreamEvent,
) => void | Promise<void>;

export type ReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh";

export type CompletionRequestInput = {
  content: string;
  maxTokens?: number;
  outputSchema?: any;
  reasoningEffort?: ReasoningEffort;
  tools?: CompletionTool[];
  onChunk?: StreamChunkCallback;
};

type JsonSchemaObject = {
  type: "object";
  [key: string]: unknown;
};

type AnthropicMessageLike = {
  content?: Array<any>;
  stop_reason?: string | null;
  parsed_output?: unknown;
  usage?: AnthropicUsage;
};

type AnthropicToolUseBlock = {
  type: "tool_use";
  id: string;
  name: string;
  input: unknown;
};

type AnthropicUsage = {
  input_tokens?: number;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  output_tokens?: number;
};

export type UsedTokens = {
  input_uncached: number;
  input_cached: number;
  output: number;
};

export type CompletionResult = {
  content?: string;
  finishReason?: string;
  error?: string;
  used_tokens?: UsedTokens;
};

export function getApiKey(options: AdapterOptions): string | undefined {
  return options.anthropicApiKey || options.antropicApiKey;
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return String(error || "Unknown error");
}

function normalizeOutputSchema(outputSchema: any): JsonSchemaObject | undefined {
  if (!outputSchema || typeof outputSchema !== "object") return undefined;
  const candidate = outputSchema.schema || outputSchema.json_schema || outputSchema;
  if (!candidate || typeof candidate !== "object") {
    return undefined;
  }
  if (candidate.type !== "object") {
    return undefined;
  }
  return candidate as JsonSchemaObject;
}

function extractOutputText(data: AnthropicMessageLike): string {
  let text = "";

  for (const block of data.content ?? []) {
    if (block?.type === "text" && typeof block.text === "string") {
      text += block.text;
    }
  }

  return text;
}

function extractReasoning(data: AnthropicMessageLike): string | undefined {
  let reasoning = "";

  for (const block of data.content ?? []) {
    if (block?.type === "thinking" && typeof block.thinking === "string") {
      reasoning += block.thinking;
    }
  }

  return reasoning || undefined;
}

function extractUsedTokens(data: AnthropicMessageLike): UsedTokens | undefined {
  const usage = data.usage;
  if (!usage) return undefined;

  const inputTokens = usage.input_tokens ?? 0;
  const cacheWriteTokens = usage.cache_creation_input_tokens ?? 0;
  const cacheReadTokens = usage.cache_read_input_tokens ?? 0;

  return {
    input_uncached: inputTokens + cacheWriteTokens,
    input_cached: cacheReadTokens,
    output: usage.output_tokens ?? 0,
  };
}

function extractToolUse(data: AnthropicMessageLike): AnthropicToolUseBlock | undefined {
  for (const block of data.content ?? []) {
    if (block?.type === "tool_use") {
      return block as AnthropicToolUseBlock;
    }
  }

  return undefined;
}

function stringifyToolResult(toolResult: unknown): string {
  if (typeof toolResult === "string") return toolResult;
  if (typeof toolResult === "undefined") return "";
  return JSON.stringify(toolResult);
}

async function executeToolCall(
  toolCall: AnthropicToolUseBlock,
  tools?: CompletionTool[],
): Promise<string> {
  const tool = tools?.find((candidate) => candidate.name === toolCall.name);
  if (!tool) {
    throw new Error(`Tool "${toolCall.name}" not found`);
  }

  const toolResult = await tool.handler(toolCall.input || {});
  return stringifyToolResult(toolResult);
}

function mapTools(tools?: CompletionTool[]) {
  if (!tools?.length) return undefined;
  return tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: tool.input_schema,
  }));
}

export function mapReasoningToThinking(
  reasoningEffort: ReasoningEffort,
  maxTokens: number,
): { type: "enabled"; budget_tokens: number } | undefined {
  if (reasoningEffort === "none") return undefined;

  const availableBudget = maxTokens - 128;
  if (availableBudget < 1024) return undefined;

  const ratioByEffort: Record<Exclude<ReasoningEffort, "none">, number> = {
    minimal: 0.25,
    low: 0.35,
    medium: 0.5,
    high: 0.7,
    xhigh: 0.85,
  };
  const targetBudget = Math.floor(maxTokens * ratioByEffort[reasoningEffort]);

  return {
    type: "enabled",
    budget_tokens: Math.max(1024, Math.min(availableBudget, targetBudget)),
  };
}

export class AnthropicMessagesService {
  private client?: Anthropic;

  constructor(private options: AdapterOptions) {}

  validate() {
    if (!getApiKey(this.options)) {
      throw new Error("anthropicApiKey is required");
    }
  }

  async measureTokensCount(content: string): Promise<number> {
    const response = await this.getClient().messages.countTokens({
      model: this.options.model || "claude-sonnet-4-5-20250929",
      messages: [{ role: "user", content }],
    });

    return response.input_tokens;
  }

  async complete(request: CompletionRequestInput): Promise<CompletionResult> {
    const {
      content,
      maxTokens: requestMaxTokens = 50,
      outputSchema: requestOutputSchema,
      reasoningEffort: requestReasoningEffort = "low",
      tools,
      onChunk: streamChunkCallback,
    } = request;
    const model = this.options.model || "claude-sonnet-4-5-20250929";
    const isStreaming = typeof streamChunkCallback === "function";
    const normalizedSchema = normalizeOutputSchema(requestOutputSchema);
    const extra = {
      ...(this.options.extraRequestBodyParameters || {}),
    } as Record<string, unknown>;
    const thinking = extra.thinking || mapReasoningToThinking(
      requestReasoningEffort,
      requestMaxTokens,
    );

    const body = {
      model,
      max_tokens: requestMaxTokens,
      messages: [{ role: "user", content }],
      tools: mapTools(tools),
      ...(thinking ? { thinking } : {}),
      ...extra,
    } as Record<string, unknown>;

    try {
      if (requestOutputSchema && !normalizedSchema) {
        return {
          error:
            "Anthropic structured output requires a top-level JSON schema object with type: \"object\"",
        };
      }

      if (normalizedSchema) {
        return this.completeWithStructuredOutput(
          body,
          normalizedSchema,
          tools,
          isStreaming,
          streamChunkCallback,
        );
      }

      if (!isStreaming) {
        return this.completeNonStreaming(body, tools);
      }

      return this.completeStreaming(body, tools, streamChunkCallback);
    } catch (error) {
      return {
        error: getErrorMessage(error),
      };
    }
  }

  private getClient(): Anthropic {
    if (!this.client) {
      const apiKey = getApiKey(this.options);
      if (!apiKey) {
        throw new Error("anthropicApiKey is required");
      }
      this.client = new Anthropic({ apiKey });
    }

    return this.client;
  }

  private async completeWithStructuredOutput(
    body: Record<string, unknown>,
    normalizedSchema: JsonSchemaObject,
    tools: CompletionTool[] | undefined,
    isStreaming: boolean,
    streamChunkCallback: StreamChunkCallback | undefined,
  ): Promise<CompletionResult> {
    const parsedMessage = (await this.getClient().messages.parse({
      ...(body as any),
      output_config: {
        format: jsonSchemaOutputFormat(normalizedSchema),
      },
    } as any)) as AnthropicMessageLike;
    const parsedOutput =
      typeof parsedMessage.parsed_output === "undefined"
        ? extractOutputText(parsedMessage)
        : JSON.stringify(parsedMessage.parsed_output);
    const parsedReasoning = extractReasoning(parsedMessage);
    const usedTokens = extractUsedTokens(parsedMessage);

    if (parsedReasoning && isStreaming) {
      await streamChunkCallback?.(parsedReasoning, {
        type: "reasoning",
        delta: parsedReasoning,
        text: parsedReasoning,
      });
    }
    if (parsedOutput && isStreaming) {
      await streamChunkCallback?.(parsedOutput, {
        type: "output",
        delta: parsedOutput,
        text: parsedOutput,
      });
    }

    const toolUse = extractToolUse(parsedMessage);
    if (toolUse) {
      try {
        const toolResult = await executeToolCall(toolUse, tools);
        if (toolResult && isStreaming) {
          await streamChunkCallback?.(toolResult, {
            type: "output",
            delta: toolResult,
            text: toolResult,
          });
        }

        return {
          content: toolResult,
          finishReason: "tool_call",
          used_tokens: usedTokens,
        };
      } catch (error) {
        return {
          error: getErrorMessage(error),
          finishReason: "tool_call",
          used_tokens: usedTokens,
        };
      }
    }

    return {
      content: parsedOutput || undefined,
      finishReason: parsedMessage.stop_reason || undefined,
      used_tokens: usedTokens,
    };
  }

  private async completeNonStreaming(
    body: Record<string, unknown>,
    tools?: CompletionTool[],
  ): Promise<CompletionResult> {
    const message = (await this.getClient().messages.create(
      body as any,
    )) as AnthropicMessageLike;
    const usedTokens = extractUsedTokens(message);

    const toolUse = extractToolUse(message);
    if (toolUse) {
      try {
        const toolResult = await executeToolCall(toolUse, tools);
        return {
          content: toolResult,
          finishReason: "tool_call",
          used_tokens: usedTokens,
        };
      } catch (error) {
        return {
          error: getErrorMessage(error),
          finishReason: "tool_call",
          used_tokens: usedTokens,
        };
      }
    }

    return {
      content: extractOutputText(message) || undefined,
      finishReason: message.stop_reason || undefined,
      used_tokens: usedTokens,
    };
  }

  private async completeStreaming(
    body: Record<string, unknown>,
    tools: CompletionTool[] | undefined,
    streamChunkCallback: StreamChunkCallback | undefined,
  ): Promise<CompletionResult> {
    const stream = this.getClient().messages.stream(body as any);
    let fullContent = "";
    let fullReasoning = "";

    for await (const event of stream as any) {
      if (event?.type !== "content_block_delta") continue;

      if (event.delta?.type === "text_delta" && typeof event.delta.text === "string") {
        fullContent += event.delta.text;
        await streamChunkCallback?.(event.delta.text, {
          type: "output",
          delta: event.delta.text,
          text: fullContent,
        });
        continue;
      }

      if (
        event.delta?.type === "thinking_delta" &&
        typeof event.delta.thinking === "string"
      ) {
        fullReasoning += event.delta.thinking;
        await streamChunkCallback?.(event.delta.thinking, {
          type: "reasoning",
          delta: event.delta.thinking,
          text: fullReasoning,
        });
      }
    }

    const message = (await stream.finalMessage()) as AnthropicMessageLike;
    const finalContent = extractOutputText(message);
    const finalReasoning = extractReasoning(message) || "";
    const usedTokens = extractUsedTokens(message);

    if (finalReasoning && finalReasoning !== fullReasoning) {
      const delta = finalReasoning.startsWith(fullReasoning)
        ? finalReasoning.slice(fullReasoning.length)
        : finalReasoning;
      if (delta) {
        fullReasoning = finalReasoning;
        await streamChunkCallback?.(delta, {
          type: "reasoning",
          delta,
          text: finalReasoning,
        });
      }
    }

    if (finalContent && finalContent !== fullContent) {
      const delta = finalContent.startsWith(fullContent)
        ? finalContent.slice(fullContent.length)
        : finalContent;
      if (delta) {
        fullContent = finalContent;
        await streamChunkCallback?.(delta, {
          type: "output",
          delta,
          text: finalContent,
        });
      }
    }

    const toolUse = extractToolUse(message);
    if (toolUse) {
      try {
        const toolResult = await executeToolCall(toolUse, tools);
        if (toolResult) {
          const delta = toolResult.startsWith(fullContent)
            ? toolResult.slice(fullContent.length)
            : toolResult;
          if (delta) {
            await streamChunkCallback?.(delta, {
              type: "output",
              delta,
              text: toolResult,
            });
          }
        }

        return {
          content: toolResult,
          finishReason: "tool_call",
          used_tokens: usedTokens,
        };
      } catch (error) {
        return {
          error: getErrorMessage(error),
          content: fullContent || undefined,
          finishReason: "tool_call",
          used_tokens: usedTokens,
        };
      }
    }

    return {
      content: fullContent || undefined,
      finishReason: message.stop_reason || undefined,
      used_tokens: usedTokens,
    };
  }
}
