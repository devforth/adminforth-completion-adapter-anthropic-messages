import type { AdapterOptions } from "./types.js";
import type {
  CompletionAdapter,
  CompletionStreamEvent,
  CompletionTool,
} from "adminforth";
import { ChatAnthropic } from "@langchain/anthropic";
import Anthropic from "@anthropic-ai/sdk";
import { jsonSchemaOutputFormat } from "@anthropic-ai/sdk/helpers/json-schema";

export type { AdapterOptions } from "./types.js";

type StreamChunkCallback = (
  chunk: string,
  event?: CompletionStreamEvent,
) => void | Promise<void>;

type ReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh";

type AgentModelPurpose = "primary" | "summary";

type CompletionRequestInput = {
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
};

type AnthropicToolUseBlock = {
  type: "tool_use";
  id: string;
  name: string;
  input: unknown;
};

function getApiKey(options: AdapterOptions): string | undefined {
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

function mapReasoningToThinking(
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

function getAgentReasoningEffort(
  purpose: AgentModelPurpose,
): ReasoningEffort {
  return purpose === "summary" ? "minimal" : "low";
}

export default class CompletionAdapterAntropicMessages
  implements CompletionAdapter
{
  options: AdapterOptions;
  private client?: Anthropic;

  constructor(options: AdapterOptions) {
    this.options = options;
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

  getLangChainAgentSpec(params: {
    maxTokens: number;
    purpose: AgentModelPurpose;
  }) {
    const extraRequestBodyParameters = {
      ...(this.options.extraRequestBodyParameters || {}),
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
        model: this.options.model || "claude-sonnet-4-5-20250929",
        apiKey: getApiKey(this.options),
        maxTokens: params.maxTokens,
        ...(thinking ? { thinking } : {}),
        invocationKwargs: extraRequestBodyParameters,
      } as any),
    };
  }

  complete = async (
    requestOrContent: CompletionRequestInput | string,
    maxTokens = 50,
    outputSchema?: any,
    reasoningEffort: ReasoningEffort = "low",
    toolsOrOnChunk?: CompletionTool[] | StreamChunkCallback,
    onChunk?: StreamChunkCallback,
  ): Promise<{
    content?: string;
    finishReason?: string;
    error?: string;
  }> => {
    const request =
      typeof requestOrContent === "string"
        ? {
            content: requestOrContent,
            maxTokens,
            outputSchema,
            reasoningEffort,
            tools: Array.isArray(toolsOrOnChunk) ? toolsOrOnChunk : undefined,
            onChunk:
              typeof toolsOrOnChunk === "function"
                ? toolsOrOnChunk
                : onChunk,
          }
        : requestOrContent;
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
            };
          } catch (error) {
            return {
              error: getErrorMessage(error),
              finishReason: "tool_call",
            };
          }
        }

        return {
          content: parsedOutput || undefined,
          finishReason: parsedMessage.stop_reason || undefined,
        };
      }

      if (!isStreaming) {
        const message = (await this.getClient().messages.create(
          body as any,
        )) as AnthropicMessageLike;

        const toolUse = extractToolUse(message);
        if (toolUse) {
          try {
            const toolResult = await executeToolCall(toolUse, tools);
            return {
              content: toolResult,
              finishReason: "tool_call",
            };
          } catch (error) {
            return {
              error: getErrorMessage(error),
              finishReason: "tool_call",
            };
          }
        }

        return {
          content: extractOutputText(message) || undefined,
          finishReason: message.stop_reason || undefined,
        };
      }

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
          };
        } catch (error) {
          return {
            error: getErrorMessage(error),
            content: fullContent || undefined,
            finishReason: "tool_call",
          };
        }
      }

      return {
        content: fullContent || undefined,
        finishReason: message.stop_reason || undefined,
      };
    } catch (error) {
      return {
        error: getErrorMessage(error),
      };
    }
  };
}