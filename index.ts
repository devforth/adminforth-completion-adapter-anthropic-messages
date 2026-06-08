import type { AdapterOptions } from "./types.js";
import type {
  CompletionAdapter,
  CompletionTool,
} from "adminforth";
import {
  AnthropicMessagesService,
  type CompletionRequestInput,
  type CompletionResult,
  type ReasoningEffort,
  type StreamChunkCallback,
} from "./anthropic.js";
import {
  createLangChainAgentSpec,
  type AgentModelPurpose,
} from "./langchain.js";

export type { AdapterOptions } from "./types.js";

class CompletionAdapterAnthropicMessages
  implements CompletionAdapter
{
  options: AdapterOptions;
  private anthropic: AnthropicMessagesService;

  constructor(options: AdapterOptions) {
    this.options = options;
    this.anthropic = new AnthropicMessagesService(options);
  }

  validate() {
    this.anthropic.validate();
  }

  measureTokensCount(content: string): Promise<number> {
    return this.anthropic.measureTokensCount(content);
  }

  getLangChainAgentSpec(params: {
    maxTokens: number;
    purpose: AgentModelPurpose;
  }) {
    return createLangChainAgentSpec({
      options: this.options,
      maxTokens: params.maxTokens,
      purpose: params.purpose,
    });
  }

  complete = async (
    requestOrContent: CompletionRequestInput | string,
    maxTokens = 50,
    outputSchema?: any,
    reasoningEffort: ReasoningEffort = "low",
    toolsOrOnChunk?: CompletionTool[] | StreamChunkCallback,
    onChunk?: StreamChunkCallback,
  ): Promise<CompletionResult> => {
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

    return this.anthropic.complete(request);
  };
}

export { CompletionAdapterAnthropicMessages };
export default CompletionAdapterAnthropicMessages;
