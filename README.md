# @adminforth/completion-adapter-antropic-messages

<img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /> <img src="https://woodpecker.devforth.io/api/badges/3848/status.svg" alt="Build Status" /> <a href="https://www.npmjs.com/package/@adminforth/completion-adapter-antropic-messages"><img src="https://img.shields.io/npm/dm/@adminforth/completion-adapter-antropic-messages" alt="npm downloads" /></a> <a href="https://www.npmjs.com/package/@adminforth/completion-adapter-antropic-messages"><img src="https://img.shields.io/npm/v/@adminforth/completion-adapter-antropic-messages" alt="npm version" /></a>

[![Ask AI](https://tluma.ai/badge)](https://tluma.ai/ask-ai/devforth/adminforth)

AdminForth completion adapter for the Anthropic Messages API.

## Installation

```bash
pnpm i @adminforth/completion-adapter-antropic-messages
```

## Usage

```ts
import CompletionAdapterAntropicMessages from "@adminforth/completion-adapter-antropic-messages";

const adapter = new CompletionAdapterAntropicMessages({
	anthropicApiKey: process.env.ANTHROPIC_API_KEY as string,
	model: "claude-sonnet-4-5-20250929",
	extraRequestBodyParameters: {
		temperature: 0.7,
	},
});
```

The adapter supports:

- regular text completion
- JSON Schema structured output via the Messages parse helper
- tool calls
- streaming output chunks
- extended thinking when the token budget allows it

## Related links

- [AdminForth website](https://adminforth.dev)
- [npm package](https://www.npmjs.com/package/@adminforth/completion-adapter-antropic-messages)
- [All completion adapters](https://adminforth.dev/docs/tutorial/Adapters/ai-completion-adapters/)
- [All AdminForth adapters](https://adminforth.dev/docs/tutorial/ListOfAdapters/)
- [Built by DevForth](https://devforth.io)
