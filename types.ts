export interface AdapterOptions {
  /**
   * Anthropic API key. Go to https://console.anthropic.com/settings/keys and create a new key.
   * Set anthropicApiKey: process.env.ANTHROPIC_API_KEY to access it.
   */
  anthropicApiKey?: string;

  /**
   * Deprecated alias kept for compatibility with the package name spelling.
   */
  antropicApiKey?: string;

  /**
   * Model name. See https://docs.anthropic.com/en/docs/about-claude/models for available models.
   * Default is `claude-sonnet-4-5-20250929`.
   */
  model?: string;

  /**
   * Additional request body parameters to include in the API request.
   */
  extraRequestBodyParameters?: Record<string, unknown>;
}