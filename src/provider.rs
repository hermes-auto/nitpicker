use crate::config::{AggregatorConfig, ProviderType, ReviewerConfig};
use crate::gemini_proxy::GeminiProxyClient;
use crate::llm::{LLMClient, LLMClientDyn, LLMProvider, WithRetryExt};
use eyre::Result;
use std::sync::Arc;

pub fn needs_gemini_oauth(provider: &ProviderType, auth: Option<&str>) -> bool {
    provider.is_gemini() && auth == Some("oauth")
}

pub fn reviewer_needs_gemini_oauth(reviewer: &ReviewerConfig) -> bool {
    needs_gemini_oauth(&reviewer.provider, reviewer.auth.as_deref())
}

pub fn aggregator_needs_gemini_oauth(agg: &AggregatorConfig) -> bool {
    needs_gemini_oauth(&agg.provider, agg.auth.as_deref())
}

pub fn provider_from_config(
    provider: &ProviderType,
    base_url: Option<&str>,
    api_key_env: Option<&str>,
) -> Result<LLMProvider> {
    match provider {
        ProviderType::Anthropic => Ok(LLMProvider::Anthropic),
        ProviderType::Gemini => Ok(LLMProvider::Gemini),
        ProviderType::AnthropicCompatible => Ok(LLMProvider::AnthropicCompatible {
            base_url: require_field(base_url, "base_url", "anthropic_compatible")?,
            api_key_env: require_field(api_key_env, "api_key_env", "anthropic_compatible")?,
        }),
        ProviderType::OpenAiCompatible => Ok(LLMProvider::OpenAICompatible {
            base_url: require_field(base_url, "base_url", "openai_compatible")?,
            api_key_env: require_field(api_key_env, "api_key_env", "openai_compatible")?,
        }),
    }
}

pub fn build_reviewer_client(
    reviewer: &ReviewerConfig,
    gemini_proxy: Option<&GeminiProxyClient>,
) -> Result<Arc<dyn LLMClientDyn>> {
    if reviewer_needs_gemini_oauth(reviewer) {
        let proxy_url = gemini_proxy
            .map(|p| p.base_url())
            .ok_or_else(|| eyre::eyre!("Gemini proxy required for OAuth but not available"))?;
        return crate::llm::create_gemini_client_with_proxy(&proxy_url);
    }

    Ok(provider_from_config(
        &reviewer.provider,
        reviewer.base_url.as_deref(),
        reviewer.api_key_env.as_deref(),
    )?
    .client_from_env()?
    .with_retry()
    .into_arc())
}

pub fn build_aggregator_client(
    agg: &AggregatorConfig,
    gemini_proxy: Option<&GeminiProxyClient>,
) -> Result<Arc<dyn LLMClientDyn>> {
    if aggregator_needs_gemini_oauth(agg) {
        let proxy_url = gemini_proxy
            .map(|p| p.base_url())
            .ok_or_else(|| eyre::eyre!("Gemini proxy required for OAuth but not available"))?;
        return crate::llm::create_gemini_client_with_proxy(&proxy_url);
    }

    Ok(provider_from_config(
        &agg.provider,
        agg.base_url.as_deref(),
        agg.api_key_env.as_deref(),
    )?
    .client_from_env()?
    .with_retry()
    .into_arc())
}

fn require_field(value: Option<&str>, field: &str, provider: &str) -> Result<String> {
    value
        .map(str::to_string)
        .ok_or_else(|| eyre::eyre!("{provider} provider requires `{field}`"))
}
