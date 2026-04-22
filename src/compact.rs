use crate::llm::{Completion, LLMClientDyn, TokenUsage};
use eyre::Result;
use rig::completion::Message;
use std::sync::Arc;

const COMPACTION_MAX_OUTPUT_TOKENS: u64 = 8_192;

pub async fn compact_history(
    client: Arc<dyn LLMClientDyn>,
    model: &str,
    system_prompt: &str,
    history: &mut Vec<Message>,
    prompt: &mut Message,
    turn: usize,
    usage: TokenUsage,
) -> Result<Option<TokenUsage>> {
    if history.is_empty() {
        return Ok(None);
    }

    let (thread, residual_messages) = split_thread(history);
    if thread.is_empty() {
        return Ok(None);
    }

    let compaction_prompt = build_compaction_prompt(turn, usage);
    let response = client
        .completion(Completion {
            model: model.to_string(),
            prompt: Message::user(compaction_prompt),
            preamble: Some(system_prompt.to_string()),
            history: thread,
            tools: Vec::new(),
            temperature: Some(0.2),
            max_tokens: Some(COMPACTION_MAX_OUTPUT_TOKENS),
            additional_params: None,
        })
        .await?;

    let text = response.text();
    let summary = extract_tag(&text, "summary")
        .ok_or_else(|| eyre::eyre!("compaction output missing <summary> block"))?;

    let mut compacted = vec![Message::user(format!(
        "compacted conversation summary before turn {turn}:\n\n{summary}"
    ))];
    compacted.extend(residual_messages);
    *history = compacted;
    *prompt = history.last().cloned().unwrap_or_else(|| prompt.clone());
    Ok(Some(response.usage))
}

fn split_thread(history: &[Message]) -> (Vec<Message>, Vec<Message>) {
    match history.last() {
        Some(Message::User { .. }) => (
            history[..history.len().saturating_sub(1)].to_vec(),
            vec![history.last().cloned().expect("last message exists")],
        ),
        _ => (history.to_vec(), Vec::new()),
    }
}

fn build_compaction_prompt(turn: usize, usage: TokenUsage) -> String {
    format!(
        "summarize the conversation so far to fit within a token budget. focus on durable user intent, constraints, decisions, discovered facts, tool outcomes, unresolved issues, and current execution status. omit verbose tool output and code unless it is essential. the result should be much shorter than the original conversation.\n\nReturn exactly one tagged block and nothing else:\n<summary>...</summary>\n\nThis compaction is happening before turn {turn}. Usage since the previous compaction: {} input, {} output, {} total tokens.",
        usage.input_tokens, usage.output_tokens, usage.total_tokens
    )
}

fn extract_tag(text: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let start = text.find(&start_tag)? + start_tag.len();
    let end = text[start..].find(&end_tag)? + start;
    let content = text[start..end].trim();
    if content.is_empty() {
        None
    } else {
        Some(content.to_string())
    }
}
