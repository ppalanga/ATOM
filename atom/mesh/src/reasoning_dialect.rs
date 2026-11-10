//! Reasoning formats the `reasoning-parser` crate does not speak, and the one
//! place a parser's prompt-side opener is read from.
//!
//! The crate's `kimi` parser reads Kimi-1.5's `◁think▷`, and its `minimax`
//! parser MiniMax-M2's `<think>`. Kimi-K3 frames its reasoning as a channel
//! (`<|open|>think<|sep|>` ... `<|close|>think<|sep|>`) and MiniMax-M3 as
//! `<mm:think>` ... `</mm:think>`; read by the crate's parsers, K3's chain of
//! thought came back as `content` and M3's answer as `reasoning_content`.
//! Mirrors `atom/entrypoints/openai/reasoning_dialects.py`.

use crate::reasoning_parser::{
    BaseReasoningParser, ParseError, ParserConfig, ParserFactory, ParserResult, ReasoningParser,
};

pub const KIMI_K3: &str = "kimi_k3";
pub const MINIMAX_M3: &str = "minimax_m3";

const K3_THINK_START: &str = "<|open|>think<|sep|>";
const K3_THINK_END: &str = "<|close|>think<|sep|>";
const K3_RESPONSE_START: &str = "<|open|>response<|sep|>";
const K3_TOOLS_START: &str = "<|open|>tools<|sep|>";
const K3_REASONING_ENDS: [&str; 3] = [K3_THINK_END, K3_RESPONSE_START, K3_TOOLS_START];
/// Paired wrappers around K3's answer. The bare `<|sep|>` and the unpaired
/// closers stay: they also occur inside a tool region, which the tool parser
/// reads after this stage (see `kimi_k3_tokens.UNPAIRED_FRAMING`).
const K3_CONTENT_FRAMING: [&str; 5] = [
    K3_RESPONSE_START,
    "<|close|>response<|sep|>",
    "<|open|>message<|sep|>",
    "<|close|>message<|sep|>",
    "<|end_of_msg|>",
];

const MM_THINK_START: &str = "<mm:think>";
const MM_THINK_END: &str = "</mm:think>";

/// The crate's factory plus the parsers above, under their own names.
pub fn parser_factory() -> ParserFactory {
    let factory = ParserFactory::new();
    let registry = factory.registry();
    registry.register_parser(KIMI_K3, || Box::new(KimiK3Parser::default()));
    registry.register_parser(MINIMAX_M3, || {
        Box::new(
            BaseReasoningParser::new(ParserConfig {
                think_start_token: MM_THINK_START.to_string(),
                think_end_token: MM_THINK_END.to_string(),
                ..ParserConfig::default()
            })
            .with_model_type(MINIMAX_M3.to_string()),
        )
    });
    factory
}

/// The parser a model name selects ahead of the crate's patterns, which are
/// first-match and would read `Kimi-K3` as `kimi` and `MiniMax-M3` as `minimax`.
pub fn parser_for_model(model: &str) -> Option<&'static str> {
    let model = model.to_lowercase().replace('_', "-");
    if model.contains("kimi-k3") {
        Some(KIMI_K3)
    } else if model.contains("minimax-m3") {
        Some(MINIMAX_M3)
    } else {
        None
    }
}

/// The opener `parser` reads, when `prompt` ends with it: the template has
/// opened the reasoning block, so the output begins inside it and carries
/// only the closer. Fed to the parser first, it starts there too.
pub fn prompt_opened_reasoning(
    parser: &dyn ReasoningParser,
    prompt: Option<&str>,
) -> Option<&'static str> {
    let opener = match parser.model_type() {
        KIMI_K3 => K3_THINK_START,
        MINIMAX_M3 => MM_THINK_START,
        "kimi" => "◁think▷",
        "passthrough" => return None,
        _ => "<think>",
    };
    prompt
        .is_some_and(|p| p.trim_end().ends_with(opener))
        .then_some(opener)
}

/// Kimi-K3's channel format. Reasoning ends at its own closer, or at the
/// opening of the answer or tools channel, which K3 often writes in its place;
/// those two stay in the text for the stages after this one.
#[derive(Default)]
pub struct KimiK3Parser {
    in_reasoning: bool,
    buffer: String,
}

impl KimiK3Parser {
    fn feed(&mut self, text: &str) -> ParserResult {
        self.buffer.push_str(text);
        let (mut reasoning, mut normal) = (String::new(), String::new());
        loop {
            let markers: &[&str] = if self.in_reasoning {
                &K3_REASONING_ENDS
            } else {
                &[K3_THINK_START]
            };
            let out = if self.in_reasoning {
                &mut reasoning
            } else {
                &mut normal
            };
            let found = markers
                .iter()
                .filter_map(|m| self.buffer.find(m).map(|at| (at, *m)))
                .min_by_key(|&(at, _)| at);
            let Some((at, marker)) = found else {
                let keep = held_suffix(&self.buffer, self.in_reasoning);
                out.extend(self.buffer.drain(..self.buffer.len() - keep));
                break;
            };
            out.extend(self.buffer.drain(..at));
            if marker == K3_THINK_START || marker == K3_THINK_END {
                self.buffer.drain(..marker.len());
            }
            self.in_reasoning = !self.in_reasoning;
        }
        for framing in K3_CONTENT_FRAMING {
            normal = normal.replace(framing, "");
        }
        ParserResult::new(normal, reasoning)
    }
}

/// How much of `buffer`'s tail could be the start of a marker still to come.
/// Every marker begins with a `<|...|>` token, which a delta carries whole.
fn held_suffix(buffer: &str, in_reasoning: bool) -> usize {
    let Some(at) = buffer.rfind("<|") else {
        return 0;
    };
    let tail = &buffer[at..];
    let is_partial = |m: &&str| m.len() > tail.len() && m.starts_with(tail);
    let partial = if in_reasoning {
        K3_REASONING_ENDS.iter().any(is_partial)
    } else {
        // The framing is held too, or a wrapper split over two deltas
        // would escape the strip.
        [K3_THINK_START]
            .iter()
            .chain(&K3_CONTENT_FRAMING)
            .any(is_partial)
    };
    if partial {
        tail.len()
    } else {
        0
    }
}

impl ReasoningParser for KimiK3Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        self.reset();
        let mut result = self.feed(text);
        let rest = std::mem::take(&mut self.buffer);
        if self.in_reasoning {
            result.reasoning_text.push_str(&rest);
        } else {
            result.normal_text.push_str(&rest);
        }
        self.reset();
        Ok(result)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        Ok(self.feed(text))
    }

    fn reset(&mut self) {
        self.in_reasoning = false;
        self.buffer.clear();
    }

    fn model_type(&self) -> &str {
        KIMI_K3
    }

    fn is_in_reasoning(&self) -> bool {
        self.in_reasoning
    }
}
