//! Reasoning formats the `reasoning-parser` crate does not speak, and the one
//! place a parser's prompt-side opener is read from.
//!
//! The crate's `kimi` parser reads Kimi-1.5's `◁think▷`, and its `minimax`
//! parser MiniMax-M2's `<think>`. Kimi-K3 frames its reasoning as a channel
//! (`<|open|>think<|sep|>` ... `<|close|>think<|sep|>`) and MiniMax-M3 as
//! `<mm:think>` ... `</mm:think>`; read by the crate's parsers, K3's chain of
//! thought came back as `content` and M3's answer as `reasoning_content`.
//! Mirrors `atom/entrypoints/openai/reasoning_dialects.py`.

use crate::reasoning_parser::{ParseError, ParserFactory, ParserResult, ReasoningParser};

pub const KIMI_K3: &str = "kimi_k3";
pub const MINIMAX_M3: &str = "minimax_m3";

const K3_THINK_START: &str = "<|open|>think<|sep|>";
const K3_THINK_END: &str = "<|close|>think<|sep|>";
const K3_RESPONSE_START: &str = "<|open|>response<|sep|>";
const K3_RESPONSE_END: &str = "<|close|>response<|sep|>";
const K3_TOOLS_START: &str = "<|open|>tools<|sep|>";
const K3_MESSAGE_START: &str = "<|open|>message<|sep|>";
const K3_MESSAGE_END: &str = "<|close|>message<|sep|>";
const K3_END_OF_MSG: &str = "<|end_of_msg|>";
/// Paired wrappers around K3's answer. The bare `<|sep|>` and the unpaired
/// closers stay: they also occur inside a tool region, which the tool parser
/// reads after this stage (see `kimi_k3_tokens.UNPAIRED_FRAMING`).
const K3_CONTENT_FRAMING: [&str; 5] = [
    K3_RESPONSE_START,
    K3_RESPONSE_END,
    K3_MESSAGE_START,
    K3_MESSAGE_END,
    K3_END_OF_MSG,
];

const MM_THINK_START: &str = "<mm:think>";
const MM_THINK_END: &str = "</mm:think>";

/// The crate's factory plus the parsers above, under their own names.
pub fn parser_factory() -> ParserFactory {
    let factory = ParserFactory::new();
    let registry = factory.registry();
    registry.register_parser(KIMI_K3, || Box::new(KimiK3Parser::default()));
    registry.register_parser(MINIMAX_M3, || Box::new(MiniMaxM3Parser::default()));
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

/// Where the output is: before the reasoning block, in it, or in the answer,
/// which is final. An opener in the answer (inside a tool's arguments, say)
/// is the model's own words, not reasoning reopening.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Phase {
    #[default]
    Before,
    Think,
    Answer,
}

/// MiniMax-M3's `<mm:think>` ... `</mm:think>`. Only the two markers are
/// removed, every other byte kept as written, whitespace included, on both
/// delivery paths; the crate's base parser trims what it splits out when
/// reading a whole output, so an answer's leading newline or a code block's
/// final one was lost there and not in the stream.
///
/// A bare closer at offset 0 of the output is M3's trained form for a turn
/// without thinking: removed, and the rest is the answer, a later opener
/// included. Anywhere else before the reasoning a closer is the model's own
/// words. Mirrors `ReasoningFilter` in `atom/entrypoints/openai/reasoning.py`.
#[derive(Default)]
pub struct MiniMaxM3Parser {
    state: Phase,
    buffer: String,
    /// Text has gone out before the reasoning opened, so a closer is no
    /// longer at offset 0.
    past_offset_zero: bool,
}

/// The markers M3 acts on in `phase`.
fn m3_markers(phase: Phase) -> &'static [&'static str] {
    match phase {
        Phase::Before => &[MM_THINK_START, MM_THINK_END],
        Phase::Think => &[MM_THINK_END],
        Phase::Answer => &[],
    }
}

impl MiniMaxM3Parser {
    fn feed(&mut self, text: &str) -> ParserResult {
        self.buffer.push_str(text);
        let mut out = ParserResult::default();
        loop {
            let (released, marker) = take_until_marker(&mut self.buffer, m3_markers(self.state));
            self.release(&released, &mut out);
            let Some(marker) = marker else { break };
            self.state = match (self.state, marker) {
                (Phase::Before, MM_THINK_START) => Phase::Think,
                (Phase::Before, _) if self.past_offset_zero => {
                    out.normal_text.push_str(marker); // the model's own words
                    Phase::Before
                }
                _ => Phase::Answer, // the reasoning's end, or declined to think
            };
        }
        out
    }

    fn release(&mut self, text: &str, out: &mut ParserResult) {
        if text.is_empty() {
            return;
        }
        if self.state == Phase::Think {
            out.reasoning_text.push_str(text);
        } else {
            self.past_offset_zero = true;
            out.normal_text.push_str(text);
        }
    }

    /// The end of the output: what is held never became a marker.
    fn finish(&mut self) -> ParserResult {
        let rest = std::mem::take(&mut self.buffer);
        if self.state == Phase::Think {
            ParserResult::reasoning(rest)
        } else {
            ParserResult::normal(rest)
        }
    }
}

impl ReasoningParser for MiniMaxM3Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        self.reset();
        let mut result = self.feed(text);
        let end = self.finish();
        result.normal_text.push_str(&end.normal_text);
        result.reasoning_text.push_str(&end.reasoning_text);
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
        self.state = Phase::Before;
        self.buffer.clear();
        self.past_offset_zero = false;
    }

    fn model_type(&self) -> &str {
        MINIMAX_M3
    }

    fn is_in_reasoning(&self) -> bool {
        self.state == Phase::Think
    }
}

/// Kimi-K3's channel format. Reasoning ends at its own closer, or at the
/// opening of the answer or tools channel, which K3 often writes in its place;
/// the tools opener stays in the text for the tool parser after this stage.
///
/// K3 sometimes skips the channel instead: it writes its reply straight into
/// the prompt-opened think channel and ends it with the answer's own closer,
/// `<|close|>response<|sep|>`, with no think closer or response opener at all.
/// If that closer was the channel's last words (whitespace and framing aside),
/// the channel did not exit to the answer's own channel, and the model stopped
/// rather than being cut off, the channel was the reply, and it goes out again
/// as content. Mirrors `ReasoningFilter` in `atom/entrypoints/openai/reasoning.py`.
#[derive(Default)]
pub struct KimiK3Parser {
    state: Phase,
    buffer: String,
    /// Text has gone out before the channel opened, so a think closer is no
    /// longer at offset 0, the one place it means "I did not think this turn".
    past_offset_zero: bool,
    /// The channel's text so far, in case it turns out to have been the reply.
    channel_text: String,
    /// The channel's last words, whitespace and framing aside, were the
    /// answer's closer.
    answer_closed: bool,
}

/// The markers K3 acts on in `phase`. Every phase removes the answer framing;
/// the think channel's own list has its ends and the answer's closer in it.
fn k3_markers(phase: Phase) -> &'static [&'static str] {
    match phase {
        Phase::Before => &K3_BEFORE_MARKERS,
        Phase::Think => &K3_CHANNEL_MARKERS,
        Phase::Answer => &K3_CONTENT_FRAMING,
    }
}

impl KimiK3Parser {
    fn feed(&mut self, text: &str) -> ParserResult {
        self.buffer.push_str(text);
        let mut out = ParserResult::default();
        loop {
            let (released, marker) = take_until_marker(&mut self.buffer, k3_markers(self.state));
            self.release(&released, &mut out);
            let Some(marker) = marker else { break };
            let normal = &mut out.normal_text;
            match (self.state, marker) {
                (Phase::Before, K3_THINK_START) => {
                    self.channel_text.clear();
                    self.answer_closed = false;
                    self.state = Phase::Think;
                }
                // The answer's channels have begun, wherever this falls: no
                // think channel comes after it. Forwarded for the tool parser.
                (Phase::Before, K3_TOOLS_START) => {
                    normal.push_str(marker);
                    self.state = Phase::Answer;
                }
                (Phase::Before, K3_THINK_END) => {
                    if self.past_offset_zero {
                        normal.push_str(marker); // the model's own words
                    } else {
                        self.state = Phase::Answer; // declined to think
                    }
                }
                (Phase::Think, K3_THINK_END | K3_RESPONSE_START | K3_TOOLS_START) => {
                    // After the think closer or the response opener the
                    // answer has its own channel, so this one was not the reply.
                    normal.push_str(&self.close_channel(marker == K3_TOOLS_START));
                    if marker == K3_TOOLS_START {
                        normal.push_str(marker);
                    }
                    self.state = Phase::Answer;
                }
                // Remembered for `close_channel` to decide on.
                (Phase::Think, K3_RESPONSE_END) => self.answer_closed = true,
                _ => {} // answer framing, removed
            }
        }
        out
    }

    /// Text read past in the current state. Words after the answer's closer
    /// mean it did not end the reply; whitespace does not.
    fn release(&mut self, text: &str, out: &mut ParserResult) {
        if text.is_empty() {
            return;
        }
        if self.state != Phase::Think {
            self.past_offset_zero = true;
            out.normal_text.push_str(text);
            return;
        }
        out.reasoning_text.push_str(text);
        self.channel_text.push_str(text);
        if self.answer_closed && !text.trim().is_empty() {
            self.answer_closed = false;
        }
    }

    /// The channel has ended; the text to send again as content if it was the
    /// reply: the answer's closer ended it, and how it ended (`may_be_reply`) allows it.
    fn close_channel(&mut self, may_be_reply: bool) -> String {
        let kept = std::mem::take(&mut self.channel_text);
        let closed = std::mem::replace(&mut self.answer_closed, false);
        if closed && may_be_reply {
            kept
        } else {
            String::new()
        }
    }

    /// The end of the output, which `stopped` says the model chose rather
    /// than being cut off: an output cut off at `max_tokens` right after a
    /// quoted closer looks like a finished reply, and only the finish reason
    /// tells the two apart. What is held never became a marker and goes out
    /// as it is; it is not words, so it does not reopen `answer_closed`.
    fn finish(&mut self, stopped: bool) -> ParserResult {
        let rest = std::mem::take(&mut self.buffer);
        if self.state != Phase::Think {
            return ParserResult::new(rest, String::new());
        }
        let reply = self.close_channel(stopped);
        ParserResult::new(reply, rest)
    }

    /// A complete output, read in one feed and finished.
    fn read_whole(&mut self, text: &str, stopped: bool) -> ParserResult {
        self.reset();
        let mut result = self.feed(text);
        let end = self.finish(stopped);
        result.normal_text.push_str(&end.normal_text);
        result.reasoning_text.push_str(&end.reasoning_text);
        self.reset();
        result
    }
}

/// What K3's reasoning channel watches for: its ends, and the framing removed
/// from it, the answer's closer among them.
const K3_CHANNEL_MARKERS: [&str; 7] = [
    K3_THINK_END,
    K3_RESPONSE_START,
    K3_TOOLS_START,
    K3_RESPONSE_END,
    K3_MESSAGE_START,
    K3_MESSAGE_END,
    K3_END_OF_MSG,
];

/// What the output watches for before the channel opens: the opener, the
/// think closer (at offset 0, a turn without thinking), the tools opener, and
/// the framing removed.
const K3_BEFORE_MARKERS: [&str; 8] = [
    K3_THINK_START,
    K3_THINK_END,
    K3_TOOLS_START,
    K3_RESPONSE_START,
    K3_RESPONSE_END,
    K3_MESSAGE_START,
    K3_MESSAGE_END,
    K3_END_OF_MSG,
];

/// Takes `buffer` up to the first of `markers` and the marker itself: the
/// text before it, and the marker. Without one, the text up to where a marker
/// may still be starting (`held_suffix`), and no marker.
fn take_until_marker(
    buffer: &mut String,
    markers: &[&'static str],
) -> (String, Option<&'static str>) {
    let found = markers
        .iter()
        .filter_map(|m| buffer.find(m).map(|at| (at, *m)))
        .min_by_key(|&(at, _)| at);
    match found {
        Some((at, marker)) => {
            let text = buffer.drain(..at).collect();
            buffer.drain(..marker.len());
            (text, Some(marker))
        }
        None => {
            let keep = held_suffix(buffer, markers);
            (buffer.drain(..buffer.len() - keep).collect(), None)
        }
    }
}

/// How much of `buffer`'s tail could be the start of one of `markers`, held
/// until the next delta says. Framing is held too, or a wrapper split over
/// two deltas would escape the strip.
fn held_suffix(buffer: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .flat_map(|m| (1..m.len()).filter(|&n| m.is_char_boundary(n) && buffer.ends_with(&m[..n])))
        .max()
        .unwrap_or(0)
}

impl ReasoningParser for KimiK3Parser {
    /// Without a finish reason the output is not known to have been stopped,
    /// so a skipped channel stays reasoning; [`read_whole`] takes one.
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        Ok(self.read_whole(text, false))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        Ok(self.feed(text))
    }

    fn reset(&mut self) {
        self.state = Phase::Before;
        self.buffer.clear();
        self.past_offset_zero = false;
        self.channel_text.clear();
        self.answer_closed = false;
    }

    fn model_type(&self) -> &str {
        KIMI_K3
    }

    fn is_in_reasoning(&self) -> bool {
        self.state == Phase::Think
    }
}

/// A complete output split into reasoning and content by `parser`, the model
/// having ended it itself when `stopped`. The crate's trait has no finish
/// reason to pass, and Kimi-K3 is the one format that needs it.
pub fn read_whole(
    parser: &mut dyn ReasoningParser,
    text: &str,
    stopped: bool,
) -> Result<ParserResult, ParseError> {
    if parser.model_type() == KIMI_K3 {
        Ok(KimiK3Parser::default().read_whole(text, stopped))
    } else {
        parser.detect_and_parse_reasoning(text)
    }
}

/// One streamed response's reasoning parser, with the end-of-output step the
/// crate's trait lacks: Kimi-K3's skipped channel is decided only there, and
/// MiniMax-M3 may still hold the first bytes of a closer.
pub enum StreamReasoning {
    KimiK3(KimiK3Parser),
    MiniMaxM3(MiniMaxM3Parser),
    Crate(Box<dyn ReasoningParser>),
}

impl StreamReasoning {
    pub fn new(parser: Box<dyn ReasoningParser>) -> Self {
        match parser.model_type() {
            KIMI_K3 => Self::KimiK3(KimiK3Parser::default()),
            MINIMAX_M3 => Self::MiniMaxM3(MiniMaxM3Parser::default()),
            _ => Self::Crate(parser),
        }
    }

    pub fn parser(&mut self) -> &mut dyn ReasoningParser {
        match self {
            Self::KimiK3(p) => p,
            Self::MiniMaxM3(p) => p,
            Self::Crate(p) => &mut **p,
        }
    }

    /// The end of the output, which `stopped` says the model chose.
    pub fn finish(&mut self, stopped: bool) -> ParserResult {
        match self {
            Self::KimiK3(p) => p.finish(stopped),
            Self::MiniMaxM3(p) => p.finish(),
            Self::Crate(_) => ParserResult::default(),
        }
    }
}
