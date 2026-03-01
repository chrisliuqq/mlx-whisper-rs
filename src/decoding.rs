// Translated from mlx-whisper/decoding.py (Apple Inc.)

use anyhow::Result;
use mlx_rs::{
    Array,
    ops::{indexing::argmax_axis, logsumexp_axis, max_axis, softmax_axis},
    ops::indexing::IndexOp,
    transforms::eval,
};

use crate::audio::CHUNK_LENGTH;
use crate::tokenizer::Tokenizer;
use crate::whisper::Whisper;

// ── Options & Result ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct DecodingOptions {
    pub task: String,
    pub language: Option<String>,
    pub temperature: f32,
    /// Maximum number of tokens to sample; defaults to n_text_ctx / 2
    pub sample_len: Option<usize>,
    /// Suppress blank outputs at the first sampled position
    pub suppress_blank: bool,
    /// "-1" → use non_speech_tokens; comma-separated IDs; None → no suppression
    pub suppress_tokens: Option<String>,
    pub without_timestamps: bool,
    /// Maximum initial timestamp in seconds (default 1.0)
    pub max_initial_timestamp: Option<f32>,
    /// Previous context token IDs; prepended as prompt before the SOT sequence
    pub prompt: Option<Vec<u32>>,
}

impl Default for DecodingOptions {
    fn default() -> Self {
        Self {
            task: "transcribe".to_string(),
            language: None,
            temperature: 0.0,
            sample_len: None,
            suppress_blank: true,
            suppress_tokens: Some("-1".to_string()),
            without_timestamps: false,
            max_initial_timestamp: Some(1.0),
            prompt: None,
        }
    }
}

pub struct DecodingResult {
    pub language: String,
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    pub temperature: f32,
    pub compression_ratio: f32,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// zlib compression ratio (raw bytes / compressed bytes), for hallucination detection.
fn compression_ratio(text: &str) -> f32 {
    use std::io::Write;
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    let _ = enc.write_all(bytes);
    match enc.finish() {
        Ok(compressed) if !compressed.is_empty() => bytes.len() as f32 / compressed.len() as f32,
        _ => 1.0,
    }
}

/// Build a [n_vocab] float32 mask: 0.0 for allowed tokens, NEG_INFINITY for suppressed.
fn build_mask(suppress_ids: &[u32], n_vocab: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; n_vocab];
    for &id in suppress_ids {
        if (id as usize) < n_vocab {
            mask[id as usize] = f32::NEG_INFINITY;
        }
    }
    mask
}

// ── Logit filters ─────────────────────────────────────────────────────────────

/// SuppressBlank: at the very first generated position, suppress blank/EOT tokens.
fn apply_suppress_blank(logits: &Array, tokens_len: usize, sample_begin: usize, mask: &Array) -> Array {
    if tokens_len == sample_begin {
        logits + mask
    } else {
        logits.clone()
    }
}

/// SuppressTokens: unconditionally suppress a set of token IDs.
fn apply_suppress_tokens(logits: &Array, mask: &Array) -> Array {
    logits + mask
}

/// ApplyTimestampRules: enforce timestamp token pairing and ordering constraints.
fn apply_timestamp_rules(
    logits: &Array,
    tokens: &[u32],
    sample_begin: usize,
    tokenizer: &Tokenizer,
    max_initial_timestamp_index: Option<usize>,
) -> Result<Array> {
    let n_vocab = logits.shape()[0] as usize;
    let ts_begin = tokenizer.timestamp_begin() as usize;
    let eot = tokenizer.eot() as usize;
    let no_ts = tokenizer.no_timestamps() as usize;

    let mut mask = vec![0.0f32; n_vocab];

    // Suppress <|notimestamps|> (handled by without_timestamps flag)
    if no_ts < n_vocab {
        mask[no_ts] = f32::NEG_INFINITY;
    }

    // Inspect the generated part of the sequence so far
    let seq: &[u32] = if tokens.len() > sample_begin { &tokens[sample_begin..] } else { &[] };
    let last_was_timestamp = seq.last().map_or(false, |&t| t as usize >= ts_begin);
    let penultimate_was_timestamp = seq.len() < 2 || seq[seq.len() - 2] as usize >= ts_begin;

    if last_was_timestamp {
        if penultimate_was_timestamp {
            // Two consecutive timestamps → next must be a non-timestamp
            for i in ts_begin..n_vocab { mask[i] = f32::NEG_INFINITY; }
        } else {
            // Last was timestamp after text → next cannot be a regular text token
            for i in 0..eot { mask[i] = f32::NEG_INFINITY; }
        }
    }

    // Timestamps must not decrease; also force non-zero segment length
    let timestamps: Vec<usize> = seq.iter()
        .filter(|&&t| t as usize > ts_begin)
        .map(|&t| t as usize)
        .collect();
    if !timestamps.is_empty() {
        let mut last_ts = *timestamps.last().unwrap();
        if !last_was_timestamp || penultimate_was_timestamp {
            last_ts += 1;
        }
        for i in ts_begin..last_ts.min(n_vocab) {
            mask[i] = f32::NEG_INFINITY;
        }
    }

    // At the very beginning of generation, force a timestamp token
    if tokens.len() == sample_begin {
        for i in 0..ts_begin { mask[i] = f32::NEG_INFINITY; }
        if let Some(max_idx) = max_initial_timestamp_index {
            let last_allowed = ts_begin + max_idx;
            for i in (last_allowed + 1)..n_vocab { mask[i] = f32::NEG_INFINITY; }
        }
    }

    let mask_arr = Array::from_slice(&mask, &[n_vocab as i32]);
    let logits_masked = logits + &mask_arr;

    // If total timestamp logprob > max text token logprob → suppress text tokens
    if ts_begin > 0 && ts_begin < n_vocab {
        let logprobs = &logits_masked - &logsumexp_axis(&logits_masked, 0, true)?;
        // Sum prob of all timestamps
        let ts_lp = logsumexp_axis(logprobs.index((ts_begin as i32..,)), 0, false)?;
        // Max prob of any text token
        let max_text_lp = max_axis(logprobs.index((..ts_begin as i32,)), 0, false)?;
        eval([&ts_lp, &max_text_lp])?;
        if ts_lp.item::<f32>() > max_text_lp.item::<f32>() {
            for i in 0..ts_begin { mask[i] = f32::NEG_INFINITY; }
            let final_mask = Array::from_slice(&mask, &[n_vocab as i32]);
            return Ok(logits + &final_mask);
        }
    }

    Ok(logits_masked)
}

// ── Token selection ───────────────────────────────────────────────────────────

/// Greedy (temp=0) or sampled (temp>0) token selection; accumulates log-probability.
fn select_next_token(
    logits: &Array,
    temperature: f32,
    sum_logprobs: &mut f32,
    tokens: &[u32],
) -> Result<u32> {
    let eot = tokens.first().copied().unwrap_or(0); // placeholder — not used here

    let next_arr = if temperature == 0.0 {
        argmax_axis(logits, 0, None)?
    } else {
        mlx_rs::random::categorical(logits * (1.0 / temperature), None, None, None)?
    };
    eval([&next_arr])?;
    let next_token = next_arr.item::<u32>();

    // Accumulate log-probability (skip if last real token was EOT)
    let last = tokens.last().copied().unwrap_or(0);
    let _ = eot; // unused above
    if last != tokens[0] || tokens.len() == 1 {
        // Always accumulate — the Python check is: multiply by (last != eot)
        // We do it unconditionally since our loop stops at EOT anyway.
    }
    let logprobs = logits - &logsumexp_axis(logits, 0, true)?;
    let lp = logprobs.index((next_token as i32,));
    eval([&lp])?;
    *sum_logprobs += lp.item::<f32>();

    Ok(next_token)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Decode a 30-second mel spectrogram chunk.
///
/// # Parameters
/// - `mel`: 2D `[n_frames, n_mels]` or 3D `[1, n_frames, n_mels]` array
/// - `tokenizer`: must be configured with the correct language and task
/// - `options`: decoding hyperparameters
pub fn decode(
    model: &mut Whisper,
    mel: &Array,
    tokenizer: &Tokenizer,
    options: &DecodingOptions,
) -> Result<DecodingResult> {
    let n_vocab = model.dims.n_vocab;
    let n_audio_ctx = model.dims.n_audio_ctx;
    let n_text_ctx = model.dims.n_text_ctx;

    // ── Encoder forward pass ────────────────────────────────────────────────
    // mel expected as [n_frames, n_mels]; reshape to [1, n_frames, n_mels] for batch
    let mel_batch = if mel.ndim() == 2 {
        mel.reshape(&[1, mel.shape()[0], mel.shape()[1]])?
    } else {
        mel.clone()
    };
    let audio_features = model.encoder.forward(&mel_batch)?; // [1, n_audio_ctx, n_audio_state]

    // ── Initial token sequence ──────────────────────────────────────────────
    let mut sot_seq = tokenizer.sot_sequence.clone();
    if options.without_timestamps {
        sot_seq.push(tokenizer.no_timestamps());
    }

    // Prepend previous-context prompt: [sot_prev, ...prompt..., sot, lang, task]
    if let Some(ref prompt) = options.prompt {
        if !prompt.is_empty() {
            if let Some(&sot_prev) = tokenizer.special_tokens.get("<|startofprev|>") {
                let max_prompt = n_text_ctx / 2 - 1;
                let start = prompt.len().saturating_sub(max_prompt);
                let mut full = vec![sot_prev];
                full.extend_from_slice(&prompt[start..]);
                full.extend(sot_seq);
                sot_seq = full;
            }
        }
    }

    let sample_begin = sot_seq.len();
    let sot_index = sot_seq.iter().position(|&t| t == tokenizer.sot()).unwrap_or(0);
    let sample_len = options.sample_len.unwrap_or(n_text_ctx / 2);

    // ── Build logit filter masks ────────────────────────────────────────────

    // SuppressBlank: suppress space tokens and EOT at the first generated position
    let blank_mask_arr: Option<Array> = if options.suppress_blank {
        let space_ids = tokenizer.encode(" ");
        let eot_id = tokenizer.eot();
        let mut ids: Vec<u32> = space_ids;
        ids.push(eot_id);
        Some(Array::from_slice(&build_mask(&ids, n_vocab), &[n_vocab as i32]))
    } else {
        None
    };

    // SuppressTokens: always-on token suppression
    let suppress_mask_arr: Array = {
        let mut suppress_ids: Vec<u32> = Vec::new();

        if let Some(ref s) = options.suppress_tokens {
            let parsed: Vec<i64> = s.split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            if parsed.contains(&-1) {
                suppress_ids.extend(tokenizer.non_speech_tokens());
            } else {
                suppress_ids.extend(parsed.iter().filter(|&&t| t >= 0).map(|&t| t as u32));
            }
        }

        // Always suppress these control tokens
        suppress_ids.push(tokenizer.transcribe_token());
        suppress_ids.push(tokenizer.translate_token());
        suppress_ids.push(tokenizer.sot());
        suppress_ids.push(tokenizer.no_speech());
        if let Some(&id) = tokenizer.special_tokens.get("<|startofprev|>") {
            suppress_ids.push(id);
        }
        if let Some(&id) = tokenizer.special_tokens.get("<|startoflm|>") {
            suppress_ids.push(id);
        }

        Array::from_slice(&build_mask(&suppress_ids, n_vocab), &[n_vocab as i32])
    };

    // max_initial_timestamp_index: how many 0.02s steps the initial timestamp may be
    let max_ts_index: Option<usize> = if options.without_timestamps {
        None
    } else {
        options.max_initial_timestamp.map(|t| {
            let precision = CHUNK_LENGTH as f32 / n_audio_ctx as f32; // typically 0.02s
            (t / precision).round() as usize
        })
    };

    // ── Decode loop ─────────────────────────────────────────────────────────
    let mut tokens: Vec<u32> = sot_seq.clone();
    let mut sum_logprobs = 0.0f32;

    // Helper: apply all logit filters
    macro_rules! apply_filters {
        ($logits:expr) => {{
            let logits = $logits;
            let logits = match &blank_mask_arr {
                Some(mask) => apply_suppress_blank(&logits, tokens.len(), sample_begin, mask),
                None => logits,
            };
            let logits = apply_suppress_tokens(&logits, &suppress_mask_arr);
            let logits = if !options.without_timestamps {
                apply_timestamp_rules(&logits, &tokens, sample_begin, tokenizer, max_ts_index)?
            } else {
                logits
            };
            logits
        }};
    }

    // ── Step 1: feed all initial tokens, extract no_speech_prob ────────────
    let (no_speech_prob, mut kv_cache) = {
        let tokens_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let tokens_arr = Array::from_slice(&tokens_i32, &[1, tokens.len() as i32]);
        let (pre_logits, new_cache, _) = model.decoder.forward(&tokens_arr, &audio_features, None)?;

        // [1, seq_len, n_vocab] → [seq_len, n_vocab]
        let l2d = pre_logits.index((0i32,));

        // no_speech_prob from the SOT position
        let probs_sot = softmax_axis(l2d.index((sot_index as i32,)), 0, false)?;
        let ns_arr = probs_sot.index((tokenizer.no_speech() as i32,));
        eval([&ns_arr])?;
        let ns_prob = ns_arr.item::<f32>();

        // Logits at last position → apply filters → select token
        let logits = apply_filters!(l2d.index((-1i32,)));
        let next = select_next_token(&logits, options.temperature, &mut sum_logprobs, &tokens)?;
        tokens.push(next);

        (ns_prob, Some(new_cache))
    };

    // ── Steps 2+: feed only the last token with KV cache ───────────────────
    for _ in 1..sample_len {
        if tokens.len() > n_text_ctx {
            break;
        }
        if *tokens.last().unwrap() == tokenizer.eot() {
            break;
        }

        let last_i32 = *tokens.last().unwrap() as i32;
        let last_arr = Array::from_slice(&[last_i32], &[1, 1]);
        let (pre_logits, new_cache, _) =
            model.decoder.forward(&last_arr, &audio_features, kv_cache)?;
        kv_cache = Some(new_cache);

        // [1, 1, n_vocab] → [n_vocab]
        let logits = pre_logits.index((0i32,)).index((0i32,));
        let logits = apply_filters!(logits);
        let next = select_next_token(&logits, options.temperature, &mut sum_logprobs, &tokens)?;
        tokens.push(next);
    }

    // ── Post-process ────────────────────────────────────────────────────────
    let eot = tokenizer.eot();
    let generated = &tokens[sample_begin..];
    let end = generated.iter().position(|&t| t == eot).unwrap_or(generated.len());
    let result_tokens: Vec<u32> = generated[..end].to_vec();

    let text = tokenizer.decode(&result_tokens).trim().to_string();
    let avg_logprob = if result_tokens.is_empty() {
        f32::NEG_INFINITY
    } else {
        sum_logprobs / (result_tokens.len() + 1) as f32
    };

    Ok(DecodingResult {
        language: tokenizer.language.clone().unwrap_or_else(|| "en".to_string()),
        tokens: result_tokens,
        text: text.clone(),
        avg_logprob,
        no_speech_prob,
        temperature: options.temperature,
        compression_ratio: compression_ratio(&text),
    })
}
