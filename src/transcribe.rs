// Translated from mlx-whisper/transcribe.py (Apple Inc.)

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use mlx_rs::{Array, ops::{indexing::IndexOp, softmax_axis}, transforms::eval};

use crate::audio::{
    log_mel_spectrogram, pad_or_trim, HOP_LENGTH, N_FRAMES, SAMPLE_RATE,
};
use crate::decoding::{decode, DecodingOptions, DecodingResult};
use crate::tokenizer::{get_tokenizer, LANGUAGES};
use crate::whisper::Whisper;

// ── Types ─────────────────────────────────────────────────────────────────────

pub struct Segment {
    pub id: usize,
    /// Frame offset in the original mel spectrogram
    pub seek: usize,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    pub no_speech_prob: f32,
}

pub struct TranscribeResult {
    pub text: String,
    pub segments: Vec<Segment>,
    pub language: String,
}

pub struct TranscribeOptions {
    /// Language code (e.g. "en", "zh"); None → auto-detect for multilingual models
    pub language: Option<String>,
    pub task: String,
    /// Temperature schedule: tries each value in order if quality check fails
    pub temperatures: Vec<f32>,
    /// Compression ratio above this → retry with next temperature (too repetitive)
    pub compression_ratio_threshold: Option<f32>,
    /// Avg log-probability below this → retry with next temperature
    pub logprob_threshold: Option<f32>,
    /// No-speech probability above this → skip the segment (silence detection)
    pub no_speech_threshold: Option<f32>,
    /// Feed the previous window's output as a prompt for the next window
    pub condition_on_previous_text: bool,
    /// Text to prepend as context for the first window
    pub initial_prompt: Option<String>,
    /// Suppress blank tokens at the first generated position
    pub suppress_blank: bool,
    /// Token suppression: "-1" = non-speech symbols, or comma-separated IDs
    pub suppress_tokens: Option<String>,
    pub without_timestamps: bool,
    /// Print decoded text to stderr as it is produced
    pub verbose: bool,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            task: "transcribe".to_string(),
            temperatures: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold: Some(2.4),
            logprob_threshold: Some(-1.0),
            no_speech_threshold: Some(0.6),
            condition_on_previous_text: true,
            initial_prompt: None,
            suppress_blank: true,
            suppress_tokens: Some("-1".to_string()),
            without_timestamps: false,
            verbose: false,
        }
    }
}

// ── Language detection ────────────────────────────────────────────────────────

/// Detect the spoken language in a mel spectrogram.
///
/// Returns `(best_language_code, probabilities)` where `probabilities` maps
/// each supported language code to its probability.
///
/// `mel` can be:
/// - 2-D `[n_frames, n_mels]` — will be padded/trimmed to 30 s and batched
/// - 3-D `[1, n_frames, n_mels]` — used as-is (must already be 30-s window)
///
/// Only meaningful for multilingual models; returns `"en"` with p=1.0 for
/// English-only models.
pub fn detect_language(
    model: &mut Whisper,
    mel: &Array,
    assets_dir: &Path,
) -> Result<(String, HashMap<String, f32>)> {
    let n_mels = model.dims.n_mels;

    if !model.is_multilingual() {
        let mut probs = HashMap::new();
        probs.insert("en".to_string(), 1.0f32);
        return Ok(("en".to_string(), probs));
    }

    // Normalise to [1, N_FRAMES, n_mels]
    let mel_batch = match mel.ndim() {
        2 => pad_or_trim(mel.clone(), N_FRAMES)?
            .reshape(&[1, N_FRAMES as i32, n_mels as i32])?,
        3 => mel.clone(),
        d => anyhow::bail!("mel must be 2-D or 3-D, got {d}-D"),
    };

    let audio_features = model.encoder.forward(&mel_batch)?;
    let num_langs = model.num_languages();
    let bare_tok = get_tokenizer(model.is_multilingual(), num_langs, None, None, assets_dir)?;

    // Feed the SOT token, look at position 0 logits
    let sot_arr = Array::from_slice(&[bare_tok.sot() as i32], &[1, 1]);
    let (logits, _, _) = model.decoder.forward(&sot_arr, &audio_features, None)?;
    let logits_1d = logits.index((0i32,)).index((0i32,)); // [n_vocab]

    // Mask out all non-language tokens
    let n_vocab = model.dims.n_vocab;
    let lang_tokens = bare_tok.all_language_tokens();
    let lang_codes = bare_tok.all_language_codes();
    let mut mask = vec![f32::NEG_INFINITY; n_vocab];
    for &lt in &lang_tokens {
        if (lt as usize) < n_vocab {
            mask[lt as usize] = 0.0;
        }
    }
    let mask_arr = Array::from_slice(&mask, &[n_vocab as i32]);
    let probs_arr = softmax_axis(&logits_1d + &mask_arr, 0, false)?;
    eval([&probs_arr])?;
    let probs_data: &[f32] = probs_arr.as_slice();

    // Build probability map and find best language
    let mut lang_probs: HashMap<String, f32> = HashMap::new();
    let mut best_lang = "en".to_string();
    let mut best_prob = 0.0f32;
    for (&tok, code) in lang_tokens.iter().zip(lang_codes.iter()) {
        let p = probs_data.get(tok as usize).copied().unwrap_or(0.0);
        if p > best_prob {
            best_prob = p;
            best_lang = code.clone();
        }
        lang_probs.insert(code.clone(), p);
    }

    Ok((best_lang, lang_probs))
}

/// Internal helper used by `transcribe()`: accepts a pre-batched mel and returns
/// only the best language string.
fn detect_language_best(
    model: &mut Whisper,
    mel_batch: &Array,
    assets_dir: &Path,
) -> Result<String> {
    let (lang, _) = detect_language(model, mel_batch, assets_dir)?;
    Ok(lang)
}

// ── Fallback decode ───────────────────────────────────────────────────────────

fn decode_with_fallback(
    model: &mut Whisper,
    mel_batch: &Array,
    tokenizer: &crate::tokenizer::Tokenizer,
    base_options: &DecodingOptions,
    temperatures: &[f32],
    compression_ratio_threshold: Option<f32>,
    logprob_threshold: Option<f32>,
    no_speech_threshold: Option<f32>,
) -> Result<DecodingResult> {
    let mut result: Option<DecodingResult> = None;

    for &temp in temperatures {
        let mut opts = base_options.clone();
        opts.temperature = temp;
        let r = decode(model, mel_batch, tokenizer, &opts)?;

        let mut needs_fallback = false;
        if let Some(crt) = compression_ratio_threshold {
            if r.compression_ratio > crt {
                needs_fallback = true;
            }
        }
        if let Some(lpt) = logprob_threshold {
            if r.avg_logprob < lpt {
                needs_fallback = true;
            }
        }
        // No-speech probability overrides: if it looks like silence, don't retry
        if let Some(nst) = no_speech_threshold {
            if r.no_speech_prob > nst {
                needs_fallback = false;
            }
        }

        let done = !needs_fallback;
        result = Some(r);
        if done {
            break;
        }
    }

    result.ok_or_else(|| anyhow::anyhow!("No decode result produced"))
}

// ── Timestamp formatting ──────────────────────────────────────────────────────

fn format_timestamp(seconds: f32) -> String {
    let ms = (seconds * 1000.0).round() as u64;
    let h = ms / 3_600_000;
    let ms = ms % 3_600_000;
    let m = ms / 60_000;
    let ms = ms % 60_000;
    let s = ms / 1_000;
    let ms = ms % 1_000;
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}.{ms:03}")
    } else {
        format!("{m:02}:{s:02}.{ms:03}")
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Transcribe a 16 kHz mono audio array using a sliding 30-second window.
///
/// # Parameters
/// - `audio`: 1-D float32 array of 16 kHz mono samples
/// - `model`: pre-loaded Whisper model
/// - `assets_dir`: directory containing `*.tiktoken` vocab files and `mel_filters_*.npy`
/// - `options`: transcription settings
pub fn transcribe(
    audio: Array,
    model: &mut Whisper,
    assets_dir: &Path,
    options: &TranscribeOptions,
) -> Result<TranscribeResult> {
    let n_mels = model.dims.n_mels;
    let n_audio_ctx = model.dims.n_audio_ctx;

    // ── Compute mel spectrogram ─────────────────────────────────────────────
    let mel = log_mel_spectrogram(audio, n_mels, assets_dir)?; // [n_frames, n_mels]
    let content_frames = mel.shape()[0] as usize;             // total mel frames

    // ── Determine language ──────────────────────────────────────────────────
    let language: String = if let Some(ref lang) = options.language {
        lang.clone()
    } else if !model.is_multilingual() {
        "en".to_string()
    } else {
        if options.verbose {
            eprintln!("Detecting language from first 30 seconds…");
        }
        // Take first N_FRAMES (or less) and pad to N_FRAMES
        let first_segment_size = content_frames.min(N_FRAMES);
        let mel_slice = mel.index((..first_segment_size as i32,));
        let mel_30s = pad_or_trim(mel_slice, N_FRAMES)?
            .reshape(&[1, N_FRAMES as i32, n_mels as i32])?;
        let detected = detect_language_best(model, &mel_30s, assets_dir)?;
        if options.verbose {
            let lang_name = LANGUAGES
                .iter()
                .find(|(c, _)| *c == detected)
                .map(|(_, n)| *n)
                .unwrap_or(&detected);
            let mut name = lang_name.to_string();
            if let Some(c) = name.get_mut(0..1) {
                c.make_ascii_uppercase();
            }
            eprintln!("Detected language: {name}");
        }
        detected
    };

    // ── Build tokenizer ─────────────────────────────────────────────────────
    let tokenizer = get_tokenizer(
        model.is_multilingual(),
        model.num_languages(),
        Some(&language),
        Some(&options.task),
        assets_dir,
    )?;

    // ── Timing constants ────────────────────────────────────────────────────
    // input_stride: mel frames per decoder output token (typically 2)
    let input_stride = N_FRAMES / n_audio_ctx;
    // time_precision: seconds per decoder output token (typically 0.02 s)
    let time_precision = input_stride as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32;

    // ── Build base DecodingOptions (shared across all windows) ─────────────
    let base_decode_opts = DecodingOptions {
        task: options.task.clone(),
        language: Some(language.clone()),
        temperature: options.temperatures.first().copied().unwrap_or(0.0),
        suppress_blank: options.suppress_blank,
        suppress_tokens: options.suppress_tokens.clone(),
        without_timestamps: options.without_timestamps,
        max_initial_timestamp: Some(1.0),
        prompt: None, // filled per-window below
        ..Default::default()
    };

    // ── Prepare initial prompt tokens ───────────────────────────────────────
    let mut all_tokens: Vec<u32> = Vec::new();
    let mut prompt_reset_since = 0usize;

    if let Some(ref init_prompt) = options.initial_prompt {
        let pt = tokenizer.encode(&format!(" {}", init_prompt.trim()));
        all_tokens.extend_from_slice(&pt);
        prompt_reset_since = 0; // include initial prompt always
    }

    // ── Sliding window decode loop ──────────────────────────────────────────
    let mut all_segments: Vec<Segment> = Vec::new();
    let mut seek = 0usize;

    while seek < content_frames {
        let time_offset = seek as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32;
        let segment_size = N_FRAMES.min(content_frames - seek);
        let segment_duration = segment_size as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32;

        // Slice mel and pad/trim to exactly N_FRAMES
        let mel_slice = mel.index((seek as i32..(seek + segment_size) as i32,));
        let mel_segment = pad_or_trim(mel_slice, N_FRAMES)?
            .reshape(&[1, N_FRAMES as i32, n_mels as i32])?;

        // Build per-window prompt from accumulated tokens
        let prompt: Option<Vec<u32>> = if options.condition_on_previous_text {
            let p = &all_tokens[prompt_reset_since..];
            if p.is_empty() { None } else { Some(p.to_vec()) }
        } else {
            None
        };
        let mut window_opts = base_decode_opts.clone();
        window_opts.prompt = prompt;

        // Decode with temperature fallback
        let result = decode_with_fallback(
            model,
            &mel_segment,
            &tokenizer,
            &window_opts,
            &options.temperatures,
            options.compression_ratio_threshold,
            options.logprob_threshold,
            options.no_speech_threshold,
        )?;

        // ADDED FOR DEBUGGING
        if options.verbose {
            eprintln!("DEBUG: window decode result tokens = {:?}", result.tokens);
            eprintln!("DEBUG: window decode result text = {:?}", result.text);
        }

        let tokens = &result.tokens;
        let previous_seek = seek;

        // ── Silence check ───────────────────────────────────────────────────
        if let Some(nst) = options.no_speech_threshold {
            let should_skip = result.no_speech_prob > nst
                && options
                    .logprob_threshold
                    .map_or(true, |lpt| result.avg_logprob < lpt);
            if should_skip {
                seek += segment_size;
                continue;
            }
        }

        // ── Parse timestamp tokens into segments ────────────────────────────
        let ts_begin = tokenizer.timestamp_begin();
        let ts_flags: Vec<bool> = tokens.iter().map(|&t| t >= ts_begin).collect();

        // Is the last token a single (unpaired) timestamp?
        let single_ts_ending = ts_flags.len() >= 2
            && !ts_flags[ts_flags.len() - 2]
            && *ts_flags.last().unwrap_or(&false);

        // Find positions where two consecutive timestamp tokens appear
        let consecutive: Vec<usize> = (0..ts_flags.len().saturating_sub(1))
            .filter(|&i| ts_flags[i] && ts_flags[i + 1])
            .map(|i| i + 1)
            .collect();

        let mut current_segments: Vec<Segment> = Vec::new();

        if !consecutive.is_empty() {
            let mut slices = consecutive.clone();
            if single_ts_ending {
                slices.push(tokens.len());
            }

            let mut last_slice = 0usize;
            for &current_slice in &slices {
                let sliced = &tokens[last_slice..current_slice];
                if sliced.is_empty() {
                    last_slice = current_slice;
                    continue;
                }
                let start_ts = sliced[0].saturating_sub(ts_begin) as f32;
                let end_ts = sliced[sliced.len() - 1].saturating_sub(ts_begin) as f32;
                let text_tokens: Vec<u32> = sliced.iter().copied()
                    .filter(|&t| t < ts_begin)
                    .collect();
                current_segments.push(Segment {
                    id: all_segments.len() + current_segments.len(),
                    seek: previous_seek,
                    start: time_offset + start_ts * time_precision,
                    end: time_offset + end_ts * time_precision,
                    text: tokenizer.decode(&text_tokens).trim().to_string(),
                    tokens: sliced.to_vec(),
                    temperature: result.temperature,
                    avg_logprob: result.avg_logprob,
                    compression_ratio: result.compression_ratio,
                    no_speech_prob: result.no_speech_prob,
                });
                last_slice = current_slice;
            }

            if single_ts_ending {
                seek += segment_size;
            } else {
                // Seek to position of the last consumed timestamp
                let last_ts = tokens
                    .get(last_slice.saturating_sub(1))
                    .copied()
                    .unwrap_or(ts_begin)
                    .saturating_sub(ts_begin) as usize;
                seek += last_ts * input_stride;
            }
        } else {
            // No consecutive timestamp pairs — use full segment duration or last timestamp
            let mut duration = segment_duration;
            let ts_vals: Vec<u32> = tokens.iter().copied().filter(|&t| t >= ts_begin).collect();
            if let Some(&last_ts) = ts_vals.last() {
                if last_ts != ts_begin {
                    duration = (last_ts - ts_begin) as f32 * time_precision;
                }
            }
            let text_tokens: Vec<u32> = tokens.iter().copied()
                .filter(|&t| t < ts_begin)
                .collect();
            current_segments.push(Segment {
                id: all_segments.len(),
                seek: previous_seek,
                start: time_offset,
                end: time_offset + duration,
                text: tokenizer.decode(&text_tokens).trim().to_string(),
                tokens: tokens.clone(),
                temperature: result.temperature,
                avg_logprob: result.avg_logprob,
                compression_ratio: result.compression_ratio,
                no_speech_prob: result.no_speech_prob,
            });
            seek += segment_size;
        }

        // ── Verbose output ──────────────────────────────────────────────────
        if options.verbose {
            for seg in &current_segments {
                eprintln!(
                    "[{} --> {}] text={:?} tokens={:?}",
                    format_timestamp(seg.start),
                    format_timestamp(seg.end),
                    seg.text,
                    seg.tokens
                );
            }
        }

        // ── Accumulate tokens for condition_on_previous_text ────────────────
        for seg in &current_segments {
            all_tokens.extend_from_slice(&seg.tokens);
        }
        // If temperature was high or condition disabled, reset the prompt cursor
        if !options.condition_on_previous_text || result.temperature > 0.5 {
            prompt_reset_since = all_tokens.len();
        }

        // Drop empty/whitespace-only segments
        let current_segments: Vec<Segment> = current_segments
            .into_iter()
            .filter(|s| s.start < s.end && !s.text.is_empty())
            .collect();

        all_segments.extend(current_segments);

        // Guard against seek not advancing (avoid infinite loop)
        if seek <= previous_seek {
            seek = previous_seek + segment_size;
        }
    }

    // ── Assemble full text ──────────────────────────────────────────────────
    // Re-index segment IDs sequentially
    for (i, seg) in all_segments.iter_mut().enumerate() {
        seg.id = i;
    }

    // Decode all accumulated tokens (excluding the initial prompt) for the full text
    let initial_prompt_len = if options.initial_prompt.is_some() {
        // The initial prompt tokens were prepended before the loop
        // Find where the loop tokens start by subtracting initial prompt length
        all_tokens
            .len()
            .min(tokenizer.encode(&format!(" {}", options.initial_prompt.as_deref().unwrap_or("").trim())).len())
    } else {
        0
    };
    let full_text = tokenizer.decode(&all_tokens[initial_prompt_len..]).trim().to_string();

    Ok(TranscribeResult {
        text: full_text,
        segments: all_segments,
        language,
    })
}
