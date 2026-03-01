// Translated from mlx-whisper/tokenizer.py (Apple Inc.)

use std::collections::HashMap;
use anyhow::{bail, Result};

// ── Language maps ──────────────────────────────────────────────────────────

pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "english"), ("zh", "chinese"), ("de", "german"), ("es", "spanish"),
    ("ru", "russian"), ("ko", "korean"), ("fr", "french"), ("ja", "japanese"),
    ("pt", "portuguese"), ("tr", "turkish"), ("pl", "polish"), ("ca", "catalan"),
    ("nl", "dutch"), ("ar", "arabic"), ("sv", "swedish"), ("it", "italian"),
    ("id", "indonesian"), ("hi", "hindi"), ("fi", "finnish"), ("vi", "vietnamese"),
    ("he", "hebrew"), ("uk", "ukrainian"), ("el", "greek"), ("ms", "malay"),
    ("cs", "czech"), ("ro", "romanian"), ("da", "danish"), ("hu", "hungarian"),
    ("ta", "tamil"), ("no", "norwegian"), ("th", "thai"), ("ur", "urdu"),
    ("hr", "croatian"), ("bg", "bulgarian"), ("lt", "lithuanian"), ("la", "latin"),
    ("mi", "maori"), ("ml", "malayalam"), ("cy", "welsh"), ("sk", "slovak"),
    ("te", "telugu"), ("fa", "persian"), ("lv", "latvian"), ("bn", "bengali"),
    ("sr", "serbian"), ("az", "azerbaijani"), ("sl", "slovenian"), ("kn", "kannada"),
    ("et", "estonian"), ("mk", "macedonian"), ("br", "breton"), ("eu", "basque"),
    ("is", "icelandic"), ("hy", "armenian"), ("ne", "nepali"), ("mn", "mongolian"),
    ("bs", "bosnian"), ("kk", "kazakh"), ("sq", "albanian"), ("sw", "swahili"),
    ("gl", "galician"), ("mr", "marathi"), ("pa", "punjabi"), ("si", "sinhala"),
    ("km", "khmer"), ("sn", "shona"), ("yo", "yoruba"), ("so", "somali"),
    ("af", "afrikaans"), ("oc", "occitan"), ("ka", "georgian"), ("be", "belarusian"),
    ("tg", "tajik"), ("sd", "sindhi"), ("gu", "gujarati"), ("am", "amharic"),
    ("yi", "yiddish"), ("lo", "lao"), ("uz", "uzbek"), ("fo", "faroese"),
    ("ht", "haitian creole"), ("ps", "pashto"), ("tk", "turkmen"), ("nn", "nynorsk"),
    ("mt", "maltese"), ("sa", "sanskrit"), ("lb", "luxembourgish"), ("my", "myanmar"),
    ("bo", "tibetan"), ("tl", "tagalog"), ("mg", "malagasy"), ("as", "assamese"),
    ("tt", "tatar"), ("haw", "hawaiian"), ("ln", "lingala"), ("ha", "hausa"),
    ("ba", "bashkir"), ("jw", "javanese"), ("su", "sundanese"), ("yue", "cantonese"),
];

pub fn to_language_code(name: &str) -> Option<&'static str> {
    // 直接比對 code
    if LANGUAGES.iter().any(|(code, _)| *code == name) {
        return Some(LANGUAGES.iter().find(|(c, _)| *c == name).unwrap().0);
    }
    // 比對語言名稱
    if let Some((code, _)) = LANGUAGES.iter().find(|(_, lang)| *lang == name) {
        return Some(code);
    }
    // aliases
    match name {
        "burmese" => Some("my"), "valencian" => Some("ca"), "flemish" => Some("nl"),
        "haitian" => Some("ht"), "letzeburgesch" => Some("lb"), "pushto" => Some("ps"),
        "panjabi" => Some("pa"), "moldavian" | "moldovan" => Some("ro"),
        "sinhalese" => Some("si"), "castilian" => Some("es"), "mandarin" => Some("zh"),
        _ => None,
    }
}

// ── BPE Tokenizer ──────────────────────────────────────────────────────────

/// rank → bytes（decode 用）
type Vocab = HashMap<u32, Vec<u8>>;
/// bytes → rank（encode 用）
type Encoder = HashMap<Vec<u8>, u32>;

pub struct Encoding {
    pub name: String,
    vocab: Vocab,
    encoder: Encoder,
    pub special_tokens: HashMap<String, u32>,
    pub eot_token: u32,
}

impl Encoding {
    /// 從 .tiktoken 格式載入 vocab，並加上 special tokens
    /// .tiktoken 格式：每行 `<base64_bytes> <rank>`
    pub fn from_tiktoken_file(
        path: &std::path::Path,
        name: &str,
        num_languages: usize,
    ) -> Result<Self> {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD;

        let content = std::fs::read_to_string(path)?;
        let mut vocab: Vocab = HashMap::new();
        let mut encoder: Encoder = HashMap::new();

        for line in content.lines().filter(|l| !l.is_empty()) {
            let mut parts = line.split_whitespace();
            let token_b64 = parts.next().ok_or_else(|| anyhow::anyhow!("bad line"))?;
            let rank: u32 = parts.next().ok_or_else(|| anyhow::anyhow!("bad rank"))?.parse()?;
            // Python の base64.b64decode("=") は b'' を返す（寛容な実装）。
            // Rust は厳格に拒否するので、"=" や空文字は空 bytes として扱う。
            let bytes = if token_b64 == "=" || token_b64.is_empty() {
                vec![]
            } else {
                b64.decode(token_b64)?
            };
            encoder.insert(bytes.clone(), rank);
            vocab.insert(rank, bytes);
        }

        let n_vocab = vocab.len() as u32;

        // Special tokens（與 Python get_encoding 完全對應）
        let mut special_tokens: HashMap<String, u32> = HashMap::new();
        let mut next_id = n_vocab;

        let mut add_special = |token: String, map: &mut HashMap<String, u32>| {
            map.entry(token).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
        };

        add_special("<|endoftext|>".into(), &mut special_tokens);
        add_special("<|startoftranscript|>".into(), &mut special_tokens);
        for (code, _) in LANGUAGES.iter().take(num_languages) {
            add_special(format!("<|{code}|>"), &mut special_tokens);
        }
        for token in ["<|translate|>", "<|transcribe|>", "<|startoflm|>",
                       "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>"] {
            add_special(token.into(), &mut special_tokens);
        }
        for i in 0..1501u32 {
            add_special(format!("<|{:.2}|>", i as f32 * 0.02), &mut special_tokens);
        }

        // special token bytes も encoder に追加
        for (token, &id) in &special_tokens {
            let bytes = token.as_bytes().to_vec();
            vocab.insert(id, bytes.clone());
            encoder.insert(bytes, id);
        }

        let eot_token = *special_tokens.get("<|endoftext|>").unwrap();

        Ok(Self {
            name: name.to_string(),
            vocab,
            encoder,
            special_tokens,
            eot_token,
        })
    }

    /// token IDs → UTF-8 text（special tokens は除外しない）
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let bytes: Vec<u8> = token_ids
            .iter()
            .filter_map(|id| self.vocab.get(id))
            .flatten()
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// text → token IDs（BPE encode）
    pub fn encode(&self, text: &str) -> Vec<u32> {
        bpe_encode(text.as_bytes(), &self.encoder)
    }

    /// encode a single special token（常數時間）
    pub fn encode_single_token(&self, token: &str) -> Option<u32> {
        self.encoder.get(token.as_bytes()).copied()
            .or_else(|| self.special_tokens.get(token).copied())
    }
}

/// 最小化 BPE encode
fn bpe_encode(text: &[u8], encoder: &Encoder) -> Vec<u32> {
    if text.is_empty() {
        return vec![];
    }
    // 初始化：每個 byte 一個 token
    let mut tokens: Vec<Vec<u8>> = text.iter().map(|&b| vec![b]).collect();

    loop {
        // 找出相鄰 pair 中 rank 最低的
        let best = tokens.windows(2).enumerate().filter_map(|(i, pair)| {
            let merged: Vec<u8> = pair[0].iter().chain(pair[1].iter()).copied().collect();
            encoder.get(&merged).map(|&rank| (i, rank, merged))
        }).min_by_key(|&(_, rank, _)| rank);

        match best {
            None => break, // 沒有可以合併的 pair
            Some((i, _, merged)) => {
                tokens[i] = merged;
                tokens.remove(i + 1);
            }
        }
    }

    tokens.iter()
        .filter_map(|t| encoder.get(t))
        .copied()
        .collect()
}

// ── Tokenizer ──────────────────────────────────────────────────────────────

pub struct Tokenizer {
    pub encoding: Encoding,
    pub num_languages: usize,
    pub language: Option<String>,
    pub task: Option<String>,
    pub sot_sequence: Vec<u32>,
    pub special_tokens: HashMap<String, u32>,
}

impl Tokenizer {
    pub fn new(
        encoding: Encoding,
        num_languages: usize,
        language: Option<String>,
        task: Option<String>,
    ) -> Self {
        let special_tokens = encoding.special_tokens.clone();

        let sot = special_tokens["<|startoftranscript|>"];
        let translate = special_tokens["<|translate|>"];
        let transcribe = special_tokens["<|transcribe|>"];

        let mut sot_sequence = vec![sot];
        if let Some(ref lang) = language {
            let lang_idx = LANGUAGES.iter().position(|(code, _)| code == lang).unwrap_or(0);
            sot_sequence.push(sot + 1 + lang_idx as u32);
        }
        if let Some(ref task_str) = task {
            sot_sequence.push(if task_str == "transcribe" { transcribe } else { translate });
        }

        Self { encoding, num_languages, language, task, sot_sequence, special_tokens }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encoding.encode(text)
    }

    /// decode，過濾所有 special tokens（language tags, timestamps, task tokens 等）。
    /// 只保留 t < eot_token（= n_vocab）的正常詞彙 token。
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let eot = self.encoding.eot_token; // == n_vocab，第一個 special token
        let filtered: Vec<u32> = token_ids.iter().copied()
            .filter(|&t| t < eot)
            .collect();
        self.encoding.decode(&filtered)
    }

    pub fn decode_with_timestamps(&self, token_ids: &[u32]) -> String {
        self.encoding.decode(token_ids)
    }

    pub fn eot(&self) -> u32 { self.encoding.eot_token }
    pub fn sot(&self) -> u32 { self.special_tokens["<|startoftranscript|>"] }
    pub fn transcribe_token(&self) -> u32 { self.special_tokens["<|transcribe|>"] }
    pub fn translate_token(&self) -> u32 { self.special_tokens["<|translate|>"] }
    pub fn no_speech(&self) -> u32 { self.special_tokens["<|nospeech|>"] }
    pub fn no_timestamps(&self) -> u32 { self.special_tokens["<|notimestamps|>"] }
    pub fn timestamp_begin(&self) -> u32 { self.special_tokens["<|0.00|>"] }

    pub fn sot_sequence_including_notimestamps(&self) -> Vec<u32> {
        let mut seq = self.sot_sequence.clone();
        seq.push(self.no_timestamps());
        seq
    }

    pub fn language_token(&self) -> Result<u32> {
        let lang = self.language.as_deref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer has no language configured"))?;
        self.to_language_token(lang)
    }

    pub fn to_language_token(&self, language: &str) -> Result<u32> {
        let key = format!("<|{language}|>");
        self.special_tokens.get(&key)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Language {language} not found in tokenizer"))
    }

    pub fn all_language_tokens(&self) -> Vec<u32> {
        LANGUAGES.iter()
            .take(self.num_languages)
            .filter_map(|(code, _)| self.special_tokens.get(&format!("<|{code}|>")).copied())
            .collect()
    }

    pub fn all_language_codes(&self) -> Vec<String> {
        self.all_language_tokens()
            .iter()
            .map(|&t| {
                let s = self.decode_with_timestamps(&[t]);
                s.trim_matches(|c| c == '<' || c == '|' || c == '>').to_string()
            })
            .collect()
    }

    /// Returns token IDs to suppress (non-speech symbols, brackets, etc.)
    pub fn non_speech_tokens(&self) -> Vec<u32> {
        let mut result = std::collections::HashSet::new();

        // space-dash and space-apostrophe: allow only between words
        if let Some(&t) = self.encode(" -").first() { result.insert(t); }
        if let Some(&t) = self.encode(" '").first() { result.insert(t); }

        let symbols = [
            "\"", "#", "(", ")", "*", "+", "/", ":", ";", "<", "=", ">", "@",
            "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
            "「", "」", "『", "』",
            "<<", ">>", "<<<", ">>>", "--", "---", "-(", "-[",
            "('", "(\"", "((", "))", "(((", ")))", "[[", "]]", "{{", "}}", "♪♪", "♪♪♪",
        ];
        // U+2640–U+267F miscellaneous symbols: suppress first byte token
        let miscellaneous = ["♩", "♪", "♫", "♬", "♭", "♮", "♯"];

        for symbol in &symbols {
            let encoded = self.encode(symbol);
            if encoded.len() == 1 { result.insert(encoded[0]); }
            let with_space = self.encode(&format!(" {symbol}"));
            if with_space.len() == 1 { result.insert(with_space[0]); }
        }
        for symbol in &miscellaneous {
            let encoded = self.encode(symbol);
            if !encoded.is_empty() { result.insert(encoded[0]); }
            let with_space = self.encode(&format!(" {symbol}"));
            if !with_space.is_empty() { result.insert(with_space[0]); }
        }

        let mut tokens: Vec<u32> = result.into_iter().collect();
        tokens.sort();
        tokens
    }

    pub fn split_to_word_tokens(&self, tokens: &[u32]) -> (Vec<String>, Vec<Vec<u32>>) {
        let no_space_langs = ["zh", "ja", "th", "lo", "my", "yue"];
        if self.language.as_deref().map_or(false, |l| no_space_langs.contains(&l)) {
            self.split_tokens_on_unicode(tokens)
        } else {
            self.split_tokens_on_spaces(tokens)
        }
    }

    pub fn split_tokens_on_unicode(&self, tokens: &[u32]) -> (Vec<String>, Vec<Vec<u32>>) {
        let decoded_full = self.decode_with_timestamps(tokens);
        let replacement = '\u{FFFD}';
        let mut words = Vec::new();
        let mut word_tokens: Vec<Vec<u32>> = Vec::new();
        let mut current_tokens: Vec<u32> = Vec::new();
        let mut unicode_offset = 0;

        for &token in tokens {
            current_tokens.push(token);
            let decoded = self.decode_with_timestamps(&current_tokens);
            let has_replacement = decoded.contains(replacement);
            let offset_char = decoded_full.chars().nth(
                unicode_offset + decoded.chars().position(|c| c == replacement).unwrap_or(0)
            );
            if !has_replacement || offset_char == Some(replacement) {
                unicode_offset += decoded.chars().count();
                words.push(decoded);
                word_tokens.push(std::mem::take(&mut current_tokens));
            }
        }
        (words, word_tokens)
    }

    pub fn split_tokens_on_spaces(&self, tokens: &[u32]) -> (Vec<String>, Vec<Vec<u32>>) {
        let (subwords, subword_tokens_list) = self.split_tokens_on_unicode(tokens);
        let mut words: Vec<String> = Vec::new();
        let mut word_tokens: Vec<Vec<u32>> = Vec::new();

        for (subword, subtokens) in subwords.iter().zip(subword_tokens_list.iter()) {
            let special = subtokens.first().map_or(false, |&t| t >= self.eot());
            let with_space = subword.starts_with(' ');
            let punctuation = subword.trim().len() == 1
                && subword.trim().chars().all(|c: char| c.is_ascii_punctuation());
            if special || with_space || punctuation || words.is_empty() {
                words.push(subword.clone());
                word_tokens.push(subtokens.clone());
            } else {
                words.last_mut().unwrap().push_str(subword);
                word_tokens.last_mut().unwrap().extend(subtokens);
            }
        }
        (words, word_tokens)
    }
}

// ── Factory ────────────────────────────────────────────────────────────────

/// 對應 Python: get_tokenizer()
pub fn get_tokenizer(
    multilingual: bool,
    num_languages: usize,
    language: Option<&str>,
    task: Option<&str>,
    assets_dir: &std::path::Path,
) -> Result<Tokenizer> {
    let language = language.map(|l| l.to_lowercase());
    let language = if let Some(ref l) = language {
        if LANGUAGES.iter().any(|(code, _)| code == l) {
            Some(l.as_str())
        } else if let Some(code) = to_language_code(l) {
            Some(code)
        } else {
            bail!("Unsupported language: {l}");
        }
    } else {
        None
    };

    let (encoding_name, language, task) = if multilingual {
        (
            "multilingual",
            Some(language.unwrap_or("en").to_string()),
            Some(task.unwrap_or("transcribe").to_string()),
        )
    } else {
        ("gpt2", None, None)
    };

    let vocab_path = assets_dir.join(format!("{encoding_name}.tiktoken"));
    let encoding = Encoding::from_tiktoken_file(&vocab_path, encoding_name, num_languages)?;

    Ok(Tokenizer::new(encoding, num_languages, language, task))
}
