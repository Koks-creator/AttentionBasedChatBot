"""WordPiece tokenizer w czystym Pythonie.

Zamiennik keras_hub.tokenizers.WordPieceTokenizer do uruchamiania modelu na
Windowsie, gdzie nie ma pakietu tensorflow-text (keras-hub go wymaga do
tokenizacji). Odtwarza ustawienia z notebooka: lowercase=True,
strip_accents=False, split=True, split_on_cjk=True, sufiks "##", oov "[UNK]",
więc daje te same ID tokenów co keras-hub w Colabie.

Użycie:
    from wordpiece_local import LocalWordPieceTokenizer
    tokenizer = LocalWordPieceTokenizer("vocab.txt")
"""
import unicodedata

import numpy as np

# Zakresy znaków CJK (jak w keras-hub): każdy taki znak to osobny token.
_CJK_RANGES = [
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
]
# Jak domyślnie w tensorflow-text: dłuższe "słowo" (w bajtach UTF-8) → [UNK].
MAX_BYTES_PER_WORD = 100


def _is_dropped_separator(ch):
    """Tniemy tu i wyrzucamy znak: spacja oraz znaki sterujące/formatujące."""
    return ch == " " or unicodedata.category(ch) in ("Cc", "Cf")


def _is_kept_separator(ch):
    """Tniemy tu, ale znak zostaje osobnym tokenem: interpunkcja i CJK."""
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True  # cały nie-alfanumeryczny ASCII, także $ ^ ` itp.
    if unicodedata.category(ch).startswith("P"):
        return True  # interpunkcja Unicode, np. „ ” …
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _case_fold(word):
    """Małe litery jak w tensorflow-text (case folding + normalizacja NFKC)."""
    return unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", word).casefold())


class LocalWordPieceTokenizer:
    def __init__(self, vocab_path, oov_token="[UNK]", suffix_indicator="##"):
        # encoding="utf-8" jest kluczowe na Windowsie (domyślnie byłoby cp1250).
        # rstrip zamiast strip: usuwamy tylko znak końca linii, nic z samego tokenu.
        with open(vocab_path, encoding="utf-8") as f:
            self.vocabulary = [line.rstrip("\r\n") for line in f]
        self._token_to_id = {tok: i for i, tok in enumerate(self.vocabulary)}
        self.oov_id = self._token_to_id[oov_token]
        self.suffix = suffix_indicator

    # --- to samo API, którego używa notebook ---
    def vocabulary_size(self):
        return len(self.vocabulary)

    def get_vocabulary(self):
        return self.vocabulary

    def token_to_id(self, token):
        return self._token_to_id[token]

    def id_to_token(self, idx):
        return self.vocabulary[idx]

    def __call__(self, text):
        return self.tokenize(text)

    def tokenize(self, text):
        """Tekst → tablica ID tokenów (np.int32), jak tokenizer(text) w keras-hub."""
        ids = []
        for word in self._split(text):
            ids.extend(self._wordpiece(_case_fold(word)))
        return np.array(ids, dtype=np.int32)

    def detokenize(self, ids):
        """ID tokenów → tekst. Przyjmuje listę, np.array albo tf.Tensor."""
        words = []
        for idx in np.asarray(ids).reshape(-1):
            tok = self.vocabulary[int(idx)]
            if tok.startswith(self.suffix) and words:
                words[-1] += tok[len(self.suffix):]  # "##ing" doklejamy do poprzedniego
            else:
                words.append(tok)
        return " ".join(words)

    # --- krok 1: podział na słowa i interpunkcję ---
    def _split(self, text):
        words, current = [], []
        for ch in text:
            if _is_dropped_separator(ch) or _is_kept_separator(ch):
                if current:
                    words.append("".join(current))
                    current = []
                if _is_kept_separator(ch):
                    words.append(ch)
            else:
                current.append(ch)
        if current:
            words.append("".join(current))
        return words

    # --- krok 2: WordPiece, czyli najdłuższe pasujące kawałki od lewej ---
    def _wordpiece(self, word):
        if len(word.encode("utf-8")) > MAX_BYTES_PER_WORD:
            return [self.oov_id]
        ids, start = [], 0
        while start < len(word):
            end, match = len(word), None
            while start < end:
                piece = word[start:end] if start == 0 else self.suffix + word[start:end]
                if piece in self._token_to_id:
                    match = self._token_to_id[piece]
                    break
                end -= 1
            if match is None:
                return [self.oov_id]  # nie da się złożyć słowa → całe słowo to [UNK]
            ids.append(match)
            start = end
        return ids