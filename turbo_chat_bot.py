import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from config import Config
from custom_decorators import log_call, timeit
from custom_logger import CustomLogger
from model_stuff import Transformer
from tensorflow import Tensor
from wordpiece_local import LocalWordPieceTokenizer

warnings.filterwarnings("ignore", message="You are using a softmax over axis")

logger = CustomLogger(
    log_file_name=Config.LOGS_PATH,
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL
).create_logger()

class FailedToLoadParams(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class FailedToLoadModel(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

@dataclass
class ChatBot:
    model_path: Path = Config.MODEL_FOLDER / Config.MODEL / "best_turowy_v2.weights.h5"
    vocab_path: Path = Config.MODEL_FOLDER / Config.MODEL / "vocab_turowy_v2.txt"
    params_path: Path = Config.MODEL_FOLDER / Config.MODEL / "params_turowy_v2.json"

    @timeit(logger=logger)
    def __post_init__(self) -> None:
        logger.info("Initing with:\n" \
        f" - {self.model_path=}\n" \
        f" - {self.vocab_path=}\n" \
        f" - {self.params_path=}\n" \
        )
        try:
            with open(self.params_path) as f:
                model_params = json.load(f)
        except (FileExistsError, FileNotFoundError):
            logger.exception(f"Could not load model params: {self.params_path=}")
            raise FailedToLoadParams(f"Could not load model params: {self.params_path=}")

        self.max_len = model_params["max_len"]
        self.pad = model_params["pad"]
        self.start = model_params["start"]
        self.end = model_params["end"]
        self.num_layers = model_params["num_layers"]
        self.d_model = model_params["d_model"]
        self.num_heads = model_params["num_heads"]
        self.dff = model_params["dff"]
        self.dropout = model_params["dropout"]
        self.sep = model_params["sep"]
        self.context_turns = model_params["context_turns"]
        self.max_src_len = model_params["max_src_len"]

        logger.info("Loaded model params:\n" \
        f" - {self.max_len=}\n" \
        f" - {self.pad=}\n" \
        f" - {self.start=}\n" \
        f" - {self.end=}\n" \
        f" - {self.num_layers=}\n" \
        f" - {self.d_model=}\n" \
        f" - {self.num_heads=}\n" \
        f" - {self.dff=}\n" \
        f" - {self.dropout=}\n" \
        f" - {self.sep=}\n" \
        f" - {self.context_turns=}\n" \
        f" - {self.max_src_len=}\n" \
        )

        try:
            self.tokenizer = LocalWordPieceTokenizer(self.vocab_path)
            vocab = self.tokenizer.get_vocabulary()
            self.model = Transformer(
                num_layers=self.num_layers,
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                vocab_size=len(vocab),
                dropout=self.dropout,
                pad=self.pad
            )
            dummy_enc = tf.zeros((1, self.max_len), tf.int32); dummy_dec = tf.zeros((1, self.max_len), tf.int32)
            _ = self.model((dummy_enc, dummy_dec))

            self.model.load_weights(self.model_path)
            self.history = []
        except Exception as e:
            logger.exception("Could not load model")
            raise FailedToLoadModel(f"Could not load model: {e}") from e

    def reset(self) -> None:
        self.history = []

    def _encode_context(self) -> np.ndarray:
        ids = []
        for turn in self.history[-self.context_turns:]:
            ids += [int(i) for i in np.asarray(self.tokenizer(turn))] + [self.sep]
        return np.array(ids[:-1][-self.max_src_len:], dtype=np.int32)

    @staticmethod
    def to_text(x: str | Tensor) -> str:
        """The Detokenizer in newer versions of keras_hub returns a str,
            while in older versions it returns a tf.Tensor containing bytes—handle both"""
        x = x.numpy() if hasattr(x, "numpy") else x
        return x.decode("utf-8") if isinstance(x, bytes) else str(x)

    @staticmethod
    def clean(t: str) -> str:
        t = str(t).strip()
        t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"(\w)\s*'\s*(\w)", r"\1'\2", t)   # kontrakcje: it ' s -> it's
        t = re.sub(r"\s+([.,!?;:'])", r"\1", t)          # dociągnij interpunkcję
        return t.strip()

    # @log_call(logger=logger, log_params=["temperature", "top_k", "greedy"], hide_res=True)
    # @timeit(logger=logger)
    def respond(self, text: str, temperature: float = 0.7, top_k: int = 20, greedy: bool = False) -> str:
        text = self.clean(text)
        if not text:
            return ""
        self.history.append(text)
        tokens = self._encode_context()
        enc_in = tf.constant(tokens[np.newaxis], dtype=tf.int32)
        enc_mask = tf.not_equal(enc_in, self.pad)

        """
        tf.Tensor([[285  15  97  85  71  32]], shape=(1, 6), dtype=int32) encoded text
        tf.Tensor([[ True  True  True  True  True  True]], shape=(1, 6), dtype=bool) skipping pad token, in this case, there is no pad token
        """

        # The encoder reads the question ONCE
        # Its output (context) does not change during generation.
        # training=False → dropout is disabled; the model operates deterministically
        context = self.model.encoder(enc_in, enc_mask, training=False)   # (1, T, 256)

        # Loop: one new token per turn 
        output = tf.constant([[self.start]], dtype=tf.int32)             # (1, 1) answer starts with [START]
        for _ in range(self.max_len):
            # The decoder examines the response generated so far and (via cross-attention)
            # the context of the question. Each time, it computes the entire response from scratch (without caching):
            # it’s simpler, and with 40 tokens, it doesn’t matter.
            x = self.model.decoder(output, context, enc_mask, training=False)  # (1, t, 256)

            # There are logits for each position, but we only need the last one:
            # this is the prediction for the NEXT token (like label[t] during training)
            logits = self.model.final_layer(x)[:, -1, :] # (1, V) score for each token in the dictionary

            # Selecting the next token
            if greedy:
                # Always the most likely → predictable, but boring and repetitive answers
                next_id = tf.argmax(logits, axis=-1, output_type=tf.int32)[:, tf.newaxis]   # (1, 1)
            else:
                # Temperature: < 1 makes the distribution more peaked (more predictable), > 1 flattens it (more random)
                logits = logits / temperature
                if top_k:
                    # Top-k: keep only the k best candidates; assign -∞ to the rest,
                    # i.e., a probability of 0 after softmax
                    values, _ = tf.math.top_k(logits, k=top_k) # the highest logits, in descending order
                    min_val = values[:, -1:]
                    neg_inf = tf.constant(-1e9, dtype=logits.dtype)
                    logits = tf.where(logits < min_val,
                                      tf.fill(tf.shape(logits), neg_inf), logits)
                # Sampling from softmax(logits); for categorical data, softmax is applied automatically
                next_id = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)   # (1, 1)

            # We attach the selected token, and the decoder will recognize it on the next turn
            output = tf.concat([output, next_id], axis=-1)          # (1, t+1)

            # The model decides when to stop: it outputs [END] → stop
            # If it never outputs [END], the loop ends after max_len tokens
            if int(next_id[0, 0]) == self.end:
                break

        # Tokens → text
        # output[0, 1:] omits [START]; the filter removes [END] and any [PAD]
        ids = [int(i) for i in output[0, 1:].numpy() if int(i) not in (self.end, self.pad)]
        if not ids:
            return ""
        reply = self.clean(self.to_text(self.tokenizer.detokenize(tf.constant(ids, dtype=tf.int32))))
        if reply:
            self.history.append(reply)    # answer from the bot is also context, so we add it
        return reply.capitalize()



if __name__ == "__main__":
    chat_bot = ChatBot()

    while True:
        msg = input("You: ")

        if msg.strip().lower() in {"quit", "exit", ""}:
            print("bye sar")
            break

        print("Bot:", chat_bot.respond(msg, temperature=0.7, top_k=20))

    # questions = [
    #     # "Hello, how are you?",
    #     # "What do you do for work?",
    #     # "I am so tired today.",
    #     # "Do you like music?",
    #     # "What are your plans for the weekend?",
    #     "Did Hitler know about death camps?",
    #     "Do you know Stalin? If yes, then tell me something about him.",
    #     "What are you plan for today?",
    #     "What do you do for work?",
    #     "Is Putri a brat?",
    #     "Did Hitler escape to Argentina?"
    # ]
    # print()
    # for q in questions:
    #     print("You:", q)
    #     chat_bot.reset()
    #     # print(chat_bot.history)
    #     print("Bot:", chat_bot.respond(q, temperature=0.7, top_k=20))
    #     print()