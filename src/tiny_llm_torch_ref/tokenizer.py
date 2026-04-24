from transformers import AutoTokenizer


class StreamingDetokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.offset = 0
        self.tokens = []
        self._text = ""

    def add_token(self, token: int):
        """
        Input:
        - `token`: scalar token id
        """
        self.tokens.append(token)
        self._text = self._tokenizer.decode(
            self.tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def finalize(self):
        pass

    @property
    def text(self):
        return self._text

    @property
    def last_segment(self):
        """
        Returns:
        - incremental decoded text segment
        """
        segment = self._text[self.offset :]
        self.offset = len(self._text)
        return segment


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._detokenizer = StreamingDetokenizer(tokenizer)

    def __getattr__(self, attr):
        if attr == "detokenizer":
            return self._detokenizer
        return getattr(self._tokenizer, attr)

    def apply_chat_template(self, *args, **kwargs):
        filtered_kwargs = dict(kwargs)
        filtered_kwargs.pop("enable_thinking", None)
        return self._tokenizer.apply_chat_template(*args, **filtered_kwargs)


def load_tokenizer(model_name: str) -> TokenizerWrapper:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    return TokenizerWrapper(tokenizer)
