import torch
import numpy as np
import nltk


def split_sentence(sentence, use_sp=False):
    return nltk.word_tokenize(sentence)


class BBPETokenizerPPLCalc(object):
    """ base_tokenizer is based on the 'BBPE Algorithm' """

    def __init__(self, byte_encoder, base_model, base_tokenizer, device):
        self.byte_encoder = byte_encoder
        self.byte_decoder = {v: k for k, v in byte_encoder.items()}
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device

    def get_bbpe_bytes(self, words):
        bbs = []  # bbpe_bytes
        bbs_to_words = []
        for idx, word in enumerate(words):
            byte_list = [self.byte_encoder[b] for b in word.encode("utf-8")]
            bbs.extend(byte_list)
            bbs_to_words.extend([idx for i in range(len(byte_list))])
        return bbs, bbs_to_words

    def calc_sent_ppl(self, outputs, labels):
        """
        :param outputs: language model's output.
        :param labels: token ids.
        :return: sentence ppl, list of subtoken ppl.
        """
        lm_logits = outputs.logits.squeeze()  # seq-len, V
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        ll = loss_func(shift_logits, shift_labels.view(-1))
        loss = ll.mean().item()
        ll = ll.tolist()
        return loss, ll

    def get_bbs_ll(self, input_ids, ll):
        """
        :return bbs_ll: list of bbpe_byte's ll.
        """
        input_ids = input_ids.squeeze()
        tokenized_tokens = [
            self.base_tokenizer._convert_id_to_token(input_id)
            for input_id in input_ids
        ]
        bbs_ll = []
        # because GPT2 tokenizer don't include <s> before the sub_tokens,
        # so the first sub_token's ll cannot be obtained.
        byte_list = [self.byte_decoder[c] for c in tokenized_tokens[0]]
        bbs_ll.extend([0 for i in range(len(byte_list))])
        for idx, token in enumerate(tokenized_tokens[1:]):
            byte_list = [self.byte_decoder[c] for c in token]
            bbs_ll.extend(ll[idx] for i in range(len(byte_list)))
        return bbs_ll

    def calc_token_ppl(self, bbs_to_words, bbs_ll):
        """
        :param bbs_to_words: list of bytes_to_words index.
        :param bbs_ll: list of bytes ppl.
        :return: list of token ppl.
        """
        start = 0
        ll_tokens = []
        while start < len(bbs_to_words) and start < len(bbs_ll):
            end = start + 1
            while end < len(
                    bbs_to_words) and bbs_to_words[end] == bbs_to_words[start]:
                end += 1
            if end > len(bbs_ll):
                break
            ll_token = bbs_ll[start:end]
            ll_tokens.append(np.mean(ll_token))
            start = end
        return ll_tokens

    def get_begin_word_idx(self, input_ids, bbs_to_words):
        input_ids = input_ids.squeeze()
        begin_token = self.base_tokenizer._convert_id_to_token(input_ids[0])
        byte_list = [self.byte_decoder[c] for c in begin_token]
        begin_word_idx = bbs_to_words[len(byte_list) - 1] + 1
        return begin_word_idx

    def forward_calc_ppl(self, text):
        tokenized = self.base_tokenizer(text,
                                        return_tensors="pt").to(self.device)
        input_ids = tokenized.input_ids
        labels = tokenized.input_ids
        input_ids = input_ids[:, :1024, ]
        labels = labels[:, :1024, ]
        words = split_sentence(text)
        bbs, bbs_to_words = self.get_bbpe_bytes(words)

        outputs = self.base_model(input_ids=input_ids, labels=labels)
        loss, ll = self.calc_sent_ppl(outputs, labels)
        bbs_ll = self.get_bbs_ll(input_ids, ll)
        ll_tokens = self.calc_token_ppl(bbs_to_words, bbs_ll)
        begin_word_idx = self.get_begin_word_idx(input_ids, bbs_to_words)
        return [loss, begin_word_idx, ll_tokens]


class SPLlamaTokenizerPPLCalc(object):
    """ base_tokenizer is based on the `SentencePiece Algorithm` for Llama models """

    def __init__(self, base_model, base_tokenizer, device):
        # Llama tokenizer has byte level tokens for words which is not in the `tokenizer vocab`
        self.byte_encoder = {i: f'<0x{i:02X}>' for i in range(256)}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device

    def get_sp_bytes(self, words):
        bbs = []  # bytes
        bbs_to_words = []
        for idx, word in enumerate(words):
            byte_list = [self.byte_encoder[b] for b in word.encode("utf-8")]
            bbs.extend(byte_list)
            bbs_to_words.extend([idx for i in range(len(byte_list))])
        return bbs, bbs_to_words

    def calc_sent_ppl(self, outputs, labels):
        """
        :param outputs: language model's output.
        :param labels: token ids.
        :return: sentence ppl, list of subtoken ppl.
        """
        lm_logits = outputs.logits.squeeze()  # seq-len, V
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        ll = loss_func(shift_logits, shift_labels.view(-1))
        loss = ll.mean().item()
        ll = ll.tolist()
        return loss, ll

    def get_bbs_ll(self, input_ids, ll):
        """
        :return bbs_ll: list of bbpe_byte's ll.
        """
        input_ids = input_ids.squeeze()
        # because `sentencepiece tokenizer` add `<s>▁` before all sentence.
        # this step we remove `<s>`, because it is treated as a separate token.
        input_ids = input_ids[1:]
        tokenized_tokens = self.base_tokenizer.convert_ids_to_tokens(input_ids)
        bbs_ll = []
        for idx, token in enumerate(tokenized_tokens):
            if self.base_tokenizer.sp_model.IsByte(input_ids[idx].item()):
                byte_list = [token]
            else:
                byte_list = [
                    self.byte_encoder[b] for b in token.encode("utf-8")
                ]
            bbs_ll.extend(ll[idx] for i in range(len(byte_list)))
        # because `sentencepiece tokenizer` add `<s>▁` before all sentence.
        # this step we remove `▁`, because it is treated as the first token or part of the first token which corresponds to the first logit.
        bbs_ll = bbs_ll[len('▁'.encode("utf-8")):]
        return bbs_ll

    def calc_token_ppl(self, bbs_to_words, bbs_ll):
        """
        :param ll: list of bytes_to_words index.
        :param ll: list of bytes ppl.
        :return: list of token ppl.
        """
        start = 0
        ll_tokens = []
        while start < len(bbs_to_words) and start < len(bbs_ll):
            end = start + 1
            while end < len(
                    bbs_to_words) and bbs_to_words[end] == bbs_to_words[start]:
                end += 1
            if end > len(bbs_ll):
                break
            ll_token = bbs_ll[start:end]
            ll_tokens.append(np.mean(ll_token))
            start = end
        return ll_tokens

    def forward_calc_ppl(self, text):
        tokenized = self.base_tokenizer(text,
                                        max_length=1024,
                                        truncation=True,
                                        return_tensors="pt").to(self.device)
        input_ids = tokenized.input_ids
        labels = tokenized.input_ids
        input_ids = input_ids[:, :1024, ]
        labels = labels[:, :1024, ]
        words = split_sentence(text, use_sp=True)
        bbs, bbs_to_words = self.get_sp_bytes(words)

        # Here we don't pass the labels because the output_logits and labels may not on the same device is you not carefully set it.
        outputs = self.base_model(input_ids=input_ids)
        loss, ll = self.calc_sent_ppl(outputs, labels)
        bbs_ll = self.get_bbs_ll(input_ids, ll)
        ll_tokens = self.calc_token_ppl(bbs_to_words, bbs_ll)
        # ll_tokens has removed `<s>_`, the first element is the logit of the first word
        begin_word_idx = 0
        return [loss, begin_word_idx, ll_tokens]
