import torch
import transformers

from backend_utils import BBPETokenizerPPLCalc, SPLlamaTokenizerPPLCalc
from backend_utils import split_sentence
# mosec
from mosec import Worker
from mosec.mixin import MsgpackMixin
# llama
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


class SnifferBaseModel(MsgpackMixin, Worker):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = None
        self.base_model = None
        self.generate_len = 512

    def forward_calc_ppl(self):
        pass

    def forward_gen(self):
        self.base_tokenizer.padding_side = 'left'
        # 1. single generate
        if isinstance(self.text, str):
            tokenized = self.base_tokenizer(self.text, return_tensors="pt").to(
                self.device)
            tokenized = tokenized.input_ids
            gen_tokens = self.base_model.generate(tokenized,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_tokens = gen_tokens.squeeze()
            result = self.base_tokenizer.decode(gen_tokens.tolist())
            return result
        # 2. batch generate
        # msgpack.unpackb(self.text, use_list=False) == tuple
        elif isinstance(self.text, tuple):
            inputs = self.base_tokenizer(self.text,
                                         padding=True,
                                         return_tensors="pt").to(self.device)
            gen_tokens = self.base_model.generate(**inputs,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_texts = self.base_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            return gen_texts

    def forward(self, data):
        """
        :param data: ['text': str, "do_generate": bool]
        :return:
        """
        self.text = data["text"]
        self.do_generate = data["do_generate"]
        if self.do_generate:
            return self.forward_gen()
        else:
            return self.forward_calc_ppl()


class SnifferGPT2Model(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'gpt2-xl')
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'gpt2-xl')
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model.to(self.device)
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SnifferGPTJModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'EleutherAI/gpt-j-6b')
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'EleutherAI/gpt-j-6b', device_map="auto", load_in_8bit=True)
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder,
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SnifferLlama2Model(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        model_path = 'meta-llama/Llama-2-13b-chat-hf'
        self.base_tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.base_model = LlamaForCausalLM.from_pretrained(model_path,
                                                           use_safetensors=False,
                                                           device_map="auto",
                                                           load_in_8bit=True)
        self.ppl_calculator = SPLlamaTokenizerPPLCalc(self.base_model,
                                                      self.base_tokenizer,
                                                      self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SnifferAlpacaModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        model_path = '/path/to/alpaca-7b'
        self.base_tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.base_model = LlamaForCausalLM.from_pretrained(model_path,
                                                           device_map="auto",
                                                           load_in_8bit=True)
        self.ppl_calculator = SPLlamaTokenizerPPLCalc(self.base_model,
                                                      self.base_tokenizer,
                                                      self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SnifferVicunaModel(SnifferBaseModel):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        model_path = 'lmsys/vicuna-13b-v1.5'
        self.base_tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.base_model = LlamaForCausalLM.from_pretrained(model_path,
                                                           device_map="auto",
                                                           load_in_8bit=True)
        self.ppl_calculator = SPLlamaTokenizerPPLCalc(self.base_model,
                                                      self.base_tokenizer,
                                                      self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text)
