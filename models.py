from utils import *
from preprocess import *
from PIL import Image
import torch
import copy

### Based on MTV (Multi-modal Task Vectors)
class ModelHelper:
    def __init__(self):

        """
        self.model:The loaded model
        self.tokenizer: The loaded tokenizer
        self.model_config: The architecture of the model. Might need to do print(model) see how to initialize
        self.format_func: The format function for the current dataset
        self.space: Whether the model output will have a leading space
        self.cur_dataset: Name of the current dataset
        self.split_idx: The index of "layer" when you parse "attn_hook_names" with "."
        self.nonspecial_idx: The index in which the generated tokens are not special token. Used to skip special token and construct the current target output for loss calculation.
        """


    #Always return a single variable. If both text and image is returned, return in tuple
    def insert_image(self, text, image_list):

        """
        Returns an object that is the input to forward and generate.
        """
        pass
    #Takes the output of insert_image
    def forward(self, model_input, labels=None):

        """
        Forwrad function wrapper
        """

        pass
    #Takes the output of insert image
    def generate(self, model_input, max_new_tokens):

        """
        Generate function wrapper
        """
        pass


class LlamaHelper(ModelHelper):

    def __init__(self, model, tokenizer, cur_dataset, lr):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = {"n_heads":model.config.num_attention_heads,
                    "n_layers":model.config.num_hidden_layers,
                    "resid_dim":model.config.hidden_size,
                    "name_or_path":model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                    "prepend_bos":True}
        self.format_func = get_format_func(cur_dataset)
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.question_lookup = None
        self.alpha_shape = (self.model_config["n_layers"], self.model_config["n_heads"])
        init_logit = 0.0 
        self.alphas_logit = torch.nn.Parameter(torch.full(self.alpha_shape, init_logit, device='cuda')) 
        self.optimizer = torch.optim.Adam([self.alphas_logit], lr=lr)
        self.tokenizer.pad_token = tokenizer.eos_token

        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, model_input, labels=None):

        result = self.model(input_ids=model_input["input_ids"].to(self.model.device),
                attention_mask=model_input["attention_mask"].to(self.model.device)) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    
    def generate(self, model_input, max_new_tokens):
        generated_output = self.model.generate(
            input_ids=model_input["input_ids"].to(self.model.device),
            attention_mask=model_input["attention_mask"].to(self.model.device),
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            use_cache=True, # Must be True for generation mode
            top_k=1,
        )
        return self.tokenizer.batch_decode(generated_output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()


class QwenHelper(ModelHelper):

    def __init__(self, model, tokenizer, cur_dataset, lr):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = {"n_heads":model.config.num_attention_heads,
                    "n_layers":model.config.num_hidden_layers,
                    "resid_dim":model.config.hidden_size,
                    "head_dim": model.config.head_dim,
                    "name_or_path":model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                    "prepend_bos":True}
        self.format_func = get_format_func(cur_dataset)
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.question_lookup = None
        self.alpha_shape = (self.model_config["n_layers"], self.model_config["n_heads"])
        init_logit = 0.0 
        self.alphas_logit = torch.nn.Parameter(torch.full(self.alpha_shape, init_logit, device='cuda')) 
        self.optimizer = torch.optim.Adam([self.alphas_logit], lr=lr)
        self.tokenizer.pad_token = tokenizer.eos_token

        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, model_input, labels=None):

        result = self.model(input_ids=model_input["input_ids"].to(self.model.device),
                attention_mask=model_input["attention_mask"].to(self.model.device)) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    
    def generate(self, model_input, max_new_tokens):
        generated_output = self.model.generate(
            input_ids=model_input["input_ids"].to(self.model.device),
            attention_mask=model_input["attention_mask"].to(self.model.device),
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            use_cache=True,
            top_k=1,
        )
        return self.tokenizer.batch_decode(generated_output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()


class MistralHelper(ModelHelper):

    def __init__(self, model, tokenizer, cur_dataset, lr):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = {"n_heads":model.config.num_attention_heads,
                    "n_layers":model.config.num_hidden_layers,
                    "resid_dim":model.config.hidden_size,
                    "name_or_path":model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                    "prepend_bos":True}
        self.format_func = get_format_func(cur_dataset)
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.question_lookup = None
        self.alpha_shape = (self.model_config["n_layers"], self.model_config["n_heads"])
        init_logit = 0.0 
        self.alphas_logit = torch.nn.Parameter(torch.full(self.alpha_shape, init_logit, device='cuda')) 
        self.optimizer = torch.optim.Adam([self.alphas_logit], lr=lr)
        self.tokenizer.pad_token = tokenizer.eos_token

        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, model_input, labels=None):

        result = self.model(input_ids=model_input["input_ids"].to(self.model.device),
                attention_mask=model_input["attention_mask"].to(self.model.device)) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    
    def generate(self, model_input, max_new_tokens):
        generated_output = self.model.generate(
            input_ids=model_input["input_ids"].to(self.model.device),
            attention_mask=model_input["attention_mask"].to(self.model.device),
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            use_cache=True,
            top_k=1,
        )
        return self.tokenizer.batch_decode(generated_output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()


class Gemma3Helper(ModelHelper):

    def __init__(self, model, tokenizer, cur_dataset, lr):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = {"n_heads":model.config.num_attention_heads,
                    "n_layers":model.config.num_hidden_layers,
                    "resid_dim":model.config.hidden_size,
                    "head_dim": model.config.head_dim,
                    "name_or_path":model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
                    "prepend_bos":True}
        self.format_func = get_format_func(cur_dataset)
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.question_lookup = None
        self.alpha_shape = (self.model_config["n_layers"], self.model_config["n_heads"])
        init_logit = 0.0 
        self.alphas_logit = torch.nn.Parameter(torch.full(self.alpha_shape, init_logit, device='cuda')) 
        self.optimizer = torch.optim.Adam([self.alphas_logit], lr=lr)
        self.tokenizer.pad_token = tokenizer.eos_token

        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, model_input, labels=None):

        result = self.model(input_ids=model_input["input_ids"].to(self.model.device),
                attention_mask=model_input["attention_mask"].to(self.model.device)) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    
    def generate(self, model_input, max_new_tokens):
        generated_output = self.model.generate(
            input_ids=model_input["input_ids"].to(self.model.device),
            attention_mask=model_input["attention_mask"].to(self.model.device),
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            use_cache=True,
            top_k=1,
        )
        return self.tokenizer.batch_decode(generated_output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()
