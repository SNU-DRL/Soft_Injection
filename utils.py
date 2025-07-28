
from baukit import TraceDict, get_module
from models import *
from preprocess import *
import sys
import torch
from tqdm import tqdm
import bitsandbytes as bnb

from transformers import AutoModelForCausalLM, AutoTokenizer, logging, LlamaForCausalLM, BitsAndBytesConfig, Qwen3ForCausalLM, MistralForCausalLM, Gemma3ForCausalLM
import sys

logging.set_verbosity_warning()

torch.autograd.set_detect_anomaly(True)

sys.path.append('../eval_mm')

def compute_individual_token_match(prob_dist, target_id) -> int:
    """
    Individual computation of token ranks across a single distribution.

    Parameters:
    prob_dist: the distribution of scores for a single output
    target_id: the target id we care about

    Return:
    A single value 1 if the highest scoring token in prob_dis is the target_id, 0 otherwise.
    """
    if isinstance(target_id, list):
        target_id = target_id[0]

    if torch.where(torch.argsort(prob_dist.squeeze(), descending=True) == target_id)[0].item() == 0:
        score = 1
    else:
        score = 0
    return score


def load_model(model_name, cur_dataset, lr):

    """
    A function that loads the model and a corresponding model_helper. Refer to model.py for more detail.

    Parameters:
    model_name: The name of the model you are attempting to load
    cur_dataset: The name of dataset you are attempting to load

    Returns: 
    model_helper: A helper class that contains the model as well as other functionality.
    """

    if model_name.lower() == "meta-llama/Llama-3.1-8b".lower() or model_name.lower() == "meta-llama/Llama-3.1-8B-Instruct".lower():
        model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_helper = LlamaHelper(model, tokenizer, cur_dataset, lr)
    
    if model_name.lower() == "meta-llama/Llama-3.1-70b".lower() or model_name.lower() == "meta-llama/Llama-3.1-70B-Instruct".lower():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = LlamaForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_helper = LlamaHelper(model, tokenizer, cur_dataset, lr)

    
    if model_name.lower() == "Qwen/Qwen3-8B".lower():
        model = Qwen3ForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_helper = QwenHelper(model, tokenizer, cur_dataset, lr)
    
    if model_name.lower() == "Qwen/Qwen3-32B".lower():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = Qwen3ForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_helper = QwenHelper(model, tokenizer, cur_dataset, lr)
    
    if model_name.lower() == "mistralai/Mistral-7B-v0.3".lower() or model_name.lower() == "mistralai/Mistral-7B-Instruct-v0.3".lower():
        model = MistralForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_helper = MistralHelper(model, tokenizer, cur_dataset, lr)
    
    if model_name.lower() == "mistralai/Mixtral-8x7B-v0.1".lower() or model_name.lower() == "mistralai/Mixtral-8x7B-Instruct-v0.1".lower():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_helper = MistralHelper(model, tokenizer, cur_dataset, lr)

    if model_name.lower() == "google/gemma-3-4b-pt".lower() or model_name.lower() == "google/gemma-3-4b-it".lower():
        model = Gemma3ForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_helper = Gemma3Helper(model, tokenizer, cur_dataset, lr)

    return model_helper


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/308e9d174cf0a1cf910b891d340f0dfd14168668/src/utils/extract_utils.py#L15
def gather_last_attn_activations(inputs, model_helper):

    """
    A function that performs a forward pass and extract the activation at certain location of the layer.

    Parameters:
    inputs: input to the model. Created with model_helper
    model_helper

    Returns: 
    td: The attention activations.
    result: The output logits from forward method.
    """

    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], retain_input=True, retain_output=True) as td:                
        result = model_helper.forward(inputs)
    return td, result


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/308e9d174cf0a1cf910b891d340f0dfd14168668/src/utils/extract_utils.py#L65
def split_activations_by_head(activations, model_config):

    """
    The model concatenate the output of multi-headed attention to a single vector. This function splits this vector back to different heads.
    From

    Parameters:
    activations: From gather_last_attn_activations
    model_config: Refer to model.py

    Returns: 
    the activation partitioned by attention heads
    """

    new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
    if 'gemma-3' in model_config["name_or_path"].lower() or 'qwen3-32b' in model_config["name_or_path"].lower():
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['head_dim']) # split by head: + (n_attn_heads, head_dim)    
    activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
    return activations.to("cuda")


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/308e9d174cf0a1cf910b891d340f0dfd14168668/src/utils/extract_utils.py#L46
def get_last_mean_head_activations(dataset, model_helper, N_TRIALS = 50, shot=4, no_mean=False):

    """
    This function extracts the activation of the last input token.

    Parameters:
    dataset: a iterable item suitable for model_helper.format_func. Essentially a dataloader.
    model_helper:
    N_TRIALS: How many example to average the activation over
    shot: Number of shots per example
    no_mean: Whether you want to take the mean of the examples or save it for other preprocess

    Returns: 
    mean_activations: It has the dimension of (layer, head, Token_len, residual_dim) or (N_TRIALS, layer, head, Token_len, residual_dim). Token_len is set to 1 in this case.
    """

    activation_storage = None

    for n in tqdm(range(N_TRIALS)):

        inputs, _, _, _, _ = model_helper.format_func(dataset, None, num_shot=shot, model_helper=model_helper)
        activations_td, result= gather_last_attn_activations(inputs, model_helper)


        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_helper.model_config) for layer in model_helper.model_config['attn_hook_names']]).permute(0,2,1,3)
        ###Extracting only the activation of the last input_token, as seen in the -1 indexing
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)
        if activation_storage is None:
            activation_storage = cur_activation
        else:

            activation_storage = torch.vstack((activation_storage, cur_activation))
    if no_mean:
        return activation_storage
    
    mean_activations = activation_storage.mean(dim=0)
    
    return mean_activations


def optimize_head_selection(mean_activations, model_helper, gradient_opt_data, eval_data, epoch, batch_size, test_batch_size, dataset_names):

    """
    This function performs gradient descent to optimize soft head-selection parameteres.

    Returns: 
    Optimized soft head-selection parameters
    """
    eps = 1e-3

    best_val_loss = float('inf')
    best_alphas_logit = None

    with torch.set_grad_enabled(True):
        for epoch in tqdm(range(epoch)):
            model_helper.optimizer.zero_grad()
            
            new_input, _, target_outs, dataset_idx, _ = model_helper.format_func(gradient_opt_data, None, num_shot=0, model_helper=model_helper, batch_size=batch_size, dataset_names=dataset_names)
            attention_mask = new_input["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1
            last_token_indices = last_token_indices.tolist()

            target_tokens = torch.tensor([target_out[0] for target_out in target_outs]).to('cuda')
            batched_mean_activations = torch.stack([mean_activations[dataset_idx]] * len(target_outs), dim=0)

            # Compute alphas (soft head-selection parameters) from logit
            alphas = torch.sigmoid(model_helper.alphas_logit) 

            # Forward pass with modified attention heads
            out_logit = gd_activation_replacement(new_input, batched_mean_activations, model_helper, alphas, last_token_only=True, batch_size=len(target_outs), last_token_indices=last_token_indices)

            # Loss calculation
            task_loss = torch.nn.functional.cross_entropy(out_logit, target_tokens)
            
            # Backward and update
            task_loss.backward()
            model_helper.optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch} | Loss: {task_loss.item():.4f} | Mean α: {alphas.mean().item():.4f}")

            torch.cuda.empty_cache()
            if epoch % 50 == 49:  
                val_loss = validate_head_selection(model_helper, torch.sigmoid(model_helper.alphas_logit), eps, mean_activations, eval_data, epoch, batch_size=test_batch_size)
                if val_loss < best_val_loss:
                    best_alphas_logit = model_helper.alphas_logit.detach().clone()

    if best_alphas_logit is not None:
        model_helper.alphas_logit.data.copy_(best_alphas_logit)

    return torch.sigmoid(model_helper.alphas_logit)


def validate_head_selection(model_helper, alphas, eps, mean_activations, eval_data, epoch, batch_size=1):

    with torch.no_grad():
        loss_list = []
        for dataset_idx, eval_task_data in enumerate(eval_data): # added
            dataset_loss = []
            for batch_start in range(0, len(eval_task_data), batch_size):
                batch_end = min(batch_start + batch_size, len(eval_task_data))
                current_batch_size = batch_end - batch_start
                items = [eval_task_data[i] for i in range(batch_start, batch_end)]
                new_input, _, target_outs, _, _ = model_helper.format_func(None, items, num_shot=0, model_helper=model_helper, batch_size=current_batch_size)
                attention_mask = new_input["attention_mask"]
                last_token_indices = attention_mask.sum(dim=1) - 1
                last_token_indices = last_token_indices.tolist()

                target_tokens = torch.tensor([target_out[0] for target_out in target_outs]).to('cuda')
                batched_mean_activations = torch.stack([mean_activations[dataset_idx]] * current_batch_size, dim=0)
                out_logit = gd_activation_replacement(new_input, batched_mean_activations, model_helper, alphas, last_token_only=True, batch_size=current_batch_size, last_token_indices=last_token_indices)
                task_loss = torch.nn.functional.cross_entropy(out_logit, target_tokens)

                dataset_loss.append(task_loss)
            loss_list.append(torch.tensor(dataset_loss).mean().item())

        print(f"validation loss at {epoch} epoch:", torch.tensor(loss_list).mean())
    return torch.tensor(loss_list).mean().item()




def gd_activation_replacement(model_input, avg_activations, model_helper, alphas, last_token_only=True, gt=None, intervention_token=None, batch_size=None, last_token_indices=None):

    intervention_fn = last_replace_activation_w_avg(alphas=alphas, avg_activations=avg_activations, 
                                                model=model_helper.model, model_config=model_helper.model_config,
                                                batched_input=False, last_token_only=last_token_only, split_idx=model_helper.split_idx, intervention_token=intervention_token,
                                                batch_size=batch_size, last_token_indices=last_token_indices)

    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], edit_output=intervention_fn, retain_grad=True) as td: 
        if gt is None:               
            output = model_helper.forward(model_input, labels=gt).logits[torch.arange(batch_size), last_token_indices,:] # batch_size x n_tokens x vocab_size, only want last token prediction
        else:
            output = model_helper.forward(model_input, labels=gt).loss # deprecated, will be removed in future versions

    return output


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/874d6e93c099d71fe4a2d76551fab233e60062c2/src/utils/intervention_utils.py#L16
def last_replace_activation_w_avg(alphas, avg_activations, model, model_config, batched_input=False, last_token_only=False, patching=False, replace_layer = 0, split_idx=2, intervention_token=None,
                                  batch_size=None, last_token_indices=None, exact_match=False):

    """
    This function performs intervention on during generation.
    avg_activations.shape = (batch_size, n_layers, n_heads, 1, head_dim)

    This function defaults to perform intervention during the full generation. To perform intervention on certain token/generation step, modify the function accordingly.
    """

    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split('.')[split_idx])
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        intervention_embedding = avg_activations[:,current_layer,:,0].unsqueeze(1).to(device=inputs.device) # (batch_size, 1, n_heads, head_dim) # modified from the above

        # Determine shapes for intervention
        original_shape = inputs.shape
        new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        if 'gemma-3' in model_config["name_or_path"].lower() or 'qwen3-32b' in model_config["name_or_path"].lower():
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['head_dim'])
        inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), n_heads, head_dim)

        modified_inputs = inputs.clone()

        gated_value = alphas[current_layer].reshape(1, 1, -1, 1).to(inputs.device)
        
        if last_token_only: # inject only at the last token of the intial prompt (This assumes KV cache = True)
            if new_shape[1] > 1:
                modified = (1 - gated_value) * inputs[torch.arange(batch_size), last_token_indices, :, :].unsqueeze(1) + gated_value * intervention_embedding
                modified_inputs[torch.arange(batch_size), last_token_indices] = modified.squeeze(1).to(dtype=modified_inputs.dtype) # (batch_size, n_heads, head_dim)
            else:
                pass

        modified_inputs = modified_inputs.view(*original_shape)

        proj_module = get_module(model, layer_name)

        out_proj = proj_module.weight
        
        if '70b' in model_config['name_or_path'].lower() or '32b' in model_config['name_or_path'].lower() or 'mixtral' in model_config['name_or_path'].lower():
            out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.data, out_proj.quant_state)
            new_output = torch.matmul(modified_inputs, out_proj_dequant.T)
        else:
            new_output = torch.matmul(modified_inputs, out_proj.T)

        return new_output
    return rep_act


def soft_intervention_natural_text(model_input, model_helper, max_new_tokens=10, return_item="both", alphas=None, avg_activations=None, batch_size=None, last_token_indices=None, exact_match=False):
    """
    This function is a wrapper of generation intervention
    """
    #Text form to avoid for-loop inside eval loop
    clean_output, intervention_output = "None", "None"

    if exact_match:
        if return_item == "clean" or return_item == "both":
            clean_output = model_helper.generate(model_input, max_new_tokens)
    else:
        if return_item == "clean" or return_item == "both":
            clean_output = model_helper.model(**model_input).logits[torch.arange(batch_size), last_token_indices,:]

    
    if return_item == "interv" or return_item == "both":
        intervention_fn = last_replace_activation_w_avg(alphas=alphas, avg_activations=avg_activations, 
                                                    model=model_helper.model, model_config=model_helper.model_config,
                                                    batched_input=False, last_token_only=True, split_idx=model_helper.split_idx,
                                                    batch_size=batch_size, last_token_indices=last_token_indices, exact_match=exact_match)
            
        if exact_match:
            with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], edit_output=intervention_fn):     
                intervention_output = model_helper.generate(model_input, max_new_tokens)
        else:
            with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], edit_output=intervention_fn):     
                intervention_output = model_helper.model(**model_input).logits[torch.arange(batch_size), last_token_indices,:]

    return clean_output, intervention_output


