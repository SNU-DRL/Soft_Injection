from utils import *
from models import *
from preprocess import *
from tqdm import tqdm
import torch
import argparse
from time import time
torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error() 


def eval_site(args):
    os.makedirs(args.activation_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.alphas_path), exist_ok=True)
    os.makedirs(args.result_folder, exist_ok=True)
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    data_path = args.data_path
    dataset_names = args.dataset_names
    exact_match = args.exact_match # If False, use first token match for evaluation.
    
    train_dataset = [load_dataset(os.path.join(data_path, dataset_name, 'train.json')) for dataset_name in dataset_names]
    valid_dataset = [load_dataset(os.path.join(data_path, dataset_name, 'valid.json')) for dataset_name in dataset_names]
    test_dataset = [load_dataset(os.path.join(data_path, dataset_name, 'test.json')) for dataset_name in dataset_names]

    gradient_opt_data = train_dataset
    
    eval_data = []
    for dataset in valid_dataset:
        sub_dataset = dataset[:100]
        sub_dataset = [{"input": i, "output": o} for i, o in zip(sub_dataset["input"], sub_dataset["output"])]
        eval_data.append(sub_dataset)

    save_json = {}
    time_cost = {}
    accuracy = {}
    ##Load the model
    model_helper = load_model(args.model_name, args.data_name, args.lr)

    if args.cur_mode != "clean":

        # Construction of a task embedding
        start_time = time()
        mean_activations = []
        for i in range(len(train_dataset)):
            if not os.path.exists(os.path.join(args.activation_path, f"{dataset_names[i]}.pt")):
                print(f"Computing mean activation for {dataset_names[i]}")
                mean_activation = get_last_mean_head_activations([train_dataset[i]], model_helper, N_TRIALS = args.num_example, shot=args.num_shot)  
                torch.save(mean_activation, os.path.join(args.activation_path, f"{dataset_names[i]}.pt"))
            mean_activation = torch.load(os.path.join(args.activation_path, f"{dataset_names[i]}.pt"))
            mean_activations.append(mean_activation)
        mean_activations = torch.stack(mean_activations, dim=0)
        end_time = time()
        time_cost['time_for_mean_activations'] = end_time - start_time

        # Optimization of soft head-selection parameters (alphas)
        start_time = time()
        if not os.path.exists(args.alphas_path):
            alphas = optimize_head_selection(mean_activations, model_helper, gradient_opt_data, eval_data, args.epoch, batch_size, test_batch_size, dataset_names).detach().cpu()
            torch.save(alphas, args.alphas_path)
        alphas = torch.load(args.alphas_path).to(model_helper.model.device)
        end_time = time()
        time_cost['time_for_alphas'] = end_time - start_time

    else:
        alphas = None
        mean_activations = None

    ## Test process
    # 0-shot prompt + Soft Injection with task embedding and optimized soft head-selection parameters
    start_time = time()
    if exact_match and test_batch_size > 1:
        test_batch_size = 1
        print("Exact match evaluation only supports batch size 1. Setting test_batch_size to 1.")    
    mean_interv_acc = 0
    mean_clean_acc = 0
    for idx, test_data in enumerate(test_dataset):
        clean_count, interv_count = 0, 0
        for batch_start in tqdm(range(0, len(test_data), test_batch_size)):
            batch_end = min(batch_start + test_batch_size, len(test_data))
            current_batch_size = batch_end - batch_start
            items = [test_data[i] for i in range(batch_start, batch_end)]
            new_input, _, target_outs, _, target_string = model_helper.format_func([train_dataset[idx]] * current_batch_size, items, num_shot=args.eval_num_shot, model_helper=model_helper, batch_size=current_batch_size)
            attention_mask = new_input["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1
            last_token_indices = last_token_indices.tolist()
            if mean_activations is not None:
                batched_mean_activations = torch.stack([mean_activations[idx]] * current_batch_size, dim=0)
            else:
                batched_mean_activations = None

            clean_out, interv_out = soft_intervention_natural_text(new_input, model_helper, max_new_tokens=args.max_token, return_item=args.cur_mode, alphas=alphas, avg_activations=batched_mean_activations, batch_size=current_batch_size, last_token_indices=last_token_indices, exact_match=exact_match)

            if not exact_match:
                for i in range(current_batch_size):
                    if args.cur_mode == "interv" or args.cur_mode == "both":
                        interv_count += compute_individual_token_match(interv_out[i], target_outs[i])
                    if args.cur_mode == "clean" or args.cur_mode == "both":
                        clean_count += compute_individual_token_match(clean_out[i], target_outs[i])
            else:
                # St. = Saint, skip abbreviation for country-currency dataset (due to the inconsistency between samples in the dataset; some samples include abbreviations but some do not)
                target_string = target_string.replace('St.', 'Saint').split("(")[0].strip() 
                clean_out = clean_out.replace('St.', 'Saint').split("\n")[0].split("(")[0].strip() 
                interv_out = interv_out.replace('St.', 'Saint').split("\n")[0].split("(")[0].strip() 
                if 'capitalize' not in dataset_names[idx].lower() and 'lowercase' not in dataset_names[idx].lower():
                    # some datasets have inconsitent capitalization between input and output
                    target_string = target_string.lower()
                    clean_out = clean_out.lower()
                    interv_out = interv_out.lower()
                target_len = len(target_string) 
                clean_count += int(clean_out[:target_len] == target_string[:target_len])
                interv_count += int(interv_out[:target_len] == target_string[:target_len])

        if args.is_eval:
            if args.cur_mode == "interv" or args.cur_mode == "both":
                interv_acc = interv_count / len(test_data)
                accuracy[dataset_names[idx]] = {"Intervention Accuracy": interv_acc}
                mean_interv_acc += interv_acc
                print(f"Intervention Score for {dataset_names[idx]}: {interv_acc}")
            if args.cur_mode == "clean" or args.cur_mode == "both":
                clean_acc = clean_count / len(test_data)
                accuracy[dataset_names[idx]] = {"Clean Accuracy": clean_acc}
                mean_clean_acc += clean_acc
                print(f"Clean Score for {dataset_names[idx]}: {clean_acc}")
            if args.cur_mode == "both":
                accuracy[dataset_names[idx]] = {"Intervention Accuracy": interv_acc, "Clean Accuracy": clean_acc}
    mean_interv_acc /= len(test_dataset)
    mean_clean_acc /= len(test_dataset)
    accuracy["Mean"] = {"Intervention Accuracy": mean_interv_acc, "Clean Accuracy": mean_clean_acc}
    end_time = time()
    time_cost["time_for_eval"] = end_time - start_time
    save_json["accuracy"] = accuracy
    save_json["time_cost"] = time_cost

    os.makedirs(args.result_folder, exist_ok=True)
    save_json_path = os.path.join(args.result_folder, f"{args.experiment_name}.json")
    with open(save_json_path, 'w') as f:
        json.dump(save_json, f, indent=4)
    
    print(f"Successfully ran eval.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8b")
    parser.add_argument("--data_name", type=str, default="icl")
    parser.add_argument("--num_example", type=int, default=50)
    parser.add_argument("--num_shot", type=int, default=10)
    parser.add_argument("--eval_num_shot", type=int, default=0)
    parser.add_argument("--max_token", type=int, default=20)
    parser.add_argument("--is_eval", type=bool, default=True)
    parser.add_argument("--result_folder", type=str, default='./result')
    parser.add_argument("--cur_mode", type=str, default="interv")
    parser.add_argument("--experiment_name", type=str, default="exp_0")
    parser.add_argument("--activation_path", type=str, default=None)
    parser.add_argument("--alphas_path", type=str, default=None)
    parser.add_argument("--dataset_names", nargs='+', type=str, default=['ag_news'])
    parser.add_argument("--lr", type=float, default=2e-1)
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument('--exact_match', action='store_true', help='Whether to use exact match for evaluation. Otherwise, first token match for evaluation will be used.')
    parser.add_argument('--data_path', type=str, default='./dataset')
    
    args = parser.parse_args()

    if args.batch_size > 1 or args.test_batch_size > 1:
        print(f"Warning: Setting batch_size > 1 may cause unexpected results for Mistral, Gemma3 models, etc.")
        print(f"For batch_size > 1, use Llama-3.1, Qwen3 models, or raise to fp32 precision.")

    eval_site(args)

