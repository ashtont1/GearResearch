from GEARLM import GearLlamaForCausalLM
from GEARLM import GearLlamaForCausalLMNew
from transformers import LlamaForCausalLM
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
import torch
import time
import os
import argparse
import pickle
import json
#model, tokenizer = load_model(
    #"lmsys/longchat-7b-16k",
    #device="cuda",
    #num_gpus=1,
    #max_gpu_memory="32GiB",
    #load_8bit=True,
    #cpu_offloading=False,
    #debug=False,
#)
#model = AutoModelForCausalLM.from_pretrained("lmsys/longchat-7b-16k")
tokenizer = AutoTokenizer.from_pretrained("lmsys/longchat-7b-16k")

compress_config = {}
compress_config["compress_mode"] = "gear_batch" # batchwise-GEAR
#compress_config["compress_mode"] = "outlier_batch" # batchwise-GEAR
compress_config["quantize_bit"] = 4 # outlier quantization bit
compress_config["left"] = 0.01 # outlier extraction rate
#compress_config["left"] = 0.01 # outlier extraction rate
#compress_config["rank"] = 0.02  # setting rank for Key and value cache quantization error
compress_config["rank"] = 0.01  # setting rank for Key and value cache quantization error
compress_config["loop"] = 3 # a constant for power iteration(an efficient SVD solver)
compress_config["stream"] = True # streaming-gear set to true to perform better efficiency
compress_config["streaming_gap"] = 20 # re-compress every 20 iteration 

model = GearLlamaForCausalLMNew.from_pretrained(
    "lmsys/longchat-7b-16k",
    cache_dir="../cache",
    device_map="cuda",
    compress_config=compress_config,
    torch_dtype=torch.float16,
    use_cache = True,
    # torch_dtype = torch.float16,
)

text1 = "let's find a"
tokenized1 = tokenizer(text1, return_tensors="pt")

#with open("0.txt", "r") as f:
        #text = f.read()
#tokenized1 = tokenizer(text, return_tensors="pt")
input_ids1 = tokenized1.input_ids.cuda()
print("input_ids1: ", input_ids1)

gen_tokens = model.generate(
    input_ids1,
    use_cache=True,
    max_new_tokens = 2,
    return_dict_in_generate=True
)
torch.cuda.synchronize()
kv = gen_tokens['past_key_values']

text2 = "let's find a bug"
tokenized2 = tokenizer(text2, return_tensors="pt")
input_ids2 = tokenized2.input_ids.cuda()
print("input_ids2.shape: ", input_ids2.shape)
print("input_ids2: ", input_ids2)

attn = torch.cat((tokenized1["attention_mask"], tokenized2["attention_mask"]), -1)
#gen_text = tokenizer.batch_decode(gen_tokens)[0]
#print("gen_tokens: ", gen_tokens)
#print("gen_text: ", gen_text)
print("attn: ", attn)
#generated2 = model.generate(input_ids2, past_key_values = kv, attention_mask=attn)
generated2 = model.generate(input_ids2, max_new_tokens = 50, past_key_values = kv, return_dict_in_generate = True)
print("generated2[0]: ", generated2[0])
gen_text_array = tokenizer.batch_decode(generated2['sequences'][0])
print("tokens:", generated2['sequences'][0])
print("gen_text_array: ", gen_text_array)
gen_text = tokenizer.decode(generated2['sequences'][0], skip_special_tokens = True)
print("gen_text: ", gen_text)
