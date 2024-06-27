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

#text1 = "let's find a"
#tokenized1 = tokenizer(text1, return_tensors="pt")

with open("0.txt", "r") as f:
        text = f.read()
tokenized1 = tokenizer(text, return_tensors="pt")
input_ids1 = tokenized1.input_ids.cuda()
print("input_ids1.shape: ", input_ids1.shape)

gen_tokens = model.generate(
    input_ids1[:, :-1],
    use_cache=True,
    max_new_tokens = 1,
    return_dict_in_generate=True
)
torch.cuda.synchronize()
kv = gen_tokens['past_key_values']

print("past_key_values shape: (",
          len(kv), ", ",
          len(kv), ", ",
         kv[0][0].shape, ")")

print("gen_tokens[0].shape: ", gen_tokens[0].shape)

gen_text = tokenizer.batch_decode(gen_tokens['sequences'][0])
print("gen_text after first generate: ", gen_text)

text2 = "Summarize this information"
tokenized2 = tokenizer(text2, return_tensors = "pt")
input_ids2 = tokenized2.input_ids.cuda()

generated2 = model.generate(
        input_ids2,
        max_new_tokens = 100,
        past_key_values = kv, 
        return_dict_in_generate = True, 
        #I have to manually pass the old attention mask, otherwise it gives me size mismatch errrs
        #Can you verify this is the right way to do it?
        attention_mask = tokenized1["attention_mask"].cuda()
)

print("tokens:", generated2['sequences'][0])
gen_text = tokenizer.batch_decode(generated2['sequences'][0], skip_special_tokens = True)
print("gen_text after second generate: ", gen_text)
