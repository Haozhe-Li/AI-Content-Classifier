from transformers import GPT2LMHeadModel, GPT2TokenizerFast
GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="./core/model/gpt2")
GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="./core/model/gpt2")