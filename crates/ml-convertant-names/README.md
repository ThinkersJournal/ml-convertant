# ml-convertant-names

Renames LoRA adapter tensors between the PEFT (`adapter_model.safetensors`) form and the llama.cpp GGUF adapter form, in both
directions, for the Llama-style Hugging Face layout (Llama, Gemma 2, Qwen-style). A name with no mapping here is refused with a
typed error, never guessed. No I/O, no dependencies, and no MLMF: the caller reads and writes the files.
Part of [ml-convertant](https://github.com/ThinkersJournal/ml-convertant).
