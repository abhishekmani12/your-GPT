def train(dpath, model, output_path):

    train_dataset = load_dataset(dpath, split="train")

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    #Model similar to llama
    model = AutoModelForCausalLM.from_pretrained(
        model,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_int8_training(model)

    #peft model
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=4,
        optim="adamw_torch",
        logging_steps=100,
        learning_rate=2e-4,
        fp16=True,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        num_train_epochs=1,
        save_strategy="epoch",
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        peft_config=peft_config,
    )