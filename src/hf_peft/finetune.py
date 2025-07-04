# FILE: src/hf_peft/finetune.py (수정본)
import torch
# (수정) DataCollatorForLanguageModeling 임포트 추가
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import os

def run_qlora_finetuning():
    """
    GPT-2 모델을 사용하여 QLoRA 파인튜닝을 시연하는 함수입니다.
    """
    if not torch.cuda.is_available():
        print("="*60)
        print("⚠️ QLoRA는 NVIDIA GPU와 CUDA 환경이 필요합니다.")
        print("   CPU 환경에서는 실행할 수 없습니다. 스크립트를 종료합니다.")
        print("="*60)
        return

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    dataset = load_dataset("Abirate/english_quotes")
    tokenized_dataset = dataset["train"].map(
        lambda samples: tokenizer(samples["quote"], truncation=True, max_length=128), 
        batched=True,
        remove_columns=dataset["train"].column_names # <-- 불필요한 'quote' 컬럼 제거
    )

    output_dir = "./qlora_finetune_results"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    
    # (수정) 언어 모델용 데이터 콜레이터 생성
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset, # <-- 토큰화 및 정리된 데이터셋 사용
        tokenizer=tokenizer,
        data_collator=data_collator, # <-- 데이터 담당관(콜레이터) 추가
    )

    print("\nQLoRA 파인튜닝을 시작합니다...")
    trainer.train()
    print("파인튜닝이 완료되었습니다.")


if __name__ == '__main__':
    run_qlora_finetuning()