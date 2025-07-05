import torch
import nltk
import numpy as np
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
from ray import tune
from ray.air import session, RunConfig
from ray.tune.search.optuna import OptunaSearch
from src.utils.loss import GeometricLoss

class SummarizationTrainable(tune.Trainable):
    def setup(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.model.to(self.device)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        dataset = load_dataset("cnn_dailymail", "3.0.0")
        
        # ### 수정된 부분 1: 훈련/검증 데이터셋을 모두 가져옴 ###
        train_dataset_raw = dataset["train"].select(range(200))
        eval_dataset_raw = dataset["validation"].select(range(100))

        prefix = "summarize: "
        def preprocess_function(examples):
            inputs = [prefix + doc for doc in examples["article"]]
            model_inputs = self.tokenizer(inputs, max_length=512, truncation=True)
            labels = self.tokenizer(text_target=examples["highlights"], max_length=128, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # 원본 컬럼 이름을 저장해두었다가, 전처리 후 제거
        original_columns = train_dataset_raw.column_names
        
        # ### 수정된 부분 2: 훈련/검증 데이터셋을 모두 전처리하고 원본 컬럼 제거 ###
        self.train_dataset = train_dataset_raw.map(preprocess_function, batched=True, remove_columns=original_columns)
        self.eval_dataset = eval_dataset_raw.map(preprocess_function, batched=True, remove_columns=original_columns)
        
        self.trainer_class = self._create_geometric_trainer(alpha=config["alpha"])

    def _create_geometric_trainer(self, alpha):
        # (변경 없음)
        class GeometricTrainer(Seq2SeqTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.geometric_loss_calculator = GeometricLoss(alpha=alpha, gamma=0.0)

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs, labels=labels, output_hidden_states=True)
                base_loss = outputs.loss
                flow_vectors = outputs.decoder_hidden_states[-1]
                with torch.no_grad():
                    embedding_labels = labels.clone()
                    embedding_labels[embedding_labels == -100] = 0
                    target_vectors = model.get_decoder().embed_tokens(embedding_labels)
                resonance_loss = self.geometric_loss_calculator.calculate_resonance_loss(flow_vectors, target_vectors)
                final_loss = base_loss + self.geometric_loss_calculator.alpha * resonance_loss
                return (final_loss, outputs) if return_outputs else final_loss
        return GeometricTrainer

    def step(self):
        rouge = evaluate.load("rouge")
        def compute_metrics(eval_pred):
            # (변경 없음)
            predictions, labels = eval_pred
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
            result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            return {k: round(v * 100, 4) for k, v in result.items()}

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.logdir,
            learning_rate=self.config["lr"],
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            predict_with_generate=True,
            evaluation_strategy="epoch",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            # 데이터셋을 미리 완벽하게 처리했으므로 이 옵션은 더 이상 필요 없음
            # remove_unused_columns=False, 
        )

        trainer = self.trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        return {"rouge1": eval_results["eval_rouge1"], "loss": eval_results["eval_loss"]}

def main():
    # (main 함수는 변경 없음)
    nltk.download('punkt', quiet=True)
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-4),
        "alpha": tune.uniform(0, 1.0)
    }
    search_alg = OptunaSearch(metric="rouge1", mode="max")
    tuner = tune.Tuner(
        tune.with_resources(SummarizationTrainable, {"cpu": 2, "gpu": 1}),
        param_space=search_space,
        tune_config=tune.TuneConfig(search_alg=search_alg, num_samples=20),
        run_config=RunConfig(name="geometric_finetune_hpo"),
    )
    print("--- Starting HPO for Geometric Fine-tuning ---")
    results = tuner.fit()
    best_result = results.get_best_result(metric="rouge1", mode="max")
    print("\n--- HPO Finished ---")
    if best_result:
        print(f"🏆 Best trial's ROUGE-1 score: {best_result.metrics['rouge1']:.4f}")
        print("Best hyperparameters found: ", best_result.config)
    else:
        print("❌ No successful trials were completed.")
    print("--------------------")

if __name__ == '__main__':
    main()