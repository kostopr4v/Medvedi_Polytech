from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Model:
    def __init__(self):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else {"": self.device}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        self.task_prefix = "You are given list of games of particular team. Your task is to analyse it and give a logically correct answer. Based on columns and their description, suggest how you can improve game results. Answer only on russian"
    
    def predict(self, user_prompt):
        full_prompt = f"{self.task_prefix} {user_prompt}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        
        # Применяем шаблон чата, если метод доступен
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Альтернативный способ формирования запроса
            text = f"{self.system_prompt}\n{full_prompt}"
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(**model_inputs,max_new_tokens=2048)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response