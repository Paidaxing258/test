from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json


class ModelScopeModel:
    def __init__(self, model_id: str, config_dir: str, config_name: str, token=None):
        """
        初始化ModelScope模型
        
        Args:
            model_id: ModelScope模型ID（如"qwen/Qwen-7B-Chat"）
            config_dir: 配置文件目录
            config_name: 配置文件名
            token: ModelScope令牌（公开模型可不填）
        """
        print(f"检查模型在 '{config_dir}/model_ckpt' 中...")
        model_dir = f"{config_dir}/model_ckpt"
        os.makedirs(model_dir, exist_ok=True)
        # 替换模型ID中的特殊字符作为本地缓存目录
        model_path = os.path.join(model_dir, model_id.replace("/", "_"))

        # 从ModelScope下载或加载缓存模型
        if not os.path.exists(model_path):
            print(f"模型未找到，从ModelScope下载: {model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                token=token  # 私有模型需要令牌
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"模型保存至: {model_path}")
        else:
            print(f"使用缓存模型: {model_path}")

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # 加载配置和聊天模板（与原逻辑保持一致）
        self.config = json.load(open(f'{config_dir}/generation_configs/{config_name}.json'))
        chat_template = open(f'{config_dir}/{self.config["chat_template"]}').read()
        self.tokenizer.chat_template = chat_template.replace('    ', '').replace('\n', '')
        print("模型加载完成（自动分配至可用设备）")

    # 以下生成方法与HuggingFaceModel完全一致，直接复用
    def generate(self, system: str, user: str, max_length: int = 1000, **kwargs):
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(plain_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,** kwargs
        )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response

    def continue_generate(self, system: str, user1: str, assistant1: str, user2: str, max_length: int = 1000, **kwargs):
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user1},
            {'role': 'assistant', 'content': assistant1},
            {'role': 'user', 'content': user2},
        ]
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(plain_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,** kwargs
        )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response

    def conditional_generate(self, condition: str, system: str, user: str, max_length: int = 1000, **kwargs):
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + condition
        inputs = self.tokenizer(plain_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,** kwargs
        )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response