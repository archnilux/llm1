import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model ve tokenizer'ı yükle
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Daha az bellek kullanımı için
    device_map="auto",  # Otomatik cihaz ataması
    trust_remote_code=True
)

# Sohbet geçmişini yönetmek için
chat_history = []

def predict(message, history):
    # Sohbet geçmişini düzenle
    prompt = "Aşağıdaki bir sohbet geçmişidir. Sohbeti devam ettir.\n\n"
    
    for user_msg, assistant_msg in history:
        prompt += f"Kullanıcı: {user_msg}\nAsistan: {assistant_msg}\n\n"
    
    prompt += f"Kullanıcı: {message}\nAsistan: "
    
    # Yanıt oluştur
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# Gradio arayüzü oluştur
demo = gr.ChatInterface(
    fn=predict,
    title="Phi-2 Mini Sohbet Asistanı",
    description="Microsoft Phi-2 (2.7B) tabanlı basit bir LLM sohbet arayüzü",
    examples=["Merhaba, kendini tanıtır mısın?", 
              "Python'da bir sıralama algoritması yazabilir misin?",
              "Yapay zeka hakkında kısa bir şiir yazar mısın?"],
    theme="soft"
)

# Demo uygulamasını başlat
if __name__ == "__main__":
    demo.launch()