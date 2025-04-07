import os  
import sys  
import yaml  
import logging  
from openai import AzureOpenAI  
  
logger = logging.getLogger(__name__)  
AZURE_CONFIG = '/mnt/azure_config.yaml'  
  
class Gpt4oTranslate:  
    def __init__(self):  
        with open(AZURE_CONFIG, 'r') as file:  
            self.config = yaml.safe_load(file)  
        self.client = AzureOpenAI(api_key=self.config['API_KEY'],  
                                  api_version=self.config['AZURE_API_VERSION'],  
                                  azure_endpoint=self.config['AZURE_ENDPOINT'],  
                                  azure_deployment=self.config['AZURE_DEPLOYMENT'])  
  
    def translate(self, user_prompt):  
        system_prompt = f"""你現在是智慧家電的使用者，你要根據我提供的屬性內容去講出控制家電的內容。  
                            總共要 25 句不重複之文本，且語意要通順合理。   
                            結果只需回答文本即可，不需要編碼等額外資訊。  
                              
                            example:  
                            物件: 電燈  
                            行動: 開  
                            執行區域: 客廳  
                              
                            result:  
                            把客廳的燈打開  
                            開啟客廳的燈  
                            把客廳的電燈點亮  
                            點亮客廳的燈  
                            讓客廳的燈亮起來  
                            開啟客廳的電燈  
                            打開客廳的燈  
                            讓客廳的燈光亮起來  
                            讓客廳的燈點亮  
                            把客廳的燈光打開  
                            客廳的燈亮起來  
                            把客廳的電燈打開  
                            讓客廳的電燈亮起來  
                            把客廳的燈光開啟  
                            把客廳的燈點亮  
                            打開客廳的電燈  
                            讓客廳的電燈點亮  
                            讓客廳的燈打開  
                            開客廳的電燈  
                            把客廳的電燈開啟  
                            點亮客廳的電燈  
                            客廳的電燈亮起來  
                            客廳的燈光打開  
                            開起客廳的電燈  
                            讓客廳的燈光點亮  
                              
                            錯誤結果:  
                            1. XXX (出現編碼、額外資訊等，因此不允許顯示)
                            啟動客廳的燈 (啟動不是合用在燈的控制上，因此語意不合理)  
                            讓客廳充滿光亮 (未指定到電燈或是燈，缺乏物件，因此不合理)  
                            點亮客廳 (未指定到電燈或是燈，缺乏物件，因此不合理)  
                            讓客廳的燈具亮起來 (燈具不為合理之物件，因此不合理)"""  
  
        response = self.client.chat.completions.create(  
            model=self.config['AZURE_DEPLOYMENT'],  
            messages=[  
                {"role": "system", "content": system_prompt},  
                {"role": "user", "content": user_prompt}  
            ],  
            max_tokens=4000,  
            temperature=0,  
        )  
  
        return response.choices[0].message.content  
    
class FineTune:  
    def __init__(self):  
        with open(AZURE_CONFIG, 'r') as file:  
            self.config = yaml.safe_load(file)  
        self.client = AzureOpenAI(api_key=self.config['API_KEY'],  
                                  api_version=self.config['AZURE_API_VERSION'],  
                                  azure_endpoint=self.config['AZURE_ENDPOINT'],  
                                  azure_deployment=self.config['AZURE_DEPLOYMENT'])  
  
    def translate(self, user_prompt):  
        system_prompt = f"""你看到的文本皆是提供的物件、行動、執行區域所組合出來的文本共25句
                            你要判斷我提供文本是否合理或是有可以改良的文本
                            並且所有文本不可重複，且語意要通順合理，相差一個字也算不同文本

                                                        
                            語意問題之文本範例如下:
                            例如1: 烤箱這邊要的意思是啟動、運作或是工作，不要出現打開之類的，我要的是運作不是開關 (烤箱、微波爐、洗衣機、掃地機器人、咖啡機、洗碗機都同理)
                            例如2: 電燈是要打開而不是啟動或是運作 """
  
        response = self.client.chat.completions.create(  
            model=self.config['AZURE_DEPLOYMENT'],  
            messages=[  
                {"role": "system", "content": system_prompt},  
                {"role": "user", "content": user_prompt}  
            ],  
            max_tokens=4000,  
            temperature=0,  
        )  
  
        return response.choices[0].message.content  
    
    
# if __name__ == '__main__':  
#     gpt4o = Gpt4oTranslate()  
#     user_prompt = f"""  物件: 烤箱
#                         行動: 啟動、運作、工作
#                         執行區域: 廚房
                        
#                         行動只需要則一即可，不要出現打開之類的，我要的是運作不是開關"""  

#     coarse = gpt4o.translate(user_prompt)  
#     print(coarse)  
  
if __name__ == '__main__':  
    import json  
  
    gpt4o = FineTune()  
  
    # 定義 JSON 檔案的路徑  
    file_path = 'data.json'  
  
    # 打開並讀取 JSON 檔案  
    with open(file_path, 'r', encoding='utf-8') as file:  
        with open("data2.json", 'w', encoding='utf-8') as file2:  
            data = json.load(file)  
  
            # 逐個處理每個項目  
            for item in data:  
                obj = item['Object']  
                action = item['Action']  
                area = item['Area']  
                text = item['text']  
                user_prompt = f"""物件: {obj}\n行動: {action}\n執行區域: {area}\n文本: {text}"""  
  
                coarse = gpt4o.translate(user_prompt)  
                print(coarse)  


