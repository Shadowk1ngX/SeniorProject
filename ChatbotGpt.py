import openai
import requests
import json


with open('Key.json', 'r') as file: # I will have to email you the key file as github doesnt allow you to upload keys
    data = json.load(file)
    

ApiKey = data["AuthKey"] 
ApiUrl = "https://api.openai.com/v1/chat/completions"


class ChatbotGPT:
    def __init__(self, Model="gpt-4", Content = "You are a helpful asstiant"):
        self.api_key = ApiKey
        self.api_url = ApiUrl
        self.model = Model
        self.content = Content
        self.max_history = 5
        self.max_tokens = 100
        self.max_char_limit = 85
        self.temperature = 1

    def set_content(self,NewContent):
        self.content = NewContent


    def get_gpt_single_response(self, user_input):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.content},
                {"role": "user", "content": user_input}
            ]
        }

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                response_data = response.json()
                return response_data['choices'][0]['message']['content']
            else:
                print(f"Request failed with error: {response.status_code} - {response.reason}")
                return "Sorry, something went wrong."
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Sorry, something went wrong."


    def get_gpt_chat_response(self, user_input):
        
        if len(user_input) > self.max_char_limit:
            user_input = user_input[:self.max_char_limit]
            print(f"Input was too long, truncating to {self.max_char_limit} characters.")

        if not hasattr(self, "conversation_history"):
            self.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

        self.conversation_history.append({"role": "user", "content": user_input})

        if len(self.conversation_history) > self.max_history:
            self.conversation_history = (
                [self.conversation_history[0]] + self.conversation_history[-self.max_history:]
            )
   
        data = {
            "model": self.model,
            "messages": self.conversation_history,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

       
        response = requests.post(self.api_url, headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            assistant_message = response_data["choices"][0]["message"]["content"]
  
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            return assistant_message
        else:
            print(f"Request failed with error: {response.status_code} - {response.reason}")
            return "Sorry, something went wrong."
