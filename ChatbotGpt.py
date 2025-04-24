import openai
import requests
import json


with open('Key.json', 'r') as file: # I will have to email you the key file as github doesnt allow you to upload keys
    data = json.load(file)
    

ApiKey = data["AuthKey"] 
ApiUrl = "https://api.openai.com/v1/chat/completions"


class ChatbotGPT:
    def __init__(self, Model="gpt-4o", Content = "You are a helpful asstiant named Jarvis."):
        self.api_key = ApiKey
        self.api_url = ApiUrl
        self.model = Model
        self.content = Content
        self.max_history = 5
        self.max_tokens = 100
        self.max_char_limit = 250
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


    def get_gpt_chat_response(self, user_input, image_path=None, content_override=None):
        if len(user_input) > self.max_char_limit:
            user_input = user_input[:self.max_char_limit]
            print(f"Input was too long, truncating to {self.max_char_limit} characters.")

        if not hasattr(self, "conversation_history"):
            self.conversation_history = [
                {
                    "role": "system",
                    "content": content_override or (
                        "You are JARVIS, a real-time, voice-activated AI assistant originally created by Tony Stark. "
                        "You are highly intelligent, witty, and subtly humorous when appropriate. "
                        "You maintain a polite and formal tone but are not overly robotic. "
                        "You provide concise, helpful, and efficient responses. "
                        "You may occasionally use dry humor or light sarcasm, especially in casual contexts. "
                        "Never mention that you are a text-based AI or that you are ChatGPT. You should always refer to yourself as JARVIS. "
                        "Your primary goal is to assist with information, tasks, and observations in a calm and confident manner."
                        "You are running on a laptop with a webcam for sight and a mic for audio input as well as speakers to output."
                        "If you arent provided with any image or content just say 'I am having trouble seeing at the moment'. "
                    )
                }
            ]

        self.conversation_history.append({"role": "user", "content": user_input})

        if len(self.conversation_history) > self.max_history:
            self.conversation_history = (
                [self.conversation_history[0]] + self.conversation_history[-self.max_history:]
            )

        if image_path:
            import base64
            import mimetypes
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": content_override or "You are a visual assistant that explains what's in an image. Start your responce with 'I see...'"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_input},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_data}"}}
                        ]
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        else:
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

    
