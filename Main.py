import ChatbotGpt as Chat

#Most of this is just for testing. While deyployed code will look diffrent more than likely

Bot = Chat.ChatbotGPT()

print("You started a new conversation! Type quit to exit ")
Responce = Bot.get_gpt_single_response("Hello")
print(f"ChatBot: {Responce}")

while True:
    UserInput = input("You: ")

    if UserInput.lower() == "quit": #Give option to exit conversation
        print("ChatBot: Good Bye.")
        break


    Responce = Bot.get_gpt_chat_response(UserInput)
    print(f"ChatBot: {Responce}")