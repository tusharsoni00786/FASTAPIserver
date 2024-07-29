# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
#import logging
from config import SYSTEM_MESSAGE_CONTENT

# Set up logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ailanguageteacher-29atqsdlz-tushars-projects-f7169ffd.vercel.app/"],  # Allow requests from your Next.js app
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")
messages = [
    SystemMessage(content=SYSTEM_MESSAGE_CONTENT),
]



class ChatInput(BaseModel):
    message: str


@app.post("/chat")
async def chat(chat_input: ChatInput):
    try:
        messages.append(HumanMessage(content=chat_input.message))
        response = model.invoke(messages)
        ai_message = response.content
        messages.append(AIMessage(content=ai_message))
        # logger.info("Messages responses")
        # logger.info(messages)
        return {"response": ai_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reset")
async def reset_conversation():
    global messages
    messages = [
     SystemMessage(content=SYSTEM_MESSAGE_CONTENT),
    ]
    return {"message": "Conversation reset successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# # Function to add a message to the conversation and get a response
# def chat(input_text):
#     messages.append(HumanMessage(content=input_text))
#     response = model.invoke(messages)
#     messages.append(AIMessage(content=response.content))
#     return response.content

# # Main conversation loop
# print("Start chatting with the comedian AI (type 'quit' to end the conversation):")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == 'quit':
#         break
    
#     ai_response = chat(user_input)
#     print("AI: " + ai_response)

# print("Conversation ended.")
