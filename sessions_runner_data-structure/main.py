from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from question_answer_agent.agent import root_agent
from google.genai import types
import asyncio
from dotenv import load_dotenv
load_dotenv()
from icecream import ic


SESSION_ID="12904022-1234-5678-9012-345678901234"
APP_NAME="my_app"
USER_ID="user_12345"

state = {
    "Username": "JohnDoe",
    "Email": "jhonbabu@gmail.com",
    "FullName": "John Babu Doe",
    "Age": 27,
    "Gender": "Male",
    "Location": "San Francisco, USA",
    "PreferredLanguage": "English",
    "MembershipStatus": "Premium",
    "AccountCreated": "2021-05-12",
    "LastLogin": "2025-06-11 14:32:10",
    "Interests": ["Technology", "Music", "Travel", "Fitness"],
    "Occupation": "Software Engineer",
    "DailyActiveHours": "6-10 hours",
    "SupportTier": "Gold",
    "SubscriptionRenewalDate": "2026-05-12",
    "DevicesUsed": ["iPhone 14 Pro", "MacBook Pro M2", "iPad Air"],
    "PreferredContactMethod": "Email",
    "RecentActivity": [
        "Purchased: AI Masterclass Course",
        "Searched: 'How to optimize Python code'",
        "Viewed: 'Latest Tech Gadgets 2025'"
    ]
}


async def create_session()-> InMemorySessionService:
    session_service=InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=state,
    )
    return session_service


async def call_agent(runner: Runner, User_input: str, session_id,user_id) -> dict:
    
    content= types.Content(role="user",parts=[types.Part(text=User_input)])
    
    final_reponse_text=None
    
    async for events in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # print(f"  [Event] Author: {events.author}, Type: {type(events).__name__}, Final: {events.is_final_response()}, Content: {events.content}")
        
        if events.is_final_response():
            if events.content and events.content.parts:
                final_response_text = events.content.parts[0].text
            elif events.actions and events.actions.escalate:
                final_response_text = f"Agent escalated: {events.error_message or 'No specific message.'}"
                
            break
        
    return {"author": events.author, "content": events.content, "type": {type(events).__name__},"final_response":events.is_final_response(),"final_response_text": final_response_text}
        
        
async def main():
    
    try:
        my_session_service = await create_session()
        runner= Runner(
            session_service=my_session_service,
            app_name=APP_NAME,
            agent=root_agent,
        )

    except Exception as e:
        print(f"Error creating session and runner --->: {e}")
        return
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the session.")
            break
        
        try:
            response= await call_agent(runner=runner, User_input=user_input, session_id=SESSION_ID, user_id=USER_ID)
            print(response["final_response_text"])
            
            ic(response)
        except Exception as e:
            print(f"Error calling agent: {e}")
            break
        
if __name__ == "__main__":
    asyncio.run(main())