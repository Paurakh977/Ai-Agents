from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from question_answer_agent.agent import root_agent
from google.genai import types
import asyncio
from dotenv import load_dotenv
from icecream import ic

# Import the grounding handler
from question_answer_agent.grounding_handler import GroundingMetadataHandler

load_dotenv()

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

async def create_session() -> InMemorySessionService:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=state,
    )
    return session_service

async def call_agent(runner: Runner, user_input: str, session_id, user_id) -> dict:
    content = types.Content(role="user", parts=[types.Part(text=user_input)])
    
    final_response_text = None
    final_event = None
    all_events = []  # Store all events to find grounding metadata
    
    async for events in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        all_events.append(events)
        
        if events.is_final_response():
            final_event = events
            if events.content and events.content.parts:
                final_response_text = events.content.parts[0].text
            elif events.actions and events.actions.escalate:
                final_response_text = f"Agent escalated: {events.error_message or 'No specific message.'}"
            break
    
    return {
        "author": final_event.author if final_event else "unknown",
        "content": final_event.content if final_event else None,
        "type": type(final_event).__name__ if final_event else "unknown",
        "final_response": True,
        "final_response_text": final_response_text,
        "final_event": final_event,
        "all_events": all_events
    }

async def main():
    try:
        my_session_service = await create_session()
        runner = Runner(
            session_service=my_session_service,
            app_name=APP_NAME,
            agent=root_agent,
        )
        print("🚀 Session created successfully!")
        print("💡 You can now ask questions. The system will use Google Search when needed.")
        print("📋 When Google Search is used, grounding metadata will be displayed as required by Google's policies.")
        print("💬 Type 'exit' or 'quit' to end the session.\n")
        
        # Initialize grounding handler
        grounding_handler = GroundingMetadataHandler()
        
    except Exception as e:
        print(f"❌ Error creating session and runner: {e}")
        return
    
    while True:
        user_input = input("\n🤔 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting the session.")
            break
        
        try:
            response = await call_agent(
                runner=runner, 
                user_input=user_input, 
                session_id=SESSION_ID, 
                user_id=USER_ID
            )
            
            # Display the main response
            print(f"\n🤖 Assistant: {response['final_response_text']}")
            
            # Process grounding metadata using the handler
            grounding_info = None
            
            # Try to extract grounding info from final event first
            if response.get('final_event'):
                grounding_info = grounding_handler.extract_grounding_info(response['final_event'])
            
            # If not found, try all events
            if not grounding_info and response.get('all_events'):
                for event in response['all_events']:
                    grounding_info = grounding_handler.extract_grounding_info(event)
                    if grounding_info:
                        break
            
            # Display grounding information if available (Required by Google)
            if grounding_info:
                grounding_handler.display_console_grounding_info(grounding_info)
                grounding_handler.save_grounding_html(grounding_info, f"search_suggestions_{SESSION_ID}.html")
                grounding_handler.print_compliance_status()
            else:
                print("ℹ️  No Google Search was used for this query.")
            
            # Debug information (optional - remove in production)
            if True:  # Set to True for debugging
                ic(response)
            
        except Exception as e:
            print(f"❌ Error calling agent: {e}")
            ic(e)  # Debug the error
            break

if __name__ == "__main__":
    print("🔧 Starting Google Search Grounding Compliant Assistant...")
    asyncio.run(main())