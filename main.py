import asyncio
from dotenv import load_dotenv
from livekit.agents import (
    Agent, AgentSession, JobContext,
    WorkerOptions, cli, function_tool,
)
from livekit.plugins import groq, silero
from langchain_core.messages import HumanMessage
from agent import graph

load_dotenv()

TTS_AVAILABLE = True


async def safe_say(session: AgentSession, text: str):
    global TTS_AVAILABLE
    if not TTS_AVAILABLE:
        await session.send_text(f"[TTS unavailable] {text}")
        return
    try:
        await session.say(text)
    except Exception as e:
        print("⚠️  TTS failed:", e)
        TTS_AVAILABLE = False
        await session.send_text(f"[Text-to-Speech quota exhausted] {text}")


# ---------- TOOL ----------
@function_tool
async def itinerary_planner(user_input: str) -> dict:
    """
    Take raw user_input, run graph.invoke, return clean text in the schema
    LiveKit expects so the LLM can read / speak it.
    """
    try:
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
        final_data = result.get("final_data") if isinstance(result, dict) \
                     else getattr(result, "final_data", None)
        if not final_data:
            final_data = (
                "Sorry, I couldn't plan your itinerary. "
                "Please provide more details about your trip."
            )
        try:
            with open("your plan.txt", "w", encoding="utf-8") as f:
                f.write(final_data)
            saved_msg = "Your trip plan is saved."
        except Exception as file_err:
            print("⚠️  File save error:", file_err)
            saved_msg = "But I couldn't save your itinerary to file."
        #print("✅ Final data:", final_data)
        return {"response": f"{final_data}\n\n{saved_msg}"}         
    except Exception as err:
        print("⚠️  Planning error:", err)
        return {"response": f"I had a problem planning your trip: {err}"}


# ---------- ENTRYPOINT ----------
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = Agent(
        instructions="""
You are Itine, a trip-planning assistant.

Flow:
1. Greet the user and ask for trip details in one sentence.
2. As soon as the user answers, call the itinerary_planner tool with their
   exact text; do NOT ask follow-up questions.
3. Present the tool result naturally. Do not mention tools or JSON.

Always speak the complete itinerary (or fallback to text if TTS is down).
""",
        tools=[itinerary_planner],
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=groq.STT(),
        llm=groq.LLM(model="gemma2-9b-it"),
        tts=groq.TTS(model="playai-tts", voice="Arista-PlayAI"),
    )

    await session.start(agent=agent, room=ctx.room)
    await asyncio.sleep(1)

    await safe_say(session,
                   "Hello! I'm Itine, your itinerary planner. "
                   "Tell me about your trip!")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
