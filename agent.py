from typing import Dict
from typing_extensions import Literal
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dateutil.parser import parse
from tools import (
    get_flights_data,
    get_hotels_data,
    get_city_coordinates,
    get_restaurants_data
)
import os, json, re
from datetime import datetime
CURRENT_YEAR = datetime.today().year

from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = init_chat_model("groq:llama3-8b-8192")
llm2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class State(MessagesState):
    """State for the multi-agent system."""
    next_agent: str = ""
    flight_data: str = ""
    hotel_data: str = ""
    weather_data: str = ""
    restaurant_data: str = ""
    activities_data: str = ""
    final_data: str = ""
    task_complete: bool = False
    current_task: str = ""
    user_query: str = ""
    
def create_TripAgent_Chain():
    """Creates the TripAgent Decision chain."""
    TripAgent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Trip Agent that manages a team of agents to plan a trip:
        1. Flight Agent: Gather information on flights.
        2. Hotel Agent: Gather information on nearby/available hotels.
        3. Weather Agent: Checks the weather.
        4. Restaurant Agent: Finds nearby restaurants.
        5. Activity Agent: Finds activities that can be done in the destination.
        
        Based on the current state and conversation, decide which agent should work next.
        If the task is complete, respond with 'DONE'.
        
        Current State: 
        - Has flight data: {has_flight}
        - Has hotel data: {has_hotel}
        - Has weather data: {has_weather}
        - Has restaurant data: {has_restaurant}
        - Has activities data: {has_activities}
        - Has final data: {has_final_data}
        
        Respond with ONLY the agent name (Flight/Hotel/Weather/Restaurant/Activities) or 'DONE'.
         """),
        ("human", "{task}"),
    ])
    return TripAgent_prompt | llm2
    

def TripAgent(State)-> Dict:
    """TripAgent decides next agent"""
    
    messages = State["messages"]
    task = messages[-1].content if messages else "No Task"
    query = State.get("user_query", messages[-1].content if messages else "No Task")
    
    # Check if tasks are already completed
    has_flight = bool(State.get("flight_data",""))
    has_hotel = bool(State.get("hotel_data",""))
    has_weather = bool(State.get("weather_data",""))
    has_restaurant = bool(State.get("restaurant_data",""))
    has_activities = bool(State.get("activities_data",""))
    has_final_data = bool(State.get("final_data",""))
    
    # LLM decision based on current task & filled slots
    chain = create_TripAgent_Chain()
    decision = chain.invoke({
        "task": task,
        "has_flight": has_flight,
        "has_hotel": has_hotel,
        "has_weather": has_weather, 
        "has_restaurant": has_restaurant,
        "has_activities": has_activities,
        "has_final_data": has_final_data,
    })
    
    decision_next = decision.content.strip().lower()
    print(f"TripAgent Decision: {decision_next}")
    
    # Routing decision
    if "done" in decision_next or has_final_data:
        next_agent = "end"
        TripAgent_message = "All tasks are complete! Finalizing your trip. ✈️🏨☀️"
    elif "flight" in decision_next or not has_flight:
        next_agent = "FlightAgent"  
        TripAgent_message = "Let's book your flight. Assigning to Flight Agent🛫"
    elif "hotel" in decision_next or not has_hotel:
        next_agent = "HotelAgent"  
        TripAgent_message = "We'll look for hotels. Assigning to Hotel Agent🏨"
    elif "weather" in decision_next or not has_weather:
        next_agent = "WeatherAgent"  
        TripAgent_message = "Let me check the weather for your destination. Assigning to Weather Agent🌤"
    elif "restaurant" in decision_next or not has_restaurant:
        next_agent = "RestaurantAgent"  
        TripAgent_message = "Let me find some restaurants you'll love. Assigning to Restaurant Agent🍽️"
    elif "activity" in decision_next or "activities" in decision_next or not has_activities:
        next_agent = "ActivityAgent"  
        TripAgent_message = "Let's explore activities you can do! Assigning to Activities Agent🎡"
    else:
        next_agent = "end"
        TripAgent_message = "✅ Trip Agent: Task seems complete."
        
    return {
        "messages": [AIMessage(content=TripAgent_message)],
        "next_agent": next_agent,
        "current_task": task,
        "user_query": query,
    }

def FlightAgent(State):
    """Agent responsible for booking flights."""
    flight_data = State.get("flight_data", "")
    task = State.get("user_query", "")
    # print("*TASK*\n",task)
    prompt = f"""
        You are an intelligent flight booking assistant, designed to extract crucial flight details from user travel plans. Your goal is to accurately identify the departure city, arrival city, and travel date.

        Here's the user's **Travel Plan**:
        {task}

        Here's any **Previous Known Data** that might be relevant:
        {flight_data}
        Use the current year ({CURRENT_YEAR}) if the Year is not specified.

        **Your task is to extract the following information, if available:**
        - **departure_city_code**: The IATA code for the departure city (e.g., "JFK", "LAX").
        - **arrival_city_code**: The IATA code for the arrival city (e.g., "LHR", "NRT").
        - **travel_date**: The specific date of travel in 'YYYY-MM-DD' format (e.g., "2025-08-15").

        **Important:**
        - **Respond ONLY with a valid JSON object.**
        - **Do NOT include any additional text, explanations, or conversational filler.**
        - **If a piece of information is not found, its corresponding value in the JSON should be `null`.**

        **Example of Expected JSON Output:**
        ```json
        {{
        "departure_city_code": "NYC",
        "arrival_city_code": "SFO",
        "travel_date": "2025-08-15"
        }}
        **Strictly** return only the JSON do not include any other text or explanation."""

    try:
        parsed = llm.invoke([HumanMessage(content=prompt)])
        match = re.search(r"\{[\s\S]*\}", parsed.content.strip())
        if not match:
            raise ValueError("No JSON found in LLM response")
        extracted = json.loads(match.group(0))
        #extracted = json.loads(parsed.content)  
        departure = extracted.get("departure_city_code", "").strip()
        arrival = extracted.get("arrival_city_code", "").strip()
        date = extracted.get("travel_date", "")
        date = parse(date, fuzzy=True).strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Error extracting flight info: {e}")
        return {
            "messages": [AIMessage(content="⚠️ Could not extract flight info. Please try again.")],
            "next_agent": "end"
        }

    missing = []
    if not departure:
        missing.append("departure (IATA code)")
    if not arrival:
        missing.append("arrival (IATA code)")
    if not date:
        missing.append("date (YYYY-MM-DD)")

    if missing:
        msg = f"✈️ Could you please provide the following information: {', '.join(missing)}?"
        return {
            "messages": [AIMessage(content=msg)],
            "next_agent": "end",  
        }
    #print(f"Flight Agent: Extracted - Departure: {departure}, Arrival: {arrival}, Date: {date}")
    api_results = get_flights_data(departure, arrival, date)

    # format_prompt = f"""
    # You're a helpful agent. Format this flight API result into 2-3 line summary for user:\n{api_results}
    # """
    # formatted = llm.invoke([HumanMessage(content=format_prompt)])
    summary = "\n\n".join(
    [
        f"✈️ Price: {r['price']}\n"
        f"🕒 Total Duration: {r['duration']}\n"
        f"🛬 Flight Segments:\n" +
        "\n".join(
            [
                f"  • {seg['from']} → {seg['to']} | Airline: {seg['airline']} {seg['flightNumber']} | "
                f"Departs: {datetime.fromisoformat(seg['departureTime']).strftime('%b %d, %I:%M %p')} | "
                f"Arrives: {datetime.fromisoformat(seg['arrivalTime']).strftime('%b %d, %I:%M %p')}"
                for seg in r['segments']
            ]
        )
        for r in api_results
    ]
    )   

    return {
        "messages": [AIMessage(content="Flight Agent: Here are your flight options:")],
        "flight_data": summary,
        "next_agent": "TripAgent"
    }

def HotelAgent(State):
    """Agent responsible for booking hotels."""
    hotel_data = State.get("hotel_data", "")
    task = State.get("user_query", "")
    # print("*TASK*\n",task)
    prompt = f"""
        You are an intelligent Hotel Booking Agent, specializing in extracting crucial details for hotel reservations using the Amadeus API.

        **Your primary task is to extract the following information from the user's travel plan:**

        * **destination_city_code (IATA):** The 3-letter IATA code for the *city center or general city area* where the hotel is desired (e.g., "TYO" for Tokyo, "PAR" for Paris, not airport codes like "NRT" or "CDG" unless explicitly requested by the user for airport hotels).
        * **Check-in Date:** The desired check-in date in 'YYYY-MM-DD' format.
        * **Check-out Date:** The desired check-out date in 'YYYY-MM-DD' format.
        * **Number of Adults:** The number of adult guests.

        **Here's the current context:**

        * **User Travel Plan (Task):** {task}
        * **Previous Known Hotel Data:** {hotel_data}

        **Key Extraction Rules and Logic:**

        1.  **Destination City Code:** Extract the IATA city code for the hotel destination.
        2.  **Year Handling:** If a year is not explicitly mentioned for the check-in or check-out date, **always assume the current year is {CURRENT_YEAR}**. 
        3.  **Check-out Date Calculation:**
            * If the check-out date is **NOT** explicitly provided, but a **duration of stay** (e.g., "for 5 days", "for a week") is mentioned, calculate the check-out date by adding the duration to the check-in date.
            * If neither the check-out date nor the duration is specified, you **MUST leave "checkout_date" as `null`**. Do NOT guess or default to a single night unless explicitly told to.
        4.  **Number of Adults:**
            * Extract the specified number of adults.
            * If the number of adults is **NOT** specified in the `task`, **default to `1` adult**.

        **Example of Expected JSON Output:**
        ```json
        {{
        "city_code": "CDG",
        "checkin_date": "2025-12-19",
        "checkout_date": "2026-01-08",
        "adults": 1
        }}
        **Strictly** return only the JSON do not include any other text or explanation."""
    try:
        parsed = llm.invoke([HumanMessage(content=prompt)])
        match = re.search(r"\{[\s\S]*\}", parsed.content.strip())
        if not match:
            raise ValueError("No JSON found in LLM response")
        extracted = json.loads(match.group(0))
        #extracted = json.loads(parsed.content)  
        city_code = extracted.get("city_code", "").strip()
        checkin_date = extracted.get("checkin_date", "")
        checkin_date = parse(checkin_date, fuzzy=True).strftime("%Y-%m-%d")
        checkout_date = extracted.get("checkout_date", "")
        checkout_date = parse(checkout_date, fuzzy=True).strftime("%Y-%m-%d")
        adults = extracted.get("adults", 1)
    except Exception as e:
        print(f"Error extracting hotel info: {e}")
        return {
            "messages": [AIMessage(content="⚠️ Could not extract hotel info. Please try again.")],
            "next_agent": "end"
        }
    missing = []
    if not city_code:
        missing.append("City code ")
    if not checkin_date:
        missing.append("Checkin date")
    if not checkout_date:
        missing.append("Checkout date")

    if missing:
        msg = f"✈️ Could you please provide the following information: {', '.join(missing)}?"
        return {
            "messages": [AIMessage(content=msg)],
            "next_agent": "end",  
        }
    #print(f"Hotel Agent: Extracted - City: {city_code}, Checkin: {checkin_date}, Checkout: {checkout_date}, Adults: {adults}")
    api_results = get_hotels_data(city_code, checkin_date,checkout_date, adults)
    # format_prompt = f"""
    # You're a helpful agent. Format this flight API result into 2-3 line summary for user:\n{api_results}
    # """
    # formatted = llm.invoke([HumanMessage(content=format_prompt)])
    #print("Hotel API Results:", api_results)
    summary = "\n".join(
    [
        f"🏨 {r['hotel']['name']} - {r['offers'][0]['price']['total']} {r['offers'][0]['price']['currency']}"
        for r in api_results if 'hotel' in r and 'offers' in r and r['offers']
    ]
    )
    
    return {"messages": [AIMessage(content="Hotel Agent: Hotel Information:")],
            "hotel_data": summary,
            "next_agent": "TripAgent"}

def WeatherAgent(State):
    """Agent responsible for checking the weather."""
    task = State.get("user_query", "")
    # print("*TASK*\n",task)
    weather_prompt = f"""You are a Weather Agent. 
    Your task is to gather information and provide weather updates based on the following data:
    Extract the desitation from the task and forecast the weather for the whole trip.
    task: {task}
    return the response in a concise and readable format."""
    report_response = llm.invoke([HumanMessage(content=weather_prompt)])
    response = report_response.content
    return {"messages": [AIMessage(content="Weather Agent: Weather Information:")],
            "weather_data": response,
            "next_agent": "TripAgent"}
    
def RestaurantAgent(State):
    """Agent responsible for finding restaurants."""
    task = State.get("user_query", "")
    # print("*TASK*\n",task)
    prompt = f"""
    You are a Restaurant Agent. Extract the **destination city** from the user's travel plan.

    Return only the **destination city name** (e.g., Paris). Do not include the origin or any other text.
    e.g input: Plan a trip from mumbai to Paris for 5 days. output: Paris
    
    STRICTLY MUST Return just the city name (like: Paris, Rome, Tokyo). No other words or explanation.

    If no destination city is found, return just this word: None
    Travel Plan: {task}
    """

    city = llm.invoke([HumanMessage(content=prompt)]).content
    #print("🧠 Extracted city:", city)
    lat, lon = get_city_coordinates(city)
    missing = []
    if not lat:
        missing.append("Latitude of destination ")
    if not lon:
        missing.append("Longitude of destination ")

    if missing:
        msg = f"✈️ Could you please provide the following information: {', '.join(missing)}?"
        return {
            "messages": [AIMessage(content=msg)],
            "next_agent": "end",  
        }

    results = get_restaurants_data(lat, lon)

    if "error" in results:
        return {
            "messages": [AIMessage(content=f"Error fetching restaurant data: {results['error']}")],
            "restaurant_data": "",
            "next_agent": "end"
        }

    summary = "\n".join(
        [f"{r['name']} ({r['cuisine']}) - Address: {r['address']}" for r in results]
    )

    return {
        "messages": [
            AIMessage(content="Restaurant Agent: Nearby Options:")
        ],
        "restaurant_data": summary,
        "next_agent": "TripAgent"
    }


def ActivityAgent(State):
    """Agent responsible for finding activities."""
    task = State.get("user_query", "")
    flight_data = State.get("flight_data", "")
    activities_data = State.get("activities_data", "")
    hotel_data = State.get("hotel_data", "")
    restaurant_data = State.get("restaurant_data", "")
    weather_data = State.get("weather_data", "")
    activity_prompt = f"""You are an Activities Agent.
    Your task is to gather information and recommend activities that can be done around the destination based on the following data:
    task: {task}
    activities_data: {activities_data}
    consider the following data for better recommendations:
    hotel_data: {hotel_data}
    restaurant_data: {restaurant_data}
    weather_data: {weather_data}
    return the response in a concise and readable format."""

    activity_prompt = f"""You are an expert local guide and itinerary planner AI. Your goal is to create a personalized, practical, and enjoyable itinerary based on the user's profile and real-time data.

    **1. Traveler Profile:**
    {task}

    **2. Previous activities data:**
    {activities_data}

    **3. Contextual Data that can be considered:**
    * **Weather:** {weather_data}
    * **Hotel Info:** {hotel_data}
    * **Nearby Restaurants:** {restaurant_data}

    **Your Task:**
    Based on all the data above, generate a step-by-step itinerary for the specified duration.
    MUST ONLY INCLUDE activities that can be done in the destination city.

    **Output Requirements:**
    * Organize the itinerary chronologically (e.g., by Morning, Afternoon, Evening).
    * For each suggested activity, provide a brief "Why this fits your trip" justification, linking back to the traveler's interests, the weather, or the time of day.
    * Incorporate dining suggestions from the restaurant data at appropriate times (e.g., lunch, dinner).
    * Be realistic about timing and suggest a logical flow of events.
    * Format the response using Markdown for clear, readable sections.
    """

    report_response = llm.invoke([HumanMessage(content=activity_prompt)])
    response = report_response.content
    
    final_prompt = f"""
        You are a highly efficient trip planning assistant. Your task is to synthesize all available travel information into a clear, concise, and easy-to-read summary for the user.

        **Here is the complete trip data you have gathered:**

        * **Flight Details:** {flight_data}
        * **Hotel Accommodation:** {hotel_data}
        * **Restaurant Reservations:** {restaurant_data}
        * **Weather Forecast:** {weather_data}
        * **Planned Activities:** {response}

        **Your goal is to generate a comprehensive trip itinerary summary. Ensure the summary is:**

        1.  **Concise:** Get straight to the point, avoiding unnecessary jargon or overly verbose descriptions.
        2.  **Readable:** Use clear, straightforward language. Employ formatting (like bullet points, bold text) where it enhances readability.
        3.  **Comprehensive:** Cover all key aspects of the trip:
            * **Travel Dates & Destinations:** Clearly state when and where the trip is happening.
            * **Flight Information:** Include departure and arrival details.
            * **Accommodation:** Mention hotel name and dates.
            * **Dining:** Summarize any restaurant bookings.
            * **Weather Outlook:** Provide a brief overview of the expected weather.
            * **Planned Activities/Itinerary Highlights:** List the main activities or key events.
        4. Add **TIPS SECTION** for travelers like do's and don'ts, local customs, or any other relevant information that can enhance the travel experience.
        5. ** IN THE END** MUST Provide the total cost estimation of the trip in origin(from where the user is traveling) currency. [Include all costs like flight, hotel, food, activities, etc.]
        6. **Final Touch:** Conclude with a friendly note wishing the user a great trip.
        7. **Make sure you're including all the information provided by each agent so that the user has a complete overview of their trip.** (e.g provide all the flights and hotels available)

        **Focus on presenting the information in a logical flow, making it easy for the user to quickly grasp their entire trip at a glance.**
        """
    final_data = llm2.invoke([HumanMessage(content=final_prompt)]).content
    return {"messages": [AIMessage(content=f"\n {final_data}")],
            "activities_data": response,
            "next_agent": "TripAgent",
            "final_data": final_data,
            "task_complete": True}
    
def router(State) -> Literal["TripAgent", "FlightAgent", "HotelAgent", "WeatherAgent", "RestaurantAgent", "ActivityAgent", "_end_"]:
    """Routes to next agent based on state."""
    
    next_agent = State.get("next_agent", TripAgent)
    if next_agent == "end" or State.get("task_complete", False):
        return END
    elif next_agent in ["TripAgent", "FlightAgent", "HotelAgent", "WeatherAgent", "RestaurantAgent", "ActivityAgent"]:
        return next_agent
    return TripAgent

workflow = StateGraph(State)
#nodes
workflow.add_node("TripAgent", TripAgent)
workflow.add_node("FlightAgent", FlightAgent)
workflow.add_node("HotelAgent", HotelAgent) 
workflow.add_node("WeatherAgent", WeatherAgent)
workflow.add_node("RestaurantAgent", RestaurantAgent)
workflow.add_node("ActivityAgent", ActivityAgent)
#edges
workflow.set_entry_point("TripAgent")
for node in ["TripAgent", "FlightAgent", "HotelAgent", "WeatherAgent", "RestaurantAgent", "ActivityAgent"]:
    workflow.add_conditional_edges(
        node, 
        router,
        {
            "TripAgent":"TripAgent",
            "FlightAgent": "FlightAgent",
            "HotelAgent": "HotelAgent",
            "WeatherAgent": "WeatherAgent",
            "RestaurantAgent": "RestaurantAgent",
            "ActivityAgent": "ActivityAgent",
            END: END
        } 
        )
graph = workflow.compile()


# from PIL import Image
# import io
# png_data = graph.get_graph().draw_mermaid_png()
# with open("workflow.png", "wb") as f:
#     f.write(png_data)
# img = Image.open(io.BytesIO(png_data))
# img.show() 