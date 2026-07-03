# TripWise: Weather-Aware Travel Chat Assistant

TripWise is a conversational travel-planning assistant. It helps users plan trips by connecting live weather, semantic travel knowledge, and structured itinerary generation.

The assistant has a friendly, practical personality: it sounds like a calm travel-planning friend who gives concrete suggestions without overwhelming the user.

## Services

### Service 1: API Calls — Weatherstack Weather Lookup

TripWise uses the Weatherstack API as its external API service.

The user can ask for weather in a destination, such as:

> What is the weather in Lisbon?

The app calls Weatherstack's current weather endpoint and transforms the JSON response into natural language. It does not return the API output verbatim. Instead, it summarizes temperature, weather condition, humidity, wind, precipitation, and UV index in a user-friendly way.

Environment variable required:

Add to nano 05_src/.secrets:
```text
WEATHERSTACK_API_KEY="your_weatherstack_key"
```

### Service 2: Semantic Query — ChromaDB Travel Knowledge Search

TripWise uses a small custom travel knowledge dataset stored in:

```text
data/travel_knowledge.csv
```

The dataset contains packing, weather, transportation, and itinerary advice. The documents are embedded with OpenAI `text-embedding-3-small` and stored in a persistent local ChromaDB database:

```text
chroma_db/
```

To build the persistent ChromaDB index, run:

```python
build_vector_db()
```

The app then performs semantic search over the persistent ChromaDB collection whenever it needs relevant packing or itinerary context.

Embedding process:

1. Read rows from `data/travel_knowledge.csv`.
2. Combine each row's title and content into a document.
3. Embed each document using OpenAI embeddings.
4. Store documents, embeddings, and metadata in a persistent ChromaDB collection.

The dataset is intentionally small and under the assignment's file-size limit.

### Service 3: Function Calling — Day-by-Day Itinerary Builder

TripWise uses OpenAI function calling to create structured itinerary components.

The app defines a function:

```python
make_day_plan(day, location, interests, transport, weather_summary_text)
```

For itinerary requests, the model calls this function once per trip day. The function returns a structured morning, afternoon, and evening plan. The assistant then converts those tool outputs into a polished day-by-day itinerary.

Example user request:

> I am going to Porto for 3 days. I like food, museums, and walking. Make me an itinerary.

## Chat Interface

The app uses Gradio and provides a chat-based interface.

Run the app with:

```bash
python app.py
```

The interface maintains short-term memory using Gradio state. It remembers trip details such as:

- location
- number of days
- interests
- transportation mode
- requested service

For example, a user can first say:

> I am going to Tokyo for 5 days.

Then later say:

> Make me a packing list.

The assistant will remember Tokyo and 5 days during the current chat session.

## Guardrails

TripWise includes guardrails that prevent users from:

- accessing or revealing the system prompt
- modifying the system prompt directly
- asking about restricted assignment topics

Restricted topics:

- cats or dogs
- horoscopes or zodiac signs
- Taylor Swift

If the user asks about a restricted topic, TripWise politely refuses and redirects back to travel planning.

## Files

```text
05_src/assignment_chat/
├── assignment_2.ipynb
├── readme.md
└── data/
    └── travel_knowledge.csv
```

## Setup

From the `05_src/assignment_chat` folder:

```bash
export OPENAI_API_KEY="your_openai_key"
export WEATHERSTACK_API_KEY="your_weatherstack_key"

python build_vector_db.py
python app.py
```

## Example Prompts

```text
What is the weather in Barcelona?
```

```text
I am going to Lisbon for 4 days, I like museums and food, and I will use public transit. Make me a packing list.
```

```text
Now make me a day-by-day itinerary.
```

## Design Decisions

I chose a connected travel-planning theme so that the three services work together instead of feeling separate.

Weatherstack provides live weather. ChromaDB retrieves relevant travel knowledge. Function calling structures the itinerary. The final assistant response combines these pieces into practical travel advice.

The implementation is intentionally lightweight so it can be shared through GitHub and run in the course environment.
