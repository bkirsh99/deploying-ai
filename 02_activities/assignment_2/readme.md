# TripWise: A Weather-Aware Travel Chat Assistant

## Overview

TripWise is a conversational travel-planning assistant built with Gradio. It helps users prepare for trips by combining live weather information, semantic retrieval over a travel knowledge base, and AI-generated itineraries to provide concise, actionable travel advice.

---

# Services

## Service 1: Weather API

TripWise uses the Weatherstack API to retrieve current weather conditions for a destination.

The raw JSON output is converted into a user-friendly, natural-language summary including:

* temperature
* feels-like temperature
* weather conditions
* humidity
* wind speed
* precipitation
* UV index

---

## Service 2: Semantic Search

TripWise includes a semantic search service using a persistent ChromaDB database.

Travel knowledge is stored in:

```
data/travel_knowledge.csv
```

Each document is embedded using OpenAI's `text-embedding-3-small` embedding model and stored inside a persistent ChromaDB collection.

### Embedding process:

1. Read each row from the travel knowledge dataset.
2. Combine the title and content into a document.
3. Generate embeddings using OpenAI embeddings.
4. Store documents, metadata, and embeddings in a persistent ChromaDB collection.

When a user asks for packing advice, destination information, or travel recommendations, the user's query is embedded and compared against the stored vector database. The most relevant documents are retrieved and supplied to the language model to generate the final response.

---

## Service 3: Function Calling

TripWise uses OpenAI Function Calling to build structured travel itineraries.

When the user requests an itinerary, the language model invokes a function that generates structured plans for each day of the trip.

These structured outputs are then converted into a natural conversational itinerary.

Inputs include:

* destination
* travel dates
* transportation method
* interests
* weather summary

The function produces morning, afternoon, and evening activities for each day.

---

# Chat Interface

TripWise uses a Gradio chat interface.

The assistant has a friendly, encouraging personality designed to feel like an experienced travel companion.

The interface maintains short-term conversational memory using Gradio `State`. During a conversation it remembers information such as:

* destination
* travel dates
* transportation preferences
* user interests
* previous requests

This allows users to ask follow-up questions without repeating previously provided trip information.

---

# Guardrails

TripWise includes several guardrails.

The assistant refuses requests attempting to:

* reveal the hidden system prompt
* modify the system prompt
* ignore previous instructions

The application also blocks conversations on the assignment's required restricted topics:

* cats or dogs
* horoscopes or zodiac signs
* Taylor Swift

When these topics are detected, the assistant politely declines and redirects the conversation toward travel planning.

---

# Files

```
05_src/assignment_chat/
├── assignment_2.ipynb
├── readme.md
├── data/
│   └── travel_knowledge.csv
└── chroma_db/
```

---

# Setup

Set the required environment variables:

```
OPENAI_API_KEY="your_openai_key"
WEATHERSTACK_API_KEY="your_weatherstack_key"
```

Build the vector database:

```
python build_vector_db.py
```

Launch the application:

```
python app.py
```

---

# Example Prompts

```
What is the weather in Barcelona?
```

```
I'm travelling to Tokyo for five days. Make me a packing list.
```

```
Now create a day-by-day itinerary.
```

```
Will I need an umbrella?
```

---

# Design Decisions

The three services were designed to work together within a single travel-planning workflow rather than existing as unrelated features.

Weatherstack provides live environmental information, ChromaDB retrieves relevant travel knowledge, and Function Calling produces structured itineraries. The language model combines these components into natural conversational responses while maintaining context across the conversation.
