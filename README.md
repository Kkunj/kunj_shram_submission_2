# Memory-Enhanced Chatbot 

## Summary

This is an intelligent chatbot system that provides **long-term memory capabilities** to AI conversations. The system automatically extracts, stores, and retrieves meaningful information from conversations, allowing the AI to remember user preferences, habits, goals, relationships, and other important details across multiple conversation sessions.

The chatbot uses **ChromaDB** for efficient vector-based memory storage and **OpenRouter API** (OpenAI-compatible) for natural language processing. It can create memories from conversations (e.g., remembering productivity tools you use), retrieve relevant memories for context in new conversations, and update/delete memories when information changes.

## Key Features

-  **Automatic Memory Extraction**: Intelligently identifies and stores important information from conversations
-  **Contextual Memory Retrieval**: Finds relevant memories to provide personalized responses
-  **Memory Updates**: Allows updating or deleting outdated information
-  **Categorized Storage**: Organizes memories into categories (preferences, facts, habits, goals, etc.)
-  **Persistent Storage**: Memories persist across conversation sessions
-  **Rich CLI Interface**: Command-line interface with colors and formatting

## Important Classes

### 1. `Memory` (Data Model)
- **Purpose**: Represents a single memory with metadata
- **Key Attributes**: 
  - `content`: The actual memory text
  - `category`: Type of memory (preference, fact, habit, goal, etc.)
  - `confidence`: How confident the system is about this memory (0.0-1.0)
  - `user_id`: Links memory to specific user
  - `timestamp`: When the memory was created

### 2. `MemoryExtractor`
- **Purpose**: Analyzes conversations and extracts meaningful memories
- **Key Method**: `extract_memories()` - Uses LLM to identify what should be remembered
- **Process**: Sends conversation to AI model with specific prompts to extract structured memory data

### 3. `MemoryStorage`
- **Purpose**: Handles all database operations for storing and retrieving memories
- **Technology**: Uses ChromaDB (vector database) for efficient storage
- **Key Methods**:
  - `store_memory()`: Saves new memories to database
  - `get_all_user_memories()`: Retrieves all memories for a user
  - `delete_memory()`: Removes specific memories

### 4. `MemoryRetriever`
- **Purpose**: Finds relevant memories based on current conversation context
- **Key Method**: `retrieve_memories()` - Searches for memories relevant to user's current message
- **Strategy**: Currently uses recent + confidence-based filtering (can be enhanced with semantic search)

### 5. `MemoryUpdater`
- **Purpose**: Handles updating or deleting existing memories
- **Key Method**: `update_memory()` - Processes instructions like "I don't use X anymore"
- **Process**: Finds relevant memories, generates updated content, replaces old memories

### 6. `ConversationManager`
- **Purpose**: Orchestrates the entire conversation flow
- **Key Method**: `process_conversation()` - Main pipeline that:
  1. Retrieves relevant memories for context
  2. Generates AI response using memory context
  3. Extracts new memories from the conversation
  4. Stores new memories for future use

### 7. `MemoryChatbotCLI`
- **Purpose**: Provides the command-line interface for user interaction
- **Features**: 
  - Interactive chat loop
  - Memory viewing commands
  - Memory update commands
  - Rich formatting and colors

## Installation and Setup

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Get OpenRouter API Key

1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Create an account and sign in
3. Go to "Keys" section in your dashboard
4. Create a new API key
5. Copy the API key for the next step

### 3. Create Environment File

Create a `.env` file in the project root directory:

```bash
# Create .env file
touch .env
```

Add your API key to the `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

**Important**: Replace `your_api_key_here` with your actual OpenRouter API key.

### 4. Run the Chatbot

```bash
python main.py
```

## Usage Commands

Once the chatbot is running, you can use these commands:

- **Regular Chat**: Just type your message and press Enter
- **View Memories**: Type `memories` to see all stored memories
- **Update Memory**: Type `update: <instruction>` (e.g., "update: I don't use Magnet anymore")
- **Clear History**: Type `clear` to clear current conversation
- **Exit**: Type `quit` or `exit` to close the chatbot

## Output Preview: Example Usage and testing
First chat where user mentioned their preferences ( here the tools that they use )
<img width="949" height="469" alt="image" src="https://github.com/user-attachments/assets/2c3e1d79-64af-4658-a362-5bb35360a2c2" />


Second chat where extracts the pre-build memory for curating the answer.
<img width="948" height="361" alt="image" src="https://github.com/user-attachments/assets/f848855d-b7e0-4b00-9ad3-3e3cf9c167d4" />

Third chat where user mentioned changes in their preferences ( which gets recorded efficiently ) and the memory is effectively updated.
<img width="973" height="774" alt="image" src="https://github.com/user-attachments/assets/ca7fe2cf-49f9-43c9-91e0-2d3cd40cfbfc" />


