# Memory-Enhanced CLI Chatbot - Fixed Version with OpenRouter
# requirements.txt contents:
"""
chromadb==0.4.18
pydantic==2.5.0
python-dotenv==1.0.0
httpx==0.25.2
rich==13.7.0
openai==1.3.0
"""

import os
import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
from rich.live import Live

# Load environment variables
load_dotenv()

# Configure rich console
console = Console()

# Configure minimal logging (only to file, no emojis)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS
# =============================================================================

class MemoryCategory(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    HABIT = "habit"
    GOAL = "goal"
    RELATIONSHIP = "relationship"
    EXPERIENCE = "experience"
    SKILL = "skill"
    OTHER = "other"

@dataclass
class Memory:
    id: str
    user_id: str
    content: str
    category: MemoryCategory
    confidence: float
    timestamp: datetime
    conversation_id: str
    metadata: Dict[str, Any]
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['category'] = MemoryCategory(data['category'])
        return cls(**data)

@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

# =============================================================================
# MODULE 1: MEMORY EXTRACTION ENGINE
# =============================================================================

class MemoryExtractor:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = "deepseek/deepseek-r1-0528-qwen3-8b:free"
        self.extraction_prompt = """
Analyze the following conversation and extract meaningful memories about the user.
Focus on:
- Personal preferences and opinions
- Important facts about the user
- Habits and routines
- Goals and aspirations
- Relationships and social information
- Experiences and events
- Skills and abilities

For each memory, provide:
1. The memory content (what should be remembered)
2. Category (preference, fact, habit, goal, relationship, experience, skill, other)
3. Confidence score (0.0-1.0, how certain you are this is worth remembering)

Return ONLY a JSON array of memories in this format:
[
  {
    "content": "User prefers dark chocolate over milk chocolate",
    "category": "preference",
    "confidence": 0.9
  }
]

If no meaningful memories can be extracted, return an empty array: []

Conversation to analyze:
"""

    async def extract_memories(self, messages: List[Message], user_id: str, conversation_id: str) -> List[Memory]:
        logger.info(f"Starting memory extraction for user {user_id}")
        
        try:
            # Format conversation for analysis
            conversation_text = "\n".join([
                f"{msg.role}: {msg.content}" for msg in messages
            ])
            
            logger.info("Sending conversation to OpenRouter for memory analysis")
            
            # Call OpenRouter API using OpenAI client
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.extraction_prompt + conversation_text
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                if content.startswith('```json'):
                    content = content.split('```json')[1].split('```')[0]
                elif content.startswith('```'):
                    content = content.split('```')[1].split('```')[0]
                
                memories_data = json.loads(content)
                
                # Convert to Memory objects
                memories = []
                for mem_data in memories_data:
                    if mem_data.get('confidence', 0) >= 0.5:  # Filter low confidence
                        memory = Memory(
                            id=str(uuid.uuid4()),
                            user_id=user_id,
                            content=mem_data['content'],
                            category=MemoryCategory(mem_data['category']),
                            confidence=mem_data['confidence'],
                            timestamp=datetime.now(timezone.utc),
                            conversation_id=conversation_id,
                            metadata={}
                        )
                        memories.append(memory)
                
                logger.info(f"Extracted {len(memories)} new memories")
                for i, mem in enumerate(memories, 1):
                    logger.info(f"  {i}. [{mem.category.value}] {mem.content} (confidence: {mem.confidence:.2f})")
                
                return memories
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse memory extraction response: {e}")
                logger.error(f"Content: {content}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return []

# =============================================================================
# MODULE 2: MEMORY STORAGE MANAGER
# =============================================================================

class MemoryStorage:
    def __init__(self, persist_directory: str = "./chroma_db"):
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="user_memories",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB initialized successfully")
    
    async def store_memory(self, memory: Memory) -> str:
        try:
            logger.info(f"Storing memory: {memory.content[:50]}...")
            
            # Store the memory
            self.collection.add(
                documents=[memory.content],
                metadatas=[{
                    "user_id": memory.user_id,
                    "category": memory.category.value,
                    "confidence": memory.confidence,
                    "timestamp": memory.timestamp.isoformat(),
                    "conversation_id": memory.conversation_id,
                    "metadata": json.dumps(memory.metadata)
                }],
                ids=[memory.id]
            )
            
            logger.info(f"Memory stored successfully with ID: {memory.id}")
            return memory.id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        try:
            result = self.collection.get(ids=[memory_id])
            if not result['documents']:
                return None
                
            doc = result['documents'][0]
            meta = result['metadatas'][0]
            
            memory = Memory(
                id=memory_id,
                user_id=meta['user_id'],
                content=doc,
                category=MemoryCategory(meta['category']),
                confidence=meta['confidence'],
                timestamp=datetime.fromisoformat(meta['timestamp']),
                conversation_id=meta['conversation_id'],
                metadata=json.loads(meta.get('metadata', '{}'))
            )
            
            return memory
            
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            return None

    async def delete_memory(self, memory_id: str) -> bool:
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False

    async def get_all_user_memories(self, user_id: str, limit: int = 50) -> List[Memory]:
        try:
            # Get all memories for user
            results = self.collection.get(
                where={"user_id": user_id},
                limit=limit
            )
            
            memories = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    meta = results['metadatas'][i]
                    memory_id = results['ids'][i]
                    
                    memory = Memory(
                        id=memory_id,
                        user_id=meta['user_id'],
                        content=doc,
                        category=MemoryCategory(meta['category']),
                        confidence=meta['confidence'],
                        timestamp=datetime.fromisoformat(meta['timestamp']),
                        conversation_id=meta['conversation_id'],
                        metadata=json.loads(meta.get('metadata', '{}'))
                    )
                    memories.append(memory)
            
            # Sort by timestamp (newest first)
            memories.sort(key=lambda x: x.timestamp, reverse=True)
            return memories
            
        except Exception as e:
            logger.error(f"Error getting user memories: {e}")
            return []

# =============================================================================
# MODULE 3: MEMORY RETRIEVAL SYSTEM
# =============================================================================

class MemoryRetriever:
    def __init__(self, storage: MemoryStorage):
        self.storage = storage
    
    async def retrieve_memories(self, query: str, user_id: str, limit: int = 5, 
                              min_confidence: float = 0.5) -> List[Memory]:
        try:
            logger.info(f"Searching for relevant memories for query: '{query[:50]}...'")
            
            # First get all user memories, then filter by confidence
            all_memories = await self.storage.get_all_user_memories(user_id, limit=100)
            
            # Filter by confidence
            confident_memories = [m for m in all_memories if m.confidence >= min_confidence]
            
            if not confident_memories:
                logger.info("No memories found matching criteria")
                return []
            
            # For now, return most recent memories that match confidence
            # In a production system, you'd want to do semantic similarity here
            recent_memories = confident_memories[:limit]
            
            logger.info(f"Found {len(recent_memories)} relevant memories")
            for i, mem in enumerate(recent_memories, 1):
                logger.info(f"  {i}. [{mem.category.value}] {mem.content}")
            
            return recent_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

# =============================================================================
# MODULE 4: MEMORY UPDATE/DELETE HANDLER
# =============================================================================

class MemoryUpdater:
    def __init__(self, storage: MemoryStorage, api_key: str):
        self.storage = storage
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = "deepseek/deepseek-r1-0528-qwen3-8b:free"
    
    async def update_memory(self, update_instruction: str, user_id: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing memory update: '{update_instruction}'")
            
            # First, identify which memories need updating
            relevant_memories = await self._find_memories_to_update(update_instruction, user_id)
            
            if not relevant_memories:
                logger.warning("No relevant memories found to update")
                return {
                    "success": False,
                    "message": "No relevant memories found to update",
                    "updated_memories": []
                }
            
            updated_ids = []
            for memory in relevant_memories:
                logger.info(f"Updating memory: {memory.content[:50]}...")
                
                # Delete old memory
                await self.storage.delete_memory(memory.id)
                
                # Create updated memory
                updated_content = await self._generate_updated_content(
                    memory.content, update_instruction
                )
                
                if updated_content:
                    updated_memory = Memory(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        content=updated_content,
                        category=memory.category,
                        confidence=memory.confidence,
                        timestamp=datetime.now(timezone.utc),
                        conversation_id=memory.conversation_id,
                        metadata=memory.metadata
                    )
                    
                    await self.storage.store_memory(updated_memory)
                    updated_ids.append(updated_memory.id)
                    logger.info(f"Updated to: {updated_content}")
            
            logger.info(f"Successfully updated {len(updated_ids)} memories")
            return {
                "success": True,
                "message": f"Updated {len(updated_ids)} memories",
                "updated_memories": updated_ids
            }
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return {
                "success": False,
                "message": f"Error updating memory: {str(e)}",
                "updated_memories": []
            }
    
    async def _find_memories_to_update(self, instruction: str, user_id: str) -> List[Memory]:
        # Use semantic search to find relevant memories
        retriever = MemoryRetriever(self.storage)
        return await retriever.retrieve_memories(instruction, user_id, limit=3)
    
    async def _generate_updated_content(self, original_content: str, update_instruction: str) -> Optional[str]:
        try:
            prompt = f"""
Update the following memory based on the instruction:

Original memory: {original_content}
Update instruction: {update_instruction}

Provide only the updated memory content, nothing else:
"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
                    
        except Exception as e:
            logger.error(f"Error generating updated content: {e}")
            return None

# =============================================================================
# MODULE 5: CONVERSATION INTEGRATION LAYER
# =============================================================================

class ConversationManager:
    def __init__(self, extractor: MemoryExtractor, retriever: MemoryRetriever, 
                 storage: MemoryStorage, updater: MemoryUpdater, api_key: str):
        self.extractor = extractor
        self.retriever = retriever
        self.storage = storage
        self.updater = updater
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = "deepseek/deepseek-r1-0528-qwen3-8b:free"
    
    async def process_conversation(self, messages: List[Message], 
                                 user_id: str, conversation_id: str = None) -> Dict[str, Any]:
        try:
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            logger.info(f"Processing conversation for user {user_id}")
            
            # Get relevant memories for context
            latest_user_message = [msg for msg in messages if msg.role == "user"][-1]
            relevant_memories = await self.retriever.retrieve_memories(
                latest_user_message.content, user_id, limit=5
            )
            
            # Generate response with memory context
            logger.info("Generating response with memory context")
            response = await self._generate_response_with_memories(
                messages, relevant_memories
            )
            
            # Extract new memories from the conversation
            logger.info("Extracting new memories from conversation")
            new_memories = await self.extractor.extract_memories(
                messages, user_id, conversation_id
            )
            
            # Store new memories
            for memory in new_memories:
                await self.storage.store_memory(memory)
            
            logger.info("Conversation processing completed")
            
            return {
                "response": response,
                "memories_used": [mem.to_dict() for mem in relevant_memories],
                "memories_extracted": [mem.to_dict() for mem in new_memories]
            }
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your message.",
                "memories_used": [],
                "memories_extracted": []
            }
    
    async def _generate_response_with_memories(self, messages: List[Message], 
                                             memories: List[Memory]) -> str:
        try:
            # Prepare context with memories
            memory_context = ""
            if memories:
                memory_context = "Relevant information about the user:\n"
                for memory in memories:
                    memory_context += f"- {memory.content}\n"
                memory_context += "\nUse this information naturally in your response when relevant.\n\n"
                logger.info(f"Using {len(memories)} memories for context")
            else:
                logger.info("No relevant memories found for context")
            
            # Prepare conversation messages
            conversation_messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful AI assistant with access to information about the user. 
{memory_context}
Be natural and conversational. Only reference the user information when it's relevant to the conversation."""
                }
            ]
            
            # Add conversation messages
            for msg in messages:
                conversation_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Call OpenRouter API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=conversation_messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            generated_response = response.choices[0].message.content
            logger.info("Response generated successfully")
            return generated_response
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response."

# =============================================================================
# CLI INTERFACE
# =============================================================================

class MemoryChatbotCLI:
    def __init__(self):
        self.console = Console()
        
        # Initialize system components
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            self.console.print("[bold red]Error: OPENROUTER_API_KEY environment variable not found![/bold red]")
            self.console.print("Please create a .env file with your OpenRouter API key:")
            self.console.print("OPENROUTER_API_KEY=your_api_key_here")
            exit(1)
        
        self.console.print("[dim]Initializing memory system...[/dim]")
        
        self.storage = MemoryStorage()
        self.extractor = MemoryExtractor(api_key)
        self.retriever = MemoryRetriever(self.storage)
        self.updater = MemoryUpdater(self.storage, api_key)
        self.conversation_manager = ConversationManager(
            self.extractor, self.retriever, self.storage, self.updater, api_key
        )
        
        self.user_id = "default_user"  # You can modify this to handle multiple users
        self.conversation_history = []
        
    def show_welcome(self):
        welcome_panel = Panel.fit(
            Text("🤖 Memory-Enhanced Chatbot\n\nI'll remember our conversations and learn about you!", 
                 style="bold blue", justify="center"),
            title="Welcome",
            border_style="blue"
        )
        self.console.print(welcome_panel)
        self.console.print("\n[bold cyan]Commands:[/bold cyan]")
        self.console.print("  - Type [bold]'quit'[/bold] or [bold]'exit'[/bold] to end")
        self.console.print("  - Type [bold]'memories'[/bold] to view stored memories")
        self.console.print("  - Type [bold]'update: <instruction>'[/bold] to update memories")
        self.console.print("  - Type [bold]'clear'[/bold] to clear conversation history\n")
    
    async def show_memories(self):
        self.console.print("\n[bold cyan]Your Stored Memories:[/bold cyan]")
        
        memories = await self.storage.get_all_user_memories(self.user_id, limit=20)
        
        if not memories:
            self.console.print("   No memories stored yet.")
            return
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", width=12)
        table.add_column("Memory", style="white", width=60)
        table.add_column("Confidence", style="green", width=10)
        table.add_column("Date", style="dim", width=16)
        
        for memory in memories[:10]:  # Show last 10 memories
            date_str = memory.timestamp.strftime("%m/%d %H:%M")
            confidence_str = f"{memory.confidence:.2f}"
            
            table.add_row(
                memory.category.value,
                memory.content[:60] + "..." if len(memory.content) > 60 else memory.content,
                confidence_str,
                date_str
            )
        
        self.console.print(table)
        
        if len(memories) > 10:
            self.console.print(f"\n... and {len(memories) - 10} more memories")
    
    async def update_memories(self, instruction: str):
        self.console.print(f"\n[yellow]Updating memories: {instruction}[/yellow]")
        
        with self.console.status("[bold green]Processing memory update...", spinner="dots"):
            result = await self.updater.update_memory(instruction, self.user_id)
        
        if result['success']:
            self.console.print(f"[green]{result['message']}[/green]")
        else:
            self.console.print(f"[red]{result['message']}[/red]")
    
    async def chat_loop(self):
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    self.console.print("[green]Goodbye! Your memories have been saved.[/green]")
                    break
                
                if user_input.lower() == 'memories':
                    await self.show_memories()
                    continue
                
                if user_input.lower().startswith('update:'):
                    instruction = user_input[7:].strip()
                    await self.update_memories(instruction)
                    continue
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    self.console.print("[yellow]Conversation history cleared.[/yellow]")
                    continue
                
                if not user_input:
                    continue
                
                # Add user message to history
                user_message = Message(
                    role="user",
                    content=user_input,
                    timestamp=datetime.now(timezone.utc)
                )
                self.conversation_history.append(user_message)
                
                # Process conversation with loading indicator
                with self.console.status("[bold green]Thinking...", spinner="dots"):
                    result = await self.conversation_manager.process_conversation(
                        self.conversation_history, self.user_id
                    )
                
                # Display response
                response_text = result['response']
                
                response_panel = Panel(
                    Text(response_text, style="white"),
                    title="🤖 Assistant",
                    border_style="green",
                    padding=(1, 2)
                )
                self.console.print(response_panel)
                
                # Add assistant response to history
                assistant_message = Message(
                    role="assistant",
                    content=response_text,
                    timestamp=datetime.now(timezone.utc)
                )
                self.conversation_history.append(assistant_message)
                
                # Show memory stats if any were used or extracted
                memories_used = len(result['memories_used'])
                memories_extracted = len(result['memories_extracted'])
                
                if memories_used > 0 or memories_extracted > 0:
                    stats_text = f"Used {memories_used} memories, Extracted {memories_extracted} new memories"
                    self.console.print(f"[dim]{stats_text}[/dim]")
                
            except KeyboardInterrupt:
                self.console.print("\n\n[green]Goodbye! Your memories have been saved.[/green]")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                self.console.print(f"[red]An error occurred: {e}[/red]")
    
    async def run(self):
        self.show_welcome()
        await self.chat_loop()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

async def main():
    # Create .env file template if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("OPENROUTER_API_KEY=your_openrouter_api_key_here\n")
        console.print("[yellow]Created .env file template. Please add your OpenRouter API key.[/yellow]")
        return
    
    # Start the CLI chatbot
    chatbot = MemoryChatbotCLI()
    await chatbot.run()

if __name__ == "__main__":
    asyncio.run(main())