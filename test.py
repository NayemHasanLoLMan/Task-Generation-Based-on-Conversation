import os
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import re
from typing import List, Dict, Any
from collections import deque
from datetime import datetime, timedelta
import json


# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
ethical_rules = os.getenv("ETHICAL_RULES", "").strip()

# Define Serverless specifications
serverless_spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1"
)

# Index name
index_name = "the-holy-bible"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=serverless_spec
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Enhanced Task Management Class
class SpiritualTaskManager:
    def __init__(self):
        self.tasks = {}
        self.task_counter = 1
        self.tasks_directory = "spiritual_tasks"
        
        # Create the tasks directory if it doesn't exist
        if not os.path.exists(self.tasks_directory):
            os.makedirs(self.tasks_directory)

    def add_task(self, task_name: str, description: str, subtasks: List[str], due_date: str) -> str:
        """Add a spiritual task with subtasks and save to JSON file"""
        try:
            # Validate due date format (e.g., YYYY-MM-DD)
            datetime.strptime(due_date, "%Y-%m-%d")
            
            task_id = f"task_{self.task_counter}"
            self.task_counter += 1
            
            # Create subtasks dictionary
            subtasks_dict = {}
            for i, subtask in enumerate(subtasks, 1):
                subtasks_dict[f"task{i}"] = subtask
            
            task_data = {
                "task_name": task_name,
                "task_description": description,
                "tasks": subtasks_dict,
                "due_date": due_date,
                "created_at": datetime.now().strftime("%Y-%m-%d")
            }
            
            # Store in memory
            self.tasks[task_id] = task_data
            
            # Save to JSON file
            json_filename = os.path.join(self.tasks_directory, f"{task_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, indent=2)
            
            return f"I've created a spiritual task '{task_name}' to help guide your journey. It's been saved for your reference."
        except ValueError:
            return "Error: Invalid due date format. Please use YYYY-MM-DD."
        except Exception as e:
            print(f"Error saving task to JSON: {str(e)}")
            return f"Spiritual task '{task_name}' created but couldn't be saved to file."

    def get_tasks(self) -> str:
        """Get formatted list of all spiritual tasks"""
        if not self.tasks:
            return "No spiritual tasks have been created yet."
        
        task_list = "## Your Spiritual Journey Tasks\n\n"
        for task_id, info in self.tasks.items():
            task_list += f"### {info['task_name']}\n"
            task_list += f"**Description**: {info['task_description']}\n"
            task_list += f"**Due Date**: {info['due_date']}\n\n"
            
            task_list += "**Spiritual Practices**:\n"
            for subtask_id, subtask in info['tasks'].items():
                task_list += f"- {subtask}\n"
            
            task_list += "\n---\n\n"
            
        return task_list

    def clear_tasks(self) -> str:
        """Clear all tasks"""
        self.tasks.clear()
        self.task_counter = 1
        return "All spiritual tasks have been cleared from your journey."
    
    def to_json(self) -> str:
        """Export tasks as JSON string"""
        return json.dumps(self.tasks, indent=2)

# Conversation memory to store context
class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)
        self.task_manager = SpiritualTaskManager()
        
    def add_interaction(self, user_message, bot_response):
        self.history.append({"user": user_message, "bot": bot_response})
        
    def get_conversation_context(self):
        context = ""
        if self.history:
            context = "Previous conversation:\n"
            for i, interaction in enumerate(self.history):
                context += f"User: {interaction['user']}\n"
                context += f"Bot: {interaction['bot']}\n\n"
        return context
    
    def clear(self):
        self.history.clear()
        self.task_manager.clear_tasks()

# Initialize conversation memory
memory = ConversationMemory(max_history=5)

# Function to embed text using Gemini's `text-embedding-004` model
def embed_text_with_gemini(text: str) -> List[float]:
    if text.strip():
        response = genai.embed_content(
            model='models/text-embedding-004',
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']
    else:
        print("Warning: Skipping empty text.")
        return None

# Function to rerank retrieved passages by relevance
def rerank_passages(passages: List[str], query: str) -> List[str]:
    query_words = set(query.lower().split())
    passage_scores = []
    for passage in passages:
        keyword_matches = sum(1 for word in query_words if word in passage.lower())
        passage_scores.append((passage, keyword_matches))
    return [p for p, _ in sorted(passage_scores, key=lambda x: x[1], reverse=True)]

# Function to retrieve the most relevant context for a query
def retrieve_relevant_documents(query: str, top_k: int = 15) -> List[str]:
    query_embedding = embed_text_with_gemini(query)
    if not query_embedding:
        return []

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    relevant_texts = [match['metadata'].get('text', '') for match in results['matches']]
    reranked_passages = rerank_passages(relevant_texts, query)
    return reranked_passages

# Function to remove duplicate or highly similar content
def deduplicate_content(passages: List[str]) -> List[str]:
    unique_passages = []
    for passage in passages:
        normalized = re.sub(r'\s+', ' ', passage.lower().strip())
        is_duplicate = False
        for existing in unique_passages:
            existing_norm = re.sub(r'\s+', ' ', existing.lower().strip())
            if len(set(normalized.split()) & set(existing_norm.split())) / len(set(normalized.split() + existing_norm.split())) > 0.8:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_passages.append(passage)
    return unique_passages

# Function to combine passages into coherent context
def build_context(passages: List[str], max_length: int = 6000) -> str:
    unique_passages = deduplicate_content(passages)
    context = " ".join(unique_passages)
    if len(context) > max_length:
        context = context[:max_length]
    return context

# Function to detect if a user is seeking spiritual guidance or help
def detect_spiritual_guidance_request(user_message: str) -> bool:
    guidance_keywords = [
        "help me", "struggling", "guidance", "advice", "prayer", "spiritual", 
        "faith", "devotion", "strength", "journey", "seeking", "lost", 
        "direction", "purpose", "meaning", "devotional", "practice", 
        "discipline", "grow", "closer to god", "relationship with god",
        "routine", "daily", "habit", "worship", "meditation", "study",
        "disciple", "troubled", "anxious", "worried", "fear", "doubt",
        "confused", "how can i", "need to", "should i", "want to"
    ]
    
    message_lower = user_message.lower()
    
    # Check if any guidance keywords are present in the user message
    for keyword in guidance_keywords:
        if keyword in message_lower:
            return True
    
    # Check for question format that might indicate seeking guidance
    question_patterns = [
        r"how (do|can|should) i",
        r"what (is|are) the (best|right|ways)",
        r"(can|could) you (help|guide)",
        r"i('m| am) (feeling|trying|wanting|seeking)",
        r"i need (help|guidance|advice)",
        r"i want to (be|become|grow|develop)"
    ]
    
    for pattern in question_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False

# Function to generate spiritual task suggestions based on user message and biblical context
def generate_spiritual_tasks(user_message: str, biblical_context: str) -> Dict:
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Calculate a reasonable due date (14 days from now)
        due_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        
        task_prompt = f"""Based on the user's message and using spiritual context of your knowladge, create a spiritual task to help their journey.

        User's message: "{user_message}"


        Create a spiritual task with:
        1. A clear task name addressing their spiritual need
        2. A brief description of why this practice will help them
        3. Between 3-7 specific subtasks with scripture references
        4. Due date: {due_date}

        Format your response as a valid JSON object following this structure:
        {{
        "task_name": "Title of the spiritual task",
        "task_description": "Description of why this practice will help",
        "subtasks": [
            "First subtask with scripture reference",
            "Second subtask with scripture reference",
            "Third subtask with scripture reference"
        ],
        "due_date": "{due_date}"
        }}
        """

        response = model.generate_content(
            task_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1200
            )
        )
        
        # Extract JSON from the response with improved error handling
        response_text = response.text.strip()
        
        # Find JSON object in the response
        json_match = re.search(r'({[\s\S]*})', response_text)
        if json_match:
            json_str = json_match.group(1)
            try:
                task_data = json.loads(json_str)
                # Validate the task data has all required fields
                required_fields = ["task_name", "task_description", "subtasks", "due_date"]
                if all(field in task_data for field in required_fields):
                    # Ensure subtasks is a list
                    if not isinstance(task_data["subtasks"], list):
                        task_data["subtasks"] = [task_data["subtasks"]]
                    return task_data
                else:
                    print("Missing required fields in task data")
                    return None
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Problematic JSON: {json_str}")
                return None
        else:
            print("Error: Could not find valid JSON in the response")
            return None
    except Exception as e:
        print(f"Error generating spiritual tasks: {str(e)}")
        return None

# Generate response to user query - SEPARATE FROM TASK CREATION
def generate_response(context: str, user_message: str, conversation_history: str, ethical_rules: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response_prompt = f"""You are a wise, thoughtful conversational expert on the Holy Bible. 
        Respond to the user's message with insights, references, and teachings drawn from biblical scripture.

        {conversation_history}

        Biblical context relevant to the current message:
        {context}

        User's current message: "{user_message}"


        Guidelines for your response:
        1. Respond conversationally, with reverence, depth and compassion
        2. Answer their question fully first before mentioning any created tasks
        3. Seamlessly reference content from the Bible
        4. Stay in character as a knowledgeable and compassionate biblical guide
        5. Reference previous conversation points when relevant
        6. Reference scripture or chapter/verse where appropriate
        7. If you don't know something, be honest but remain aligned with biblical principles
        8. Make each answer thoughtful and unique
        9. Use proper markdown formatting for emphasis, headings, and lists where applicable:
        - Use **bold** for emphasis on important points
        - Use *italic* for scripture references
        - Use ## for section headings when organizing longer responses
        - Use - or * for bullet lists when appropriate
        10. Speak with warmth, humility, and wisdom
        11. Keep responses brief, concise and clear also don't overwhelm the user
        12. Keep answers short and well-organized when possible.

        Ethical Rules:
        {ethical_rules}

        Your conversational response:"""

        response = model.generate_content(
            response_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=350
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I apologize, but I'm having trouble responding right now. Please try again with your question."

# Function to append task information to a response
def append_task_info(response: str, task_data: Dict) -> str:
    if not task_data or not response:
        return response
        
    task_info = f"\n\n## Spiritual Task Created\n\nI've created a spiritual task called **{task_data['task_name']}** to help guide your journey. This task includes scripture readings and reflections designed to address your question."
    
    return response + task_info

# Parse task creation command
def parse_task_command(user_message: str) -> Dict[str, str]:
    task_pattern = re.compile(
        r'create task\s*(?:name\s*:\s*"([^"]+)"\s*details\s*:\s*"([^"]+)"\s*due_date\s*:\s*"([^"]+)")',
        re.IGNORECASE
    )
    match = task_pattern.search(user_message)
    if match:
        return {
            "task_name": match.group(1),
            "details": match.group(2),
            "due_date": match.group(3)
        }
    return None

# Chat orchestration - SEPARATED TASK CREATION FROM RESPONSE
def chat_with_bible_bot(user_message: str, ethical_rules: str) -> str:
    conversation_context = memory.get_conversation_context()

    # Check for manual task creation command
    task_info = parse_task_command(user_message)
    if task_info:
        # For backward compatibility - convert old format to new
        subtasks = ["Pray daily for guidance", "Read scripture related to this task", "Reflect on application in your life"]
        response = memory.task_manager.add_task(
            task_info["task_name"],
            task_info["details"],
            subtasks,
            task_info["due_date"]
        )
        memory.add_interaction(user_message, response)
        return response

    # Retrieve biblical passages relevant to the user's message
    passages = retrieve_relevant_documents(user_message, top_k=15)

    if not passages:
        context = "No specific biblical passages found."
    else:
        context = build_context(passages)
    
    # Step 1: Generate the response to the user's question
    response = generate_response(context, user_message, conversation_context, ethical_rules)
    
    # Step 2: Check if the user is seeking spiritual guidance for task creation
    task_data = None
    if detect_spiritual_guidance_request(user_message):
        task_data = generate_spiritual_tasks(user_message, context)
        if task_data and all(k in task_data for k in ["task_name", "task_description", "subtasks", "due_date"]):
            # Add the spiritual task
            task_creation_result = memory.task_manager.add_task(
                task_data["task_name"],
                task_data["task_description"],
                task_data["subtasks"],
                task_data["due_date"]
            )
            # print(f"Task created: {task_creation_result}")
            
            # Step 3: Append task info to the response
            response = append_task_info(response, task_data)

    # Save the interaction to memory
    memory.add_interaction(user_message, response)
    return response

# Enhanced interactive console chat with better Rich support
def interactive_chat():
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.theme import Theme
        from rich.prompt import Prompt
    except ImportError:
        print("\n[ERROR] This application requires the 'rich' library.")
        print("Please install it using: pip install rich\n")
        return

    # Custom theme for the Bible bot
    custom_theme = Theme({
        "user": "bold cyan",
        "bot_name": "bold gold1",
        "scripture": "italic green",
        "command": "bold red",
        "welcome": "bold blue",
        "info": "dim cyan",
        "error": "bold red",
        "task": "bold green",
        "hint": "italic dim white"
    })
    
    console = Console(theme=custom_theme)
    
    # Create task directory if it doesn't exist
    if not os.path.exists("spiritual_tasks"):
        os.makedirs("spiritual_tasks")
    
    # Just start the conversation
    console.print("\n📖 [welcome]Bible Bot[/welcome]", style="gold1")
    
    while True:
        try:
            user_input = Prompt.ask("[user]You[/user]")
            
            if user_input.lower() == "exit":
                console.print("\n[bot_name]Bible Bot:[/bot_name] Goodbye!", style="gold1")
                break
                
            # Show "thinking" indicator
            with console.status("[info]Searching biblical wisdom...[/info]"):
                response = chat_with_bible_bot(user_input, ethical_rules)
            
            console.print()  # Add spacing
            
            # Format the response with proper styling
            console.print("[bot_name]Bible Bot:[/bot_name]")
            
            # Use markdown for all responses
            console.print(Markdown(response))
            console.print()  # Add spacing after response
            
        except KeyboardInterrupt:
            console.print("\n[bot_name]Bible Bot:[/bot_name] Farewell.")
            break
        except Exception as e:
            console.print(f"[error]An error occurred: {str(e)}[/error]")








def process_backend_query(conversation_history: List[Dict[str, str]], user_input: str) -> Dict[str, Any]:
    """
    Process a user query using conversation history and return both the response and any task data.
    
    Args:
        conversation_history: List of conversation turns with role and content
        user_input: Current user message
        
    Returns:
        Dictionary containing the response text and any task data
    """
    # Update memory with conversation history
    if conversation_history:
        for turn in conversation_history:
            if turn["role"] == "user":
                user_message = turn["content"]
            elif turn["role"] == "assistant":
                bot_response = turn["content"]
                # Add previous turns to memory
                memory.add_interaction(user_message, bot_response)
    
    # Retrieve biblical passages relevant to the user's message
    passages = retrieve_relevant_documents(user_input, top_k=15)
    context = build_context(passages) if passages else "No specific biblical passages found."
    
    # Generate the response
    response = chat_with_bible_bot(user_input, ethical_rules)
    
    # Check if this is a guidance request for possible task creation
    task_data = None
    if detect_spiritual_guidance_request(user_input):
        raw_task_data = generate_spiritual_tasks(user_input, context)
        
        # If task data was generated, format it properly for the task manager
        if raw_task_data and all(k in raw_task_data for k in ["task_name", "task_description", "subtasks", "due_date"]):
            # Convert subtasks list to dictionary format
            subtasks_dict = {}
            for i, subtask in enumerate(raw_task_data["subtasks"], 1):
                subtasks_dict[f"task{i}"] = subtask
            
            # Format task data in the expected structure
            task_data = {
                "task_name": raw_task_data["task_name"],
                "task_description": raw_task_data["task_description"],
                "tasks": subtasks_dict,
                "due_date": raw_task_data["due_date"],
                "created_at": datetime.now().strftime("%Y-%m-%d")
            }
    
    # Return both the response and any task data
    return {
        "response": response,
        "task_data": task_data
    }





if __name__ == "__main__":
    # Example usage
    sample_history = [
        {"role": "user", "content": "What does the Bible say about forgiveness?"},
        {"role": "assistant", "content": "The Bible emphasizes forgiveness as essential to the Christian faith. Jesus taught in Matthew 6:14-15 that if we forgive others, our heavenly Father will also forgive us. The Lord's Prayer includes asking God to 'forgive us our debts, as we also have forgiven our debtors.' Colossians 3:13 instructs us to 'Bear with each other and forgive one another if any of you has a grievance against someone. Forgive as the Lord forgave you.'"}
    ]
    
    sample_input = "I'm struggling to forgive someone who hurt me deeply. How can I overcome this?"
    
    # Process the query using the backend integration function
    response = process_backend_query(sample_history, sample_input)
    
    print("=== RESPONSE ===")
    print(f"{response['response']}")
    
    print("\n=== TASK DATA ===")
    if response['task_data']:
        # Pretty print the task data with proper formatting
        print(json.dumps(response['task_data'], indent=2, ensure_ascii=False))
    else:
        print("No task created for this query.")



# # Entry point
# if __name__ == "__main__":
#     interactive_chat()