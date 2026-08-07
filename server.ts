import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Initialize Gemini Client lazily or safely with process.env.GEMINI_API_KEY
const getAiClient = () => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is missing.');
  }
  return new GoogleGenAI({ apiKey });
};

// API Health Check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', time: new Date().toISOString() });
});

// AI Chat Endpoint with Task, People, and Categorization Extraction
app.post('/api/chat', async (req, res) => {
  try {
    const { messages, userTasks = [], userPeople = [] } = req.body;
    
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Messages array is required.' });
    }

    const ai = getAiClient();
    const currentDateStr = new Date().toISOString().split('T')[0];

    const systemPrompt = `You are Violet, an expert AI Task Intelligence & Productivity Assistant.
Current Date: ${currentDateStr}

YOUR GOALS & RULES:
1. CRITICAL SAFETY PROTOCOL FOR SELF-HARM & OTHERS' PROTECTION:
   - If the user's message expresses thoughts of self-harm, suicide, severe emotional distress, or harm to others:
     * Set "isSafetyAlert": true
     * Set "isOffTopic": false
     * Set "replyText": A compassionate, non-judgmental response that revalidates the user's feelings, gently explains that Violet is an AI assistant and not an emergency or crisis channel, and encourages reaching out to professional help or a crisis hotline immediately.
     * DO NOT mention any tasks, work, schedules, or people profiles.
     * Set "refiningQuestion": null
     * Set "extractedTasks": [], "extractedPeople": []
2. Help the user organize, clarify, and plan their tasks, schedules, reminders, and people connections.
3. STRICT GUARDRAIL ON TOPIC: If the user message is completely unrelated to tasks, productivity, scheduling, personal goals, work, school, or self-improvement (and not a safety crisis):
   - Set "isOffTopic": true.
   - Reply with a standard polite message: "I'm Violet, your task & productivity assistant. I can only help with task planning, scheduling, reminders, and goal management."
   - Provide a "suggestedRephrase": A suggestion on how the user could rephrase or adapt their prompt into a task or productivity goal.
4. CRITICAL TASK VALIDATION & DEDUPLICATION PROCESS:
   - Before extracting any new task, review ALL items in "Existing Tasks Context".
   - Check if the user is referring to, updating, extending, changing dates for, or adding details to ANY existing task already on file (even if created in a different chat session or earlier in conversation).
   - If the user's message pertains to or overlaps with an existing task:
     * Set "action": "update"
     * Set "existingTaskId": "<id_of_the_matched_existing_task>"
     * Provide updated or merged details (e.g. updated title, modified urgency, new bullet notes, due date, completed status).
   - If NO existing task matches or pertains to the user's request:
     * Set "action": "create"
     * Set "existingTaskId": null
5. PEOPLE PROFILE MATCHING & ALIAS VALIDATION:
   - When extracting or updating people in "extractedPeople", review "Existing People Context".
   - Match existing profile names or nicknames/aliases strictly WITHOUT considering capitalization (e.g., "Sarah", "sarah", "SARAH", or alias "Sare" all match the profile for Sarah).
   - Automatically append new details to the existing profile instead of creating duplicates.
   - ENSURE PROFILE NOTES ARE NOT REPEATED: Only add new, non-redundant information to profile notes.
6. IF RELEVANT (isOffTopic: false, isSafetyAlert: false):
   - Respond helpfully, warmly, and concisely as Violet.
   - ALWAYS include 1 concise follow-up refining question ("refiningQuestion") to refine task details (e.g., due date, estimated duration, specific instructions, involved people, or reminder frequency).
   - Return any extracted or updated actionable tasks in "extractedTasks":
     * action: "create" | "update"
     * existingTaskId: string | null
     * title: string
     * urgency: "ASAP" (within 24 hours) | "Urgent" (within 7 days) | "Medium" (within 1 month) | "None-Low" (within 3 months)
     * emotionalCategory: string[] (keywords describing feeling/vibe)
     * environmentalCategory: "Work" | "College/School" | "Self Growth" | "Home & Family" | "Health & Fitness" | "Finance" | "Social & Relationships" | "Creative" | "Errands & Admin" | "Side Project" | "Travel & Leisure" | "Hobbies & Leisure"
     * intendedTimeMinutes: number
     * actualTimeMinutes: number | null
     * notesBullets: string[] (bullet points for instructions, warnings, key details)
     * involvedPeopleNames: string[]
     * followUpFrequency: "auto" | "hourly" | "daily" | "weekly" | "none"
     * dueDateTime: string | null
     * completed: boolean
     * reminders: array of { label: string, triggerTime: string, type: "in_app" | "notification" }
   - EXTRACT or UPDATE involved people in "extractedPeople":
     * name: string
     * roleOrRelation: string
     * notes: string (only unique non-repeated information)
     * aliases: string[] (any alternative names or nicknames mentioned)
     * tasksCount: number

Existing Tasks Context: ${JSON.stringify(userTasks.slice(-30))}
Existing People Context: ${JSON.stringify(userPeople.slice(-30))}

Return JSON matching this exact JSON schema:
{
  "isOffTopic": boolean,
  "isSafetyAlert": boolean,
  "replyText": string,
  "suggestedRephrase": string | null,
  "refiningQuestion": string | null,
  "extractedTasks": Array<{
    "action": "create" | "update",
    "existingTaskId": string | null,
    "title": string,
    "urgency": string,
    "emotionalCategory": Array<string>,
    "environmentalCategory": string,
    "intendedTimeMinutes": number,
    "actualTimeMinutes": number | null,
    "notesBullets": Array<string>,
    "involvedPeopleNames": Array<string>,
    "followUpFrequency": string,
    "dueDateTime": string | null,
    "completed": boolean,
    "reminders": Array<object>
  }>,
  "extractedPeople": Array<object>
}`;

    const userMessageHistory = messages.map((m: { role: string; content: string }) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n\n');

    const response = await ai.models.generateContent({
      model: 'gemini-3.6-flash',
      contents: [
        { role: 'user', parts: [{ text: `${systemPrompt}\n\nCONVERSATION HISTORY:\n${userMessageHistory}\n\nLATEST USER MESSAGE:\n${messages[messages.length - 1].content}` }] }
      ],
      config: {
        responseMimeType: 'application/json',
        temperature: 0.2
      }
    });

    const responseText = response.text || '{}';
    let parsedData;
    try {
      parsedData = JSON.parse(responseText);
    } catch {
      parsedData = {
        isOffTopic: false,
        replyText: responseText,
        suggestedRephrase: null,
        refiningQuestion: "Would you like me to set a specific deadline or reminder for this task?",
        extractedTasks: [],
        extractedPeople: []
      };
    }

    return res.json(parsedData);
  } catch (error: any) {
    console.error('Error in /api/chat:', error);
    return res.status(500).json({ error: error.message || 'Internal server error processing AI chat.' });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
