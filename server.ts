import express from "express";
import path from "path";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "5mb" }));

// Helper to initialize GoogleGenAI safely and lazily
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is required. Please set it in Settings > Secrets.");
    }
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// 1. AI Chatbot API
app.post("/api/gemini/chat", async (req, res) => {
  try {
    const { message, history, contextData } = req.body;
    if (!message) {
      return res.status(400).json({ error: "Message is required." });
    }

    const ai = getGeminiClient();

    // Construct detailed context for the AI so it knows the user's actual finance details!
    const contextPrompt = contextData
      ? `\n[USER FINANCIAL DATA CONTEXT]
Current Balance: ${contextData.currency || "₹"}${contextData.balance}
Total Income: ${contextData.currency || "₹"}${contextData.totalIncome}
Total Expenses: ${contextData.currency || "₹"}${contextData.totalExpenses}
Savings: ${contextData.currency || "₹"}${contextData.savings}
Financial Health Score: ${contextData.healthScore}/100
Active Budgets: ${JSON.stringify(contextData.budgets)}
Active Goals: ${JSON.stringify(contextData.goals)}
Recent Transactions: ${JSON.stringify(contextData.transactions)}
`
      : "";

    const systemInstruction = `You are SpendWise AI, an expert elite FinTech advisor and financial chatbot.
Your job is to provide highly precise, insightful, and actionable financial advice to users.
Keep your answers visually stunning using Markdown, emojis, clean lists, and bold headers where appropriate.
Always be polite, professional, encouraging, and deeply knowledgeable about personal finance, savings, investment, and budgeting.
Do not make generic comments; instead, reference the user's actual numbers from the context when they ask about their budget, savings, or spending!
If the user asks if they can afford something, analyze their savings and budget to give a definitive "Yes, and here's why..." or "No, but you could if you..." response.
Always format currency symbols correctly based on what the user uses (defaulting to ₹ if not specified).

${contextPrompt}`;

    // Format the history for the GoogleGenAI SDK generateContent
    // Each history item is: { role: 'user' | 'model', parts: [{ text: string }] }
    const formattedHistory = (history || []).map((h: any) => ({
      role: h.role === "assistant" ? "model" : "user",
      parts: [{ text: h.content || h.text || "" }],
    }));

    // Add the current user prompt to contents
    const contents = [...formattedHistory, { role: "user", parts: [{ text: message }] }];

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents,
      config: {
        systemInstruction,
        temperature: 0.7,
      },
    });

    res.json({ text: response.text });
  } catch (error: any) {
    console.error("Error in AI Chat:", error);
    res.status(500).json({ error: error.message || "Something went wrong with SpendWise AI Chat." });
  }
});

// 2. AI Advisor (Batch spending analysis and report generator)
app.post("/api/gemini/analyze", async (req, res) => {
  try {
    const { userData, transactions, budgets, goals } = req.body;
    const ai = getGeminiClient();

    const analysisPrompt = `Analyze the following user's financial profile and recent transaction history.
Provide a high-quality, professional, and personalized FinTech analysis in JSON format containing 3-4 specific high-impact tips, a motivational message, and a predicted next month's total expense.

User Context:
- Monthly Income: ${userData.currency || "₹"}${userData.salary || 0}
- Current Savings: ${userData.currency || "₹"}${userData.savings || 0}
- Budgets Set: ${JSON.stringify(budgets)}
- Goals Set: ${JSON.stringify(goals)}
- Recent Transactions: ${JSON.stringify(transactions)}

You MUST return a JSON object ONLY matching this schema:
{
  "summary": "A 2-3 sentence overview of their financial health.",
  "healthScoreExplanation": "A short brief on why their financial health score is what it is.",
  "smartTips": [
    {
      "category": "e.g., Food / Subscriptions / Savings",
      "text": "Specific tip e.g. You spent 42% of your income on food. Reduce restaurant expenses to save ₹2500 monthly.",
      "severity": "high" | "medium" | "low",
      "impactValue": "Estimated savings amount as a string, e.g. '₹2,500'"
    }
  ],
  "motivationalQuote": "A short, highly inspiring and modern quote to help them stay disciplined.",
  "projectedExpenses": "Estimated expense amount for next month based on spending habits as a string, e.g. '₹12,400'"
}`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: analysisPrompt,
      config: {
        responseMimeType: "application/json",
        temperature: 0.4,
      },
    });

    const resultText = response.text || "{}";
    res.json(JSON.parse(resultText.trim()));
  } catch (error: any) {
    console.error("Error in AI Analyze:", error);
    res.status(500).json({ error: error.message || "Failed to analyze data using SpendWise AI." });
  }
});

// Serve Vite middleware in development
async function setupServer() {
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[SpendWise AI Backend] Server successfully running at http://localhost:${PORT}`);
  });
}

setupServer();
