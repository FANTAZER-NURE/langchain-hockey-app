// src/server.ts
import express from 'express';
import bodyParser from 'body-parser';
import dotenv from 'dotenv';
import { ChatOpenAI, OpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { JsonOutputParser } from '@langchain/core/output_parsers';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { Document } from '@langchain/core/documents';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { RunnableSequence } from '@langchain/core/runnables';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever';

// Initialize environment variables
dotenv.config();

// Constants
const PORT = process.env.PORT || 3000;
const parser = new JsonOutputParser();

// Initialize Express app
const app = express();

// Middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Initialize OpenAI model
const model = new ChatOpenAI({
  modelName: 'gpt-4',
  temperature: 0,
});

// Test data
const json = {
  home: {
    AS: 38,
    BkS: 5,
    DZFOL: 11,
    DZFOW: 14,
    EN: 6.196587870967505,
    ES: 50.10000000000002,
    ESAS: 22,
    ESG: 0,
    ESSOG: 8,
    ES_HPSA: 4,
    ES_HPSA_OnGoal: 2,
    FOL: 27,
    FOW: 34,
    FO_perc: 55.73770491803278,
    FO_tot: 61,
    G: 1,
    HPSA: 11,
    HPSA_OnGoal: 8,
    IceTime: 3600,
    MS: 15,
    OZFOL: 8,
    OZFOW: 11,
    PIM: 15,
    PP: 321.9900000000001,
    PPAS: 14,
    PPG: 1,
    PPSOG: 8,
    PP_HPSA: 7,
    PP_HPSA_OnGoal: 6,
    PS: 0,
    PS_Failure: 0,
    PS_Success: 0,
    SH: 424.1267828709673,
    SHAS: 2,
    SHG: 0,
    SHSOG: 2,
    SH_HPSA: 0,
    SH_HPSA_OnGoal: 0,
    SOG: 18,
    XG_AS: 1.683239493339541,
    XG_ESAS: 0.5093019173512995,
    XG_PPAS: 1.1113164094729213,
  },
  away: {
    AS: 93,
    BkS: 20,
    DZFOL: 11,
    DZFOW: 8,
    EA: 6.196587870967505,
    ES: 50.10000000000002,
    ESAS: 76,
    ESG: 3,
    ESSOG: 43,
    ES_HPSA: 19,
    ES_HPSA_OnGoal: 12,
    FOL: 34,
    FOW: 27,
    FO_perc: 44.26229508196721,
    FO_tot: 61,
    G: 4,
    HPSA: 23,
    HPSA_OnGoal: 14,
    IceTime: 3600,
    MS: 19,
    OZFOL: 14,
    OZFOW: 11,
    PIM: 13,
    PP: 424.1267828709673,
    PPAS: 17,
    PPG: 1,
    PPSOG: 11,
    PP_HPSA: 4,
    PP_HPSA_OnGoal: 2,
    PS: 0,
    PS_Failure: 0,
    PS_Success: 0,
    SH: 321.9900000000001,
    SHAS: 0,
    SHG: 0,
    SHSOG: 0,
    SH_HPSA: 0,
    SH_HPSA_OnGoal: 0,
    SOG: 54,
    XG_AS: 3.6257644182741395,
    XG_ESAS: 2.6493934613270995,
    XG_PPAS: 0.97637095694704,
  },
};

// Create two separate prompt templates - one for analysis and one for chat
// const chatPrompt = ChatPromptTemplate.fromTemplate(`
//   You are an expert hockey analyst. Your task is to:
//   1. If the user asks about specific statistics or terms, explain them clearly
//   2. Keep responses conversational while maintaining analytical depth
//   3. Reference previous conversations from the chat history when relevant
//   4. Be concise in your responses, don't make too much text, if users asks the direct question, answer with only the requested information.

//   User Input: {input}
//   Retrieved Context: {context}
//   Previous Conversation: {chat_history}
// `);

// const gameAnalysisPrompt = ChatPromptTemplate.fromTemplate(`
//   You are an expert hockey analyst. Analyze the provided game data and return a JSON response.

//   Game Analysis Required Format:
//   {{
//     "game_summary": {{
//       "score": {{
//         "home": number,
//         "away": number
//       }},
//       "shots_on_goal": {{
//         "home": number,
//         "away": number
//       }},
//       "key_stats": {{
//         "home": {{
//           "power_play_goals": number,
//           "power_play_opportunities": number,
//           "faceoff_percentage": number
//         }},
//         "away": {{
//           "power_play_goals": number,
//           "power_play_opportunities": number,
//           "faceoff_percentage": number
//         }}
//       }},
//       "analysis": {{
//         "dominant_team": string,
//         "key_factors": [string],
//         "notable_statistics": [string],
//         "overall_analysis": string
//       }}
//     }}
//   }}

//   Format Instructions: {format_instructions}
//   Game Data: {data}
// `);

// Analyze the following hockey game data and provide insights.
//                 Return ONLY a JSON object with no additional text or formatting.
//               Provide detailed game analysis as text in the analysis.overall_analysis field.

// Create prompt template for vector store retrieve
const gameAnalysisPrompt = ChatPromptTemplate.fromTemplate(`
  You are an expert hockey analyst. Your task is to:
  1. If the user asks about specific statistics or terms, explain them clearly using the game context
  2. If user asks about the game, analyze relevant aspects of the provided game data
  3. Keep responses conversational while maintaining analytical depth
  4. Use the game data to provide specific examples when explaining statistics
  5. Be concise in your responses, don't make too much text, if users asks the direct question, answer with only the requested information.
  6. If the user asks about the game, analyze relevant aspects of the provided game data in the following format:

    Game Analysis Required Format:
    {{
      "game_summary": {{
        "score": {{
          "home": number,
          "away": number
        }},
        "shots_on_goal": {{
          "home": number,
          "away": number
        }},
        "key_stats": {{
          "home": {{
            "power_play_goals": number,
            "power_play_opportunities": number,
            "faceoff_percentage": number
          }},
          "away": {{
            "power_play_goals": number,
            "power_play_opportunities": number,
            "faceoff_percentage": number
          }}
        }},
        "analysis": {{
          "dominant_team": string,
          "key_factors": [string],
          "notable_statistics": [string],
          "overall_analysis": string
        }}
      }}
    }}
  

  User Input: {input}
  Game Context: This is the current game data that you can reference to provide specific examples: {data}
  
  Instructions:
  - When explaining statistics, use specific examples from the current game
  - If asked about game performance, provide detailed analysis
  - Keep responses conversational but informative
  - Reference previous conversations when relevant

  Retrieved Context: {context}
  Previous Conversation: {chat_history}
  Format Instructions: {format_instructions}
`);

// Create analysis chain
// const analysisChain = gameAnalysisPrompt.pipe(model).pipe(parser);

// const documentA = new Document({
//   pageContent:
//     'Use this website to find the meaning of the stats keys in the data https://help.sportcontract.net/terms',
// });

const fetchContextDocument = async () => {
  // Load the website
  const loader = new CheerioWebBaseLoader(
    'https://help.sportcontract.net/terms',
    {
      // optional params: ...
    }
  );

  // Split the website into chunks
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const docs = await loader.load();
  const splitDocs = await splitter.splitDocuments(docs);
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  return vectorStore;
};

const retrieveContextDocumentChain = async (
  chain: RunnableSequence,
  vectorStore: MemoryVectorStore
) => {
  const retriever = vectorStore.asRetriever();

  const retrieverPrompt = ChatPromptTemplate.fromTemplate(
    `Given the following conversation history:
    {chat_history}
    
    And the following input: {input}
    
    Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.`
  );

  const historyRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever: retriever,
    rephrasePrompt: retrieverPrompt,
  });

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: historyRetriever,
  });

  return retrievalChain;
};

// API Routes
app.post('/', async (req, res) => {
  const input = typeof req.body === 'string' ? req.body : req.body.input;

  console.log('Received input:', input);

  try {
    const chatChain = await createStuffDocumentsChain({
      llm: model,
      prompt: gameAnalysisPrompt,
    });

    const vectorStore = await fetchContextDocument();
    const retrievalChain = await retrieveContextDocumentChain(
      chatChain,
      vectorStore
    );

    // Format chat history as messages array
    const chatHistory = [
      new HumanMessage('Hi, my name is John'),
      new AIMessage('Hi, John! How can I help you today?'),
      new HumanMessage('What is the meaning of the stats key TOI?'),
      new AIMessage(
        'TOI stands for Time On Ice. It is the total amount of time a player is on the ice during a game.'
      ),
    ];

    // Wrap the chain execution in a traced run
    const response = await retrievalChain.invoke(
      {
        input: input || '',
        data: JSON.stringify(json),
        context:
          'Use the retrieved documents to understand the meaning of hockey statistics.',
        chat_history: chatHistory,
        format_instructions: parser.getFormatInstructions(),
      },
      {
        tags: ['hockey-analysis'], // Add tags for filtering in the UI
        metadata: {
          userId: req.body.userId || 'anonymous',
          requestType: 'hockey-analysis',
          inputType: input.includes('analysis')
            ? 'game-analysis'
            : 'stat-explanation',
        },
      }
    );

    res.json({
      type: 'explanation',
      content: response,
      game_data: json,
    });
  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({
      error: 'Failed to process request',
      details: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
