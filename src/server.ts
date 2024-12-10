// src/server.ts
import express from 'express';
import bodyParser from 'body-parser';
import dotenv from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';
import { JsonOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { Document } from '@langchain/core/documents';
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'



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

// Create prompt template
const gameAnalysisPrompt = ChatPromptTemplate.fromTemplate(`
  Analyze the following hockey game data and provide insights.
  Return ONLY a JSON object with no additional text or formatting.
  Provide detailed game analysis as text in the analysis.overall_analysis field.
  
  Required format:
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

  Format Instructions: {format_instructions}
  Context: {context}
  Game Data: {data}
`);

// Create analysis chain
// const analysisChain = gameAnalysisPrompt.pipe(model).pipe(parser);

// const documentA = new Document({
//   pageContent:
//     'Use this website to find the meaning of the stats keys in the data https://help.sportcontract.net/terms',
// });

const loader = new CheerioWebBaseLoader(
  'https://help.sportcontract.net/terms',
  {
    // optional params: ...
  }
);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20,
});

// API Routes
app.get('/', async (req, res) => {
  try {
    const chain2 = await createStuffDocumentsChain({
      llm: model,
      prompt: gameAnalysisPrompt,
      outputParser: parser,
    });

    const docs = await loader.load();
    const splitDocs = await splitter.splitDocuments(docs);

    console.log(splitDocs);

    const response = await chain2.invoke({
      format_instructions: parser.getFormatInstructions(),
      data: JSON.stringify(json),
      // context:
      //   'Use this website to find the meaning of the stats keys in the data https://help.sportcontract.net/terms',
      context: [...docs],
    });

    // Ensure we're returning a proper JSON object
    if (typeof response === 'string') {
      res.json(JSON.parse(response));
    } else {
      res.json(response);
    }
  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({
      error: 'Failed to analyze game data',
      details: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
