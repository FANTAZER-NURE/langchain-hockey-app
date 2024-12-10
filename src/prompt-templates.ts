import { ChatOpenAI } from '@langchain/openai';
import dotenv from 'dotenv';
import { ChatPromptTemplate } from '@langchain/core/prompts';

dotenv.config();

const model = new ChatOpenAI({
  modelName: 'gpt-4o',
  temperature: 0,
});

const templates = ChatPromptTemplate.fromTemplate(
  'You are an expert hockey coach and need to analyze the following data and give insights about the game/team/player performance depending on the {question} and {data} you receive.'
);

asdfasdfasdfadsfasdfasdfasdfadsfasdfasdfasdfasdfasdfasdfasdfasdfasdf
asdfasdfasdfasdfasdfasdfsadfasdf