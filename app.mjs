import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { config } from "dotenv";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";

config();

const loader = new PDFLoader("./src/documents/Keshav_Gorur_Sriram_Resume.pdf");
const document = await loader.load();

const docSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 10,
});

const splitDocs = await docSplitter.splitDocuments(document);

const embeddings = new OpenAIEmbeddings({
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
    batchSize: 512, //Max is 2048
    model: "text-embedding-3-large",
});

const vectorstore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
const vectorstoreRetriever = vectorstore.asRetriever();

const llm = new ChatOpenAI({
    modelName: "gpt-4o-mini"
})

const prompt = ChatPromptTemplate.fromTemplate(`Answer the user's question: {input} based on the following context {context}`);

const combineDocsChain = await createStuffDocumentsChain({
  llm,
  prompt,
});

const retrievalChain = await createRetrievalChain({
    retriever: vectorstoreRetriever,
    combineDocsChain
});

const question = "Can you summarize the professional experience of Keshav?";
const response = await retrievalChain.invoke({ input: question });

console.log("Query: ", question);
console.log("Response:", response.answer);



