// Wire types mirroring the backend's response DTOs.

export type SourceType = "note" | "url";

export interface Item {
  id: string;
  source_type: SourceType;
  title: string;
  preview: string;
  source_url: string | null;
  created_at: string;
}

export interface Citation {
  item_id: string;
  title: string;
  source_type: SourceType;
  source_url: string | null;
  snippet: string;
}

export interface Answer {
  question: string;
  answer: string;
  citations: Citation[];
}

export interface ItemDetail extends Item {
  content: string;
}

export interface IngestResult {
  item: Item;
  chunks_created: number;
}

// A single turn in the chat thread; `error` flags a failed turn.
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  error?: boolean;
}
