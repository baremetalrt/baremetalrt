import { API_URL } from "./api";

export async function fetchCompletion(prompt: string) {
  const response = await fetch(`${API_URL}/v1/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      max_tokens: 128,
      temperature: 0.7,
      top_p: 0.95,
    }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  return response.json();
}
