// Centralized API URL for backend requests
// Uses the environment variable set at build time (e.g., in Netlify)

export const API_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_URL) {
  throw new Error('NEXT_PUBLIC_API_URL is not set! Please check your environment variables.');
}

// Example usage:
// import { API_URL } from "@/lib/api";
// fetch(`${API_URL}/v1/completions`, { ... });
