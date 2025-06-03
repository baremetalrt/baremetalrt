"use client";

export default function GlobalError({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <html>
      <body>
        <h2>Global Error</h2>
        <button onClick={() => reset()}>Try again</button>
        <pre style={{ color: 'red', marginTop: 16 }}>{error.message}</pre>
      </body>
    </html>
  );
}
