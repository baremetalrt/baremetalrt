"use client";

export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <div style={{ padding: 40 }}>
      <h2>Something went wrong!</h2>
      <button onClick={() => reset()}>Try again</button>
      <pre style={{ color: 'red', marginTop: 16 }}>{error.message}</pre>
    </div>
  );
}
