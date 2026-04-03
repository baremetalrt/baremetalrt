# Supabase Integration Note

As of the MVP deployment, Supabase is disabled to allow successful frontend builds and deployment. The export in `chat-ui/lib/supabase/browser-client.ts` is currently stubbed as:

```ts
export const supabase = undefined as any;
```

If you need to enable Supabase in the future:
1. Uncomment and restore the original export in `browser-client.ts`:
   ```ts
   export const supabase = createBrowserClient<Database>(
     process.env.NEXT_PUBLIC_SUPABASE_URL!,
     process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
   )
   ```
2. Set the `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` environment variables in your deployment platform (e.g., Netlify).

See project memory for more details.
