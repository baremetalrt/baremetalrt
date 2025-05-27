import { createClient } from '@supabase/supabase-js';

const useSupabase = process.env.NEXT_PUBLIC_USE_SUPABASE === 'true';

let supabase: any;

if (useSupabase) {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
  supabase = createClient(supabaseUrl, supabaseAnonKey);
} else {
  // Minimal stub implementation for build/dev
  supabase = {
    from: () => ({
      select: () => ({
        eq: () => ({
          single: async () => ({ data: { id: 'stub-id' }, error: null }),
          maybeSingle: async () => ({ data: { id: 'stub-id' }, error: null }),
        }),
      }),
      insert: () => ({
        select: async () => ({ data: [{ id: 'stub-id' }], error: null }),
      }),
      delete: () => ({
        eq: () => ({
          single: async () => ({ data: null, error: null }),
        }),
      }),
    }),
    auth: {
      getSession: async () => ({ data: { session: null } }),
      updateUser: async () => ({ data: null, error: null }),
      exchangeCodeForSession: async () => ({ data: null, error: null }),
      getUser: async () => ({ data: { user: null } }),
    },
  };
}

export { supabase };
