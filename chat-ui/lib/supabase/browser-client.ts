import { Database } from "@/supabase/types"
import { createBrowserClient } from "@supabase/ssr"

// Supabase disabled for MVP. Uncomment and configure if Supabase is needed in the future.
// export const supabase = createBrowserClient<Database>(
//   process.env.NEXT_PUBLIC_SUPABASE_URL!,
//   process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
// )
