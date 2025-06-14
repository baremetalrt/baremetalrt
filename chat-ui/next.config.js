const withPWA = require("next-pwa")({
  dest: "public"
});

const isAnalyze = process.env.ANALYZE === "true";
const withBundleAnalyzer = isAnalyze
  ? require("@next/bundle-analyzer")({ enabled: true })
  : (config) => config;

module.exports = withBundleAnalyzer(
  withPWA({
    reactStrictMode: true,
    images: {
      remotePatterns: [
        {
          protocol: "http",
          hostname: "localhost"
        },
        {
          protocol: "http",
          hostname: "127.0.0.1"
        },
        {
          protocol: "https",
          hostname: "**"
        }
      ]
    },
    experimental: {
      serverComponentsExternalPackages: ["sharp", "onnxruntime-node"]
    }
  })
)
