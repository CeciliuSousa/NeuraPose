import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/:path*', // Proxy to Backend
      },
      {
        source: '/docs',
        destination: 'http://127.0.0.1:8000/docs', // Proxy docs
      },
      {
        source: '/openapi.json',
        destination: 'http://127.0.0.1:8000/openapi.json', // Proxy openapi
      }
    ];
  },
};

export default nextConfig;
