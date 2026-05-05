/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
      },
      colors: {
        brand: {
          50:  "#eff6ff",
          100: "#dbeafe",
          200: "#bfdbfe",
          300: "#93c5fd",
          400: "#60a5fa",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
        },
        surface: {
          DEFAULT: "#0f172a",  /* slate-900 */
          raised:  "#1e293b",  /* slate-800 */
          overlay: "#334155",  /* slate-700 */
        },
      },
      boxShadow: {
        glow: "0 0 40px rgba(59, 130, 246, 0.15)",
        card: "0 4px 24px rgba(0, 0, 0, 0.25)",
      },
    },
  },
  plugins: [],
};
