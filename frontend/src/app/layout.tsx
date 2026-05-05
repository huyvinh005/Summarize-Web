import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Summarizer",
  description: "AI-powered text summarization and conversational Q&A",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
