"use client";

import { useState, useRef, type DragEvent, type ChangeEvent } from "react";
import {
  FileUp,
  Sparkles,
  Loader2,
  ClipboardPaste,
  FileText,
  FileIcon,
  X,
} from "lucide-react";

interface TextInputProps {
  /** Called when user clicks "Summarize Now" with the entered text. */
  onSummarize: (text: string, targetWords: number) => void;
  /** Called when user drops a PDF file. */
  onSummarizePdf?: (file: File, targetWords: number) => void;
  /** Summary returned from the API (displayed below the textarea). */
  summary: string;
  loading: boolean;
}

export default function TextInput({
  onSummarize,
  onSummarizePdf,
  summary,
  loading,
}: TextInputProps) {
  const [text, setText] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [targetWords, setTargetWords] = useState(500);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);

    const file = e.dataTransfer.files?.[0];
    if (!file) {
      const droppedText = e.dataTransfer.getData("text/plain");
      if (droppedText) setText(droppedText);
      return;
    }

    // Handle dropped PDF files
    if (file.type === "application/pdf") {
      setPdfFile(file);
      setText("");
      return;
    }

    // Handle dropped text files
    if (file.type.startsWith("text/")) {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") setText(reader.result);
      };
      reader.readAsText(file);
      return;
    }

    // Handle dropped plain text
    const droppedText = e.dataTransfer.getData("text/plain");
    if (droppedText) setText(droppedText);
  }

  function handleFileSelect(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file && file.type === "application/pdf") {
      setPdfFile(file);
      setText("");
    }
  }

  function handleSubmit() {
    if (loading) return;
    if (pdfFile && onSummarizePdf) {
      onSummarizePdf(pdfFile, targetWords);
      return;
    }
    if (!text.trim()) return;
    onSummarize(text.trim(), targetWords);
  }

  function clearPdf() {
    setPdfFile(null);
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-5 py-4 border-b border-slate-800">
        <FileText className="w-5 h-5 text-brand-400" />
        <h2 className="text-sm font-semibold text-slate-200">
          Document Input
        </h2>
      </div>

      {/* Textarea / Drop Zone */}
      <div className="flex-1 p-4 overflow-hidden flex flex-col gap-3">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`relative flex-1 rounded-xl border transition-all ${
            dragOver
              ? "border-brand-500 bg-brand-500/5"
              : "border-slate-700 bg-surface-raised"
          }`}
        >
          {pdfFile ? (
            <div className="w-full h-full flex flex-col items-center justify-center p-4">
              <FileIcon className="w-12 h-12 text-red-400 mb-3" />
              <p className="text-sm font-medium text-slate-200 mb-1">
                {pdfFile.name}
              </p>
              <p className="text-xs text-slate-500 mb-4">
                {(pdfFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
              <button
                onClick={clearPdf}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-colors"
              >
                <X className="w-3.5 h-3.5" />
                Remove PDF
              </button>
            </div>
          ) : (
            <>
              <textarea
                ref={textareaRef}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste or drop your article / text here..."
                className="w-full h-full resize-none p-4 bg-transparent text-sm text-slate-200 placeholder-slate-600 focus:outline-none custom-scrollbar"
              />

              {/* Empty-state drag prompt */}
              {!text && !dragOver && (
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                  <FileUp className="w-10 h-10 text-slate-700 mb-3" />
                  <p className="text-sm text-slate-600">
                    Drag & drop text or PDF file
                  </p>
                  <label className="mt-3 px-3 py-1.5 rounded-lg text-xs text-brand-400 hover:text-brand-300 hover:bg-brand-500/10 transition-colors cursor-pointer pointer-events-auto">
                    Or click to browse PDF
                    <input
                      type="file"
                      accept=".pdf"
                      className="hidden"
                      onChange={handleFileSelect}
                    />
                  </label>
                </div>
              )}

              {dragOver && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-brand-500/5 rounded-xl pointer-events-none">
                  <ClipboardPaste className="w-10 h-10 text-brand-400 mb-3" />
                  <p className="text-sm text-brand-300">Drop to upload</p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Target words slider */}
        <div className="flex items-center gap-3 px-1">
          <span className="text-xs text-slate-500 whitespace-nowrap">
            Target words:
          </span>
          <input
            type="range"
            min={100}
            max={1500}
            step={50}
            value={targetWords}
            onChange={(e) => setTargetWords(Number(e.target.value))}
            className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-brand-500"
          />
          <span className="text-xs font-medium text-brand-400 w-12 text-right">
            {targetWords}
          </span>
        </div>

        {/* Actions */}
        <button
          onClick={handleSubmit}
          disabled={(!text.trim() && !pdfFile) || loading}
          className="flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-semibold text-white transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-brand-600 hover:bg-brand-500"
          style={{
            boxShadow: text.trim() || pdfFile ? "var(--shadow-glow)" : "none",
          }}
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Sparkles className="w-4 h-4" />
          )}
          {loading
            ? "Summarizing..."
            : pdfFile
            ? "Summarize PDF"
            : "Summarize Now"}
        </button>
      </div>

      {/* Summary Output */}
      {summary && (
        <div className="border-t border-slate-800 p-4">
          <p className="text-xs font-medium text-brand-400 uppercase tracking-wider mb-2">
            Summary
          </p>
          <div className="p-4 rounded-xl text-sm leading-relaxed text-slate-300 bg-surface-raised border border-slate-700 max-h-48 overflow-y-auto custom-scrollbar">
            {summary}
          </div>
        </div>
      )}
    </div>
  );
}
