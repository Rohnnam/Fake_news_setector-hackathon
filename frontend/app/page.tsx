"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { FileText, Link, Undo2, Search } from "lucide-react";
import { cn } from "@/lib/utils";

export default function Home() {
  const [inputType, setInputType] = useState<"text" | "url">("text");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    prediction: string;
    type: "real" | "fake";
    confidence: number;
    processingTime: number;
  } | null>(null);

  const analyzeNews = async () => {
    if (content.length < 5) {
      alert("Please enter more content for analysis");
      return;
    }

    const startTime = performance.now();
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content,
          inputType,
        }),
      });

      if (!response.ok) {
        throw new Error("Analysis failed");
      }

      const data = await response.json();
      const endTime = performance.now();
      const processingTime = Number(((endTime - startTime) / 1000).toFixed(1));

      setResult({
        ...data,
        processingTime,
      });
    } catch (error) {
      alert("An error occurred during analysis. Please try again.");
      console.error("Analysis error:", error);
    } finally {
      setLoading(false);
    }
  };

  const clearInput = () => {
    setContent("");
    setResult(null);
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-200 p-4 md:p-6">
      <div className="mx-auto max-w-3xl">
        <Card className="bg-white/95 backdrop-blur">
          <div className="border-b p-6 text-center">
            <h1 className="text-4xl font-bold text-gray-900">
              Fake News Detector
            </h1>
            <p className="mt-2 text-lg text-gray-600">
              Advanced AI-powered news verification tool
            </p>
          </div>

          <div className="p-6">
            <div className="mb-6 grid grid-cols-2 gap-4">
              <Button
                variant={inputType === "text" ? "default" : "outline"}
                onClick={() => setInputType("text")}
                className="gap-2"
              >
                <FileText className="h-4 w-4" />
                Text
              </Button>
              <Button
                variant={inputType === "url" ? "default" : "outline"}
                onClick={() => setInputType("url")}
                className="gap-2"
              >
                <Link className="h-4 w-4" />
                URL
              </Button>
            </div>

            <Textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder={
                inputType === "text"
                  ? "Paste your news article here..."
                  : "Enter the URL of the news article..."
              }
              className="mb-4 min-h-[200px] resize-y"
            />

            <div className="flex justify-center gap-4">
              <Button onClick={analyzeNews} disabled={loading} className="gap-2">
                <Search className="h-4 w-4" />
                Analyze
              </Button>
              <Button
                variant="secondary"
                onClick={clearInput}
                disabled={loading}
                className="gap-2"
              >
                <Undo2 className="h-4 w-4" />
                Clear
              </Button>
            </div>

            {loading && (
              <div className="mt-8 text-center">
                <div className="mx-auto h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                <p className="mt-2 text-sm text-gray-600">Analyzing content...</p>
              </div>
            )}

            {result && (
              <div className="mt-8 border-t pt-6">
                <Card className="p-6">
                  <div className="mb-6 flex items-center justify-center gap-2 text-2xl font-bold">
                    <div
                      className={cn(
                        "rounded-full p-1",
                        result.type === "real"
                          ? "text-green-500"
                          : "text-red-500"
                      )}
                    >
                      {result.type === "real" ? "✓" : "⚠"}
                    </div>
                    <span>{result.prediction}</span>
                  </div>

                  <div className="mb-6">
                    <Progress value={result.confidence} className="h-6" />
                    <p className="mt-2 text-sm text-gray-600">
                      Confidence Level: {result.confidence}%
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-gray-50 p-4 text-center">
                      <p className="text-sm text-gray-600">Processing Time</p>
                      <p className="text-lg font-semibold">
                        {result.processingTime}s
                      </p>
                    </div>
                    <div className="rounded-lg bg-gray-50 p-4 text-center">
                      <p className="text-sm text-gray-600">Content Length</p>
                      <p className="text-lg font-semibold">
                        {content.length} chars
                      </p>
                    </div>
                  </div>
                </Card>
              </div>
            )}
          </div>

          <div className="border-t p-4 text-center text-sm text-gray-600">
            Powered by ByteRush • Updated 2025
          </div>
        </Card>
      </div>
    </main>
  );
}