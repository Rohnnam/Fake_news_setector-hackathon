import { NextResponse } from "next/server";
import { PythonShell } from "python-shell";
import path from "path";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { content, inputType } = body;

    // Create PythonShell options
    const options = {
      mode: "json",
      pythonPath: "python3",
      scriptPath: path.join(process.cwd(), "scripts"),
      pythonOptions: ["-u"], // unbuffered output
    };

    // Run Python script
    const result = await new Promise((resolve, reject) => {
      PythonShell.run("analyze.py", options)
        .then((messages) => {
          resolve(messages[0]); // Get the first (and only) message
        })
        .catch((err) => reject(err));
    });

    return NextResponse.json(result);
  } catch (error) {
    console.error("Analysis error:", error);
    return NextResponse.json(
      { error: "Failed to analyze content" },
      { status: 500 }
    );
  }
}