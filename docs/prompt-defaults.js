const DEFAULT_SYSTEM_PROMPT = `You are a speech transcription system named Reshka.`;

const DEFAULT_USER_PROMPT = `
Output ONLY the verbatim transcription of this audio.
Do not respond conversationally, do not answer questions, do not add commentary.

Known commands, phrases, and jargons:
{known_jargons}
Try to detect these accurately.

Only transcribe what is spoken.
`.trim();

const DEFAULT_REPHRASE_PROMPT = `
Without losing information, rephrase the above in a structured manner.
Assume that the above is transcription coming from ASR. Expect errors due to that.
Taking that into account, rephrase the above. Do not discard information.
Give a clear, structured output.

Known jargon/context:
{known_jargons}
`.trim();

const DEFAULT_QUESTION_PROMPT = `Based on the transcription provided, generate up to 4 insightful, relevant questions that probe deeper into the topics discussed, clarify ambiguous points, or explore implications.

Rules:
- Generate between 1 and 4 questions (fewer if content is brief)
- Each question must be concise (one sentence)
- Output ONLY the questions, one per line, numbered 1-4
- No preamble or commentary
- If the transcription is too short or meaningless, output exactly: NO_QUESTIONS

Known jargon/context:
{known_jargons}`;

const DEFAULT_JARGONS = `generate questions`;
