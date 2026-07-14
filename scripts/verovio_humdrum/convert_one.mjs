#!/usr/bin/env node
/**
 * Convert one Humdrum file to MIDI using the Verovio Humdrum toolkit (VHV engine).
 *
 * Usage:
 *   node convert_one.mjs input.hum output.mid
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import createVerovioModule from "verovio/wasm-hum";
import { VerovioToolkit } from "verovio/esm";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function decodeBase64Midi(base64) {
  return Buffer.from(base64, "base64");
}

async function convertHumToMidi(inputPath, outputPath) {
  const humdrum = fs.readFileSync(inputPath, "utf8");
  const VerovioModule = await createVerovioModule();
  const toolkit = new VerovioToolkit(VerovioModule);

  toolkit.setOptions({
    inputFrom: "humdrum",
    breaks: "none",
  });

  const loaded = toolkit.loadData(humdrum);
  if (!loaded) {
    throw new Error(`Verovio failed to load Humdrum data: ${inputPath}`);
  }

  const midiBase64 = toolkit.renderToMIDI({ breaks: "none" });
  if (!midiBase64) {
    throw new Error(`Verovio returned empty MIDI for: ${inputPath}`);
  }

  const midiBytes = decodeBase64Midi(midiBase64);
  if (midiBytes.length < 14) {
    throw new Error(`MIDI output too small (${midiBytes.length} bytes): ${inputPath}`);
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, midiBytes);
  toolkit.destroy();
}

async function main() {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];
  if (!inputPath || !outputPath) {
    console.error(`Usage: node ${path.basename(process.argv[1])} input.hum output.mid`);
    process.exit(2);
  }
  await convertHumToMidi(path.resolve(inputPath), path.resolve(outputPath));
}

main().catch((err) => {
  console.error(err.message || err);
  process.exit(1);
});
